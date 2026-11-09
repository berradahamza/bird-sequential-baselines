"""Frozen BIRD catalogue and visitor split; one causal prefix per next POI."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    """Atomic writes keep an interrupted process from corrupting a report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class BirdData:
    """IDs: 0=padding, 1..N=artworks, N+1=augmentation mask (never a target)."""

    def __init__(self, directory, max_history=32, split_seed=2022):
        self.directory = Path(directory)
        if max_history < 1:
            raise ValueError("max_history must be positive")
        self.max_history = max_history
        self.split_seed = int(split_seed)
        names = ("trajectories_poi.json", "artworks_semantics.json", "split.json", "data_audit.json")
        self.hashes = {name: sha256(self.directory / name) for name in names}
        manifest = self.directory.parent / "manifest.json"
        if manifest.exists() and read_json(manifest) != self.hashes:
            raise ValueError("Data differ from data/manifest.json; use the original packaged JSONs.")
        trajectories = read_json(self.directory / names[0])
        catalogue = read_json(self.directory / names[1])
        # These five immutable partitions reproduce STKG_CLL_POI/features.py::visitor_split.
        # Recompute and verify them, rather than silently trusting split_seed or a CSV.
        split_path = self.directory.parent.parent / "splits" / f"split_{self.split_seed}.json"
        split_doc = read_json(split_path)
        if split_doc["split_seed"] != self.split_seed:
            raise ValueError("Split seed mismatch")
        self.split_file_hash = sha256(split_path)
        self.split = split_doc["visitors"]
        import random
        expected_ids = sorted(trajectories["visitors"])
        random.Random(self.split_seed).shuffle(expected_ids)
        held = max(1, round(len(expected_ids) * 0.15))
        expected = {"train": expected_ids[:-2*held],
                    "validation": expected_ids[-2*held:-held],
                    "test": expected_ids[-held:]}
        if self.split != expected:
            raise ValueError("Split not identical to CLL visitor_split algorithm")
        if self.split_seed == 2022 and self.split != read_json(self.directory / names[2])["visitors"]:
            raise ValueError("Frozen 2022 CLL split differs from data/processed/split.json")
        self.poi_ids = sorted(catalogue["items"])
        self.index = {poi: i + 1 for i, poi in enumerate(self.poi_ids)}
        self.mask_id = len(self.poi_ids) + 1
        visitors = trajectories["visitors"]
        if set(self.split) != {"train", "validation", "test"}:
            raise ValueError("Expected train/validation/test split")
        flat = [v for ids in self.split.values() for v in ids]
        if len(flat) != len(set(flat)) or set(flat) != set(visitors):
            raise ValueError("Split must cover each visitor exactly once")
        self.examples = {}
        for fold, ids in self.split.items():
            rows = []
            for visitor in ids:
                events = visitors[visitor]["events"]
                times = [e["t_start_seconds"] for e in events]
                if any(not math.isfinite(t) or t < 0 for t in times) or any(b <= a for a, b in zip(times, times[1:])):
                    raise ValueError(f"Invalid start times for {visitor}")
                seq = [self.index[e["poi_id"]] for e in events]
                for i in range(len(seq) - 1):
                    rows.append({"visitor": visitor, "event_index": i,
                                 "history": seq[max(0, i + 1 - max_history):i + 1],
                                 "target": seq[i + 1]})
            if not rows:
                raise ValueError(f"Empty split: {fold}")
            self.examples[fold] = rows

    def summary(self):
        return {"catalogue_size": len(self.poi_ids), "max_history": self.max_history,
                "split_seed": self.split_seed, "split_file_sha256": self.split_file_hash,
                "visitors": {k: len(v) for k, v in self.split.items()},
                "prefixes": {k: len(v) for k, v in self.examples.items()},
                "processed_input_sha256": self.hashes}


def pack_histories(histories, width=None):
    """Right-padding preserves position zero and avoids fully masked attention rows."""
    lengths = torch.tensor([len(h) for h in histories], dtype=torch.long)
    if not len(histories) or int(lengths.min()) < 1:
        raise ValueError("Empty histories are not supported")
    width = width or int(lengths.max())
    if int(lengths.max()) > width:
        raise ValueError("History exceeds requested width")
    ids = torch.zeros((len(histories), width), dtype=torch.long)
    for i, history in enumerate(histories):
        ids[i, :len(history)] = torch.tensor(history, dtype=torch.long)
    return ids, lengths


def batch_tensors(rows, device):
    ids, lengths = pack_histories([r["history"] for r in rows])
    targets = torch.tensor([r["target"] - 1 for r in rows], dtype=torch.long)
    return ids.to(device), lengths.to(device), targets.to(device)
