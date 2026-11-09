"""CLI for BIRD / STKG_CLL_POI visitor-disjoint five-split comparisons."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .data import BirdData, read_json
from .engine import baselines, evaluate_test, report, runtime, train

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = dict(hidden=64, dropout=0.2, max_history=32, blocks=2, heads=2,
                batch_size=64, epochs=30, patience=5, lr=.001, weight_decay=.00005,
                seed=2022, bootstrap=1000, log_every=40)


def validate(c):
    for key in ("hidden", "max_history", "blocks", "heads", "batch_size", "epochs", "patience", "log_every"):
        if c[key] < 1:
            raise ValueError(f"{key} must be positive")
    if c["hidden"] % c["heads"]:
        raise ValueError("hidden must be divisible by heads")
    if not 0 <= c["dropout"] < 1 or c["lr"] <= 0 or c["weight_decay"] < 0 or c["bootstrap"] < 0:
        raise ValueError("Invalid optimization parameters")


def main():
    parser = argparse.ArgumentParser(description="BIRD CLL-aligned GRU4Rec, SASRec, Markov-1/2 and popularity")
    sub = parser.add_subparsers(dest="command", required=True)

    def shared(p):
        p.add_argument("--data-dir", type=Path, default=ROOT / "data/processed")
        p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        p.add_argument("--threads", type=int, default=4)

    p = sub.add_parser("doctor", help="Check environment and the requested exact CLL split")
    shared(p)
    p.add_argument("--split-seed", type=int, default=2022)

    p = sub.add_parser("train", help="Train; choose by validation MRR; no test evaluation")
    shared(p)
    p.add_argument("--model", choices=("all", "gru4rec", "sasrec"), default="all")
    p.add_argument("--run-root", type=Path, default=ROOT / "runs/split2022/seed2022")
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--split-seed", type=int, default=2022)
    p.add_argument("--config", type=Path)
    for key, value in DEFAULTS.items():
        p.add_argument("--" + key.replace("_", "-"), type=type(value), default=None)

    p = sub.add_parser("resume", help="Resume unfinished run from last.pt (same config & split)")
    shared(p)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=30)

    p = sub.add_parser("evaluate-test", help="Read best frozen checkpoint, evaluate test once")
    shared(p)
    p.add_argument("--run-dir", type=Path, required=True)

    p = sub.add_parser("baselines", help="Popularity, Markov-1, Markov-2 ONCE per split on both folds")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/processed")
    p.add_argument("--run-root", type=Path, default=ROOT / "runs/split2022")
    p.add_argument("--split-seed", type=int, default=2022)
    p.add_argument("--bootstrap", type=int, default=1000)

    p = sub.add_parser("report", help="Within-one-split scores; full campaign use run_campaign.py summary")
    p.add_argument("--run-root", type=Path, default=ROOT / "runs/split2022/seed2022")
    p.add_argument("--fold", choices=("validation", "test"), default="validation")

    a = parser.parse_args()
    if a.command == "doctor":
        import platform
        import torch
        import numpy as np
        print(json.dumps({"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "device": str(runtime(a.device, a.threads)),
            "cuda_available": torch.cuda.is_available(),
            "data": BirdData(a.data_dir, split_seed=a.split_seed).summary()}, indent=2))
    elif a.command == "train":
        c = DEFAULTS.copy()
        if a.config:
            custom = read_json(a.config)
            if set(custom) - set(DEFAULTS):
                raise ValueError(f"Unknown config keys: {set(custom)-set(DEFAULTS)}")
            c.update(custom)
        c.update({k: getattr(a, k) for k in DEFAULTS if getattr(a, k) is not None})
        c["split_seed"] = a.split_seed
        validate(c)
        if a.model == "all" and a.run_dir:
            raise ValueError("Use --run-root with --model all")
        names = ("gru4rec", "sasrec") if a.model == "all" else (a.model,)
        for name in names:
            train(a.data_dir, a.run_dir or a.run_root / name, dict(c, model=name), a.device, a.threads)
    elif a.command == "resume":
        if a.epochs < 1:
            raise ValueError("epochs must be positive")
        train(a.data_dir, a.run_dir, {"epochs": a.epochs}, a.device, a.threads, resume=True)
    elif a.command == "evaluate-test":
        result = evaluate_test(a.data_dir, a.run_dir, a.device, a.threads)
        print(json.dumps(result["test"]["micro"], indent=2))
    elif a.command == "baselines":
        result = baselines(a.data_dir, a.run_root, a.bootstrap, a.split_seed)
        for name in ("popularity", "markov1", "markov2"):
            print(f"{name} split={a.split_seed} TEST", json.dumps(result[name]["test"]["micro"]))
    elif a.command == "report":
        report(a.run_root, a.fold)


if __name__ == "__main__":
    main()
