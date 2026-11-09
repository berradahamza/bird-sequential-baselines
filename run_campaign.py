#!/usr/bin/env python3
"""Five CLL visitor splits x three initialization seeds, with resumable training.

No test metrics are read during training; test is invoked separately.
All published CLL split/initialization seed IDs are retained.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

from bird_baselines.data import BirdData, read_json

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/processed"
RUNS = ROOT / "runs"
SPLITS = (2022, 2023, 2024, 2025, 2026)
SEEDS = (2022, 2023, 2024)
MODELS = ("gru4rec", "sasrec")
METRICS = ("acc1", "hit5", "hit10", "mrr")


def call(*args):
    command = [sys.executable, "-u", str(ROOT / "run.py"), *map(str, args)]
    print("\n> " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def save_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def make_summary(splits=SPLITS, seeds=SEEDS, models=MODELS):
    """Never count parameter-free estimates three times; group neural seeds then splits."""
    raw = []
    for split in splits:
        root = RUNS / f"split{split}"
        base = root / "baselines.json"
        if base.exists():
            doc = read_json(base)
            for method in ("popularity", "markov1", "markov2"):
                for fold in ("validation", "test"):
                    m = doc[method][fold]["micro"]
                    raw.append(dict(model=method, split_seed=split, seed="", fold=fold,
                                    selected_epoch="", examples=m["examples"],
                                    **{k:m[k] for k in METRICS}))
        for seed in seeds:
            for method in models:
                d = root / f"seed{seed}" / method
                val_file, test_file = d / "metrics.json", d / "test_metrics.json"
                if val_file.exists() and read_json(val_file).get("status", "").startswith("completed"):
                    doc = read_json(val_file)
                    m = doc["validation"]["micro"]
                    raw.append(dict(model=method, split_seed=split, seed=seed, fold="validation",
                                    selected_epoch=doc["selected_epoch"], examples=m["examples"],
                                    **{k:m[k] for k in METRICS}))
                if test_file.exists():
                    doc = read_json(test_file)
                    m = doc["test"]["micro"]
                    raw.append(dict(model=method, split_seed=split, seed=seed, fold="test",
                                    selected_epoch=doc["selected_epoch"], examples=m["examples"],
                                    **{k:m[k] for k in METRICS}))
    summary_dir = RUNS / "summary"
    save_csv(summary_dir / "results_per_run.csv", raw)
    per_split = []
    for fold in ("validation", "test"):
        for split in splits:
            for model in (*models, "popularity", "markov1", "markov2"):
                subset = [r for r in raw if r["model"] == model and r["split_seed"] == split and r["fold"] == fold]
                if not subset:
                    continue
                expected = len(seeds) if model in MODELS else 1
                row = dict(model=model, fold=fold, split_seed=split,
                           n_runs=len(subset), expected_runs=expected, complete=len(subset)==expected)
                for metric in METRICS:
                    values = [r[metric] for r in subset]
                    row[f"{metric}_mean"] = statistics.mean(values)
                    row[f"{metric}_std_seeds"] = statistics.stdev(values) if len(values)>1 else ""
                per_split.append(row)
    save_csv(summary_dir / "results_per_split.csv", per_split)
    final = []
    for fold in ("validation", "test"):
        for model in (*models, "popularity", "markov1", "markov2"):
            sub = [r for r in per_split if r["model"] == model and r["fold"] == fold and r["complete"]]
            row = dict(model=model, fold=fold, completed_splits=len(sub), expected_splits=len(splits),
                       complete=len(sub)==len(splits))
            for metric in METRICS:
                values = [r[f"{metric}_mean"] for r in sub]
                row[f"{metric}_mean"] = statistics.mean(values) if values else ""
                row[f"{metric}_std_splits"] = statistics.stdev(values) if len(values)>1 else ""
            final.append(row)
    save_csv(summary_dir / "results_final.csv", final)
    print(f"\nSynthèses écrites dans {summary_dir}. Une ligne partielle n'est pas un résultat final.")


def main():
    parser = argparse.ArgumentParser(description="BIRD baselines | same 5 splits / 3 seeds as CLL")
    parser.add_argument("mode", choices=("plan", "train", "test", "summary"), nargs="?", default="plan")
    parser.add_argument("--splits", nargs="+", type=int, default=list(SPLITS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if len(set(args.splits)) != len(args.splits) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Duplicate split or initialization seed")
    if any(s not in SPLITS for s in args.splits) or any(s not in SEEDS for s in args.seeds):
        parser.error("CLL campaign: split seeds 2022..2026; initialization seeds 2022..2024")

    if args.mode == "plan":
        for split in args.splits:
            d = BirdData(DATA, split_seed=split)
            counts = {k:len(v) for k,v in d.examples.items()}
            print(f"split={split} visitors={ {k:len(v) for k,v in d.split.items()} } "
                  f"prefixes={counts} train_batches_64={math.ceil(counts['train']/64)} "
                  f"seeds={args.seeds} models={args.models}")
        print(f"Prévu : {len(args.splits)*len(args.seeds)*len(args.models)} entraînements; "
              f"{len(args.splits)} calculs non neuronaux uniques (3 estimateurs chacun).")
        return

    if args.mode == "summary":
        make_summary(args.splits, args.seeds, args.models)
        return

    if args.mode == "train":
        for split in args.splits:
            root = RUNS / f"split{split}"
            baselines_path = root / "baselines.json"
            if not baselines_path.exists():
                call("baselines", "--split-seed", split, "--run-root", root)
            else:
                d = read_json(baselines_path)
                current = BirdData(DATA, split_seed=split)
                if d["split_file_sha256"] != current.split_file_hash or d["processed_input_sha256"] != current.hashes:
                    raise ValueError(f"Statistical run {split} uses different data/split")
                print(f"Baseline non neuronale déjà calculée : split {split}")
            for seed in args.seeds:
                for model in args.models:
                    d = root / f"seed{seed}" / model
                    meta = d / "metrics.json"
                    if meta.exists() and read_json(meta).get("status", "").startswith("completed"):
                        c = read_json(d / "config.json")
                        if (c["split_seed"],c["seed"],c["model"],c["epochs"],c["patience"],c["batch_size"]) != (split,seed,model,30,5,64):
                            raise ValueError(f"Conflicting existing run: {d}")
                        print(f"Déjà entraîné : {d}")
                    elif (d / "last.pt").exists() and (d / "config.json").exists():
                        c = read_json(d / "config.json")
                        if (c["split_seed"],c["seed"],c["model"],c["epochs"],c["patience"],c["batch_size"]) != (split,seed,model,30,5,64):
                            raise ValueError(f"Conflicting partial run: {d}")
                        call("resume", "--run-dir", d, "--epochs", 30, "--device", args.device, "--threads", args.threads)
                    else:
                        if d.exists() and any(d.iterdir()):
                            raise ValueError(f"Nonempty incomplete run without last.pt: {d}. Inspect manually.")
                        call("train", "--model", model, "--run-dir", d,
                             "--split-seed", split, "--seed", seed,
                             "--device", args.device, "--threads", args.threads)
                    make_summary(args.splits,args.seeds,args.models)

    if args.mode == "test":
        for split in args.splits:
            for seed in args.seeds:
                for model in args.models:
                    d = RUNS / f"split{split}" / f"seed{seed}" / model
                    c = read_json(d / "config.json")
                    if (c["split_seed"],c["seed"],c["model"],c["epochs"],c["patience"],c["batch_size"]) != (split,seed,model,30,5,64):
                        raise ValueError(f"Wrong protocol: {d}")
                    if not read_json(d / "metrics.json").get("status", "").startswith("completed"):
                        raise ValueError(f"Not trained: {d}")
                    call("evaluate-test", "--run-dir", d, "--device", args.device, "--threads", args.threads)
                    make_summary(args.splits,args.seeds,args.models)
    make_summary(args.splits,args.seeds,args.models)


if __name__ == "__main__":
    main()
