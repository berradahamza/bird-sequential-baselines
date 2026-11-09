"""Training, validation selection, checkpoint recovery and separate held-out test."""
from __future__ import annotations

import csv
import math
import os
import platform
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import BirdData, batch_tensors, read_json, sha256, write_json
from .evaluation import ranking_records, summarize
from .models import LABELS, build_model


def runtime(device="auto", threads=4):
    torch.set_num_threads(max(1, min(threads, os.cpu_count() or 1)))
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Use --device cpu or install the NVIDIA PyTorch build; see README.md.")
    return torch.device(device)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def log(run_dir, message):
    print(message, flush=True)
    with (Path(run_dir) / "training.log").open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def save_checkpoint(path, checkpoint):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save(checkpoint, temp)
    temp.replace(path)


def verify(data, config):
    if data.hashes != config["processed_input_sha256"] or data.poi_ids != config["poi_ids"]:
        raise ValueError("Checkpoint and data differ; keep the same JSON files and catalogue.")
    if data.split != config["split"] or data.split_seed != config["split_seed"]:
        raise ValueError("Checkpoint and visitor split differ")
    if data.split_file_hash != config["split_file_sha256"]:
        raise ValueError("Frozen split file changed")


@torch.inference_mode()
def evaluate(model, rows, data, batch_size, device, bootstrap=0, seed=2022):
    model.eval()
    records = []
    ce_sum = 0.0
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        ids, lengths, targets = batch_tensors(batch, device)
        scores = model(ids, lengths)
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Non-finite evaluation scores")
        ce_sum += float(F.cross_entropy(scores, targets, reduction="sum"))
        part = ranking_records(scores.cpu().numpy(), targets.cpu().numpy(),
                               [r["visitor"] for r in batch], data.poi_ids)
        for record, example in zip(part, batch):
            record["event_index"] = example["event_index"]
        records.extend(part)
    metrics = summarize(records, bootstrap, seed)
    metrics["ce"] = ce_sum / len(rows)
    return metrics, records


def predictions_csv(path, records):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["visitor", "event_index", "target", "rank", "top5"])
        for r in records:
            w.writerow([r["visitor"], r["event_index"], r["target"], r["rank"], ";".join(r["top5"])])


def save_history(run_dir, history):
    write_json(Path(run_dir) / "history.json", history)
    if history:
        with (Path(run_dir) / "history.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0]))
            w.writeheader()
            w.writerows(history)


def train(data_dir, run_dir, config, device="auto", threads=4, resume=False):
    """Last-position CE on every train prefix. Test rows never enter forward passes."""
    run_dir = Path(run_dir)
    selected_device = runtime(device, threads)
    if resume:
        if (run_dir / "test_metrics.json").exists():
            raise ValueError("This run already has test results. Preserve it and use a new run for another experiment.")
        stored = read_json(run_dir / "config.json")
        current_code = {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
        if stored["source_sha256"] != current_code:
            raise ValueError("Code changed since training. Restore it before resuming this checkpoint.")
        stored["epochs"] = config["epochs"]
        config = stored
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"{run_dir} already contains a run. Use resume or another --run-dir.")
    run_dir.mkdir(parents=True, exist_ok=True)
    data = BirdData(data_dir, config["max_history"], config["split_seed"])
    if resume:
        verify(data, config)
    else:
        config = dict(config, model_label=LABELS[config["model"]],
                      processed_input_sha256=data.hashes, poi_ids=data.poi_ids, split=data.split,
                      split_file_sha256=data.split_file_hash,
                      selection_metric="validation.micro.mrr", candidate_mask=False,
                      data_summary=data.summary(),
                      source_sha256={p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                      environment={"python": platform.python_version(), "torch": str(torch.__version__),
                                   "numpy": np.__version__, "platform": platform.platform(),
                                   "device": str(selected_device), "threads": torch.get_num_threads()})
    write_json(run_dir / "config.json", config)
    seed_all(config["seed"])
    model = build_model(config, len(data.poi_ids)).to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    start_epoch, best_mrr, best_epoch, stale, history = 1, -1.0, 0, 0, []
    if resume:
        ckpt = torch.load(run_dir / "last.pt", map_location=selected_device, weights_only=True)
        if ckpt["data_hashes"] != data.hashes:
            raise ValueError("Resume checkpoint data mismatch")
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_mrr, best_epoch, stale = ckpt["best_mrr"], ckpt["best_epoch"], ckpt["stale"]
        history = ckpt["history"]
        # Recover best.pt even if interruption occurred between the two atomic saves.
        save_checkpoint(run_dir / "best.pt", {"state_dict": ckpt["best_state_dict"],
                        "epoch": best_epoch, "config": config})
    log(run_dir, f"{config['model_label']} | device={selected_device} | parameters={sum(p.numel() for p in model.parameters()):,}")
    log(run_dir, f"train={len(data.examples['train'])} | validation={len(data.examples['validation'])} | catalogue={len(data.poi_ids)} | batch={config['batch_size']}")
    if resume:
        best_state = ckpt["best_state_dict"]
    else:
        best_state = None
    for epoch in range(start_epoch, config["epochs"] + 1):
        # Epoch-local seeds allow exact boundary resume on the same CPU/runtime.
        seed_all(config["seed"] + epoch)
        order_rng = random.Random(config["seed"] + epoch)
        order = list(range(len(data.examples["train"])))
        order_rng.shuffle(order)
        model.train()
        totals = {"loss": 0.0, "ce": 0.0}
        epoch_start = time.perf_counter()
        batches = math.ceil(len(order) / config["batch_size"])
        for batch_no, start in enumerate(range(0, len(order), config["batch_size"]), 1):
            rows = [data.examples["train"][i] for i in order[start:start + config["batch_size"]]]
            ids, lengths, targets = batch_tensors(rows, selected_device)
            optimizer.zero_grad(set_to_none=True)
            ce = F.cross_entropy(model(ids, lengths), targets)
            loss = ce
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss; last completed epoch remains in last.pt")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            for key, val in (("loss", loss), ("ce", ce)):
                totals[key] += float(val.detach()) * len(rows)
            if batch_no % config["log_every"] == 0 or batch_no == batches:
                log(run_dir, f"epoch {epoch:02d} | batch {batch_no:03d}/{batches} | CE={float(ce.detach()):.4f}")
        validation, _ = evaluate(model, data.examples["validation"], data, config["batch_size"], selected_device)
        m = validation["micro"]
        row = {"epoch": epoch, **{f"train_{k}": v / len(order) for k, v in totals.items()},
               "validation_ce": validation["ce"], "validation_mrr": m["mrr"],
               "validation_acc1": m["acc1"], "validation_hit5": m["hit5"],
               "validation_hit10": m["hit10"], "seconds": time.perf_counter() - epoch_start}
        history.append(row)
        improved = m["mrr"] > best_mrr
        if improved:
            best_mrr, best_epoch, stale = m["mrr"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        save_checkpoint(run_dir / "last.pt", {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "epoch": epoch, "best_mrr": best_mrr, "best_epoch": best_epoch,
                        "stale": stale, "history": history, "best_state_dict": best_state, "data_hashes": data.hashes})
        if improved:
            save_checkpoint(run_dir / "best.pt", {"state_dict": best_state, "epoch": epoch, "config": config})
        save_history(run_dir, history)
        write_json(run_dir / "metrics.json", {"status": "training", "selected_epoch": best_epoch,
                   "trained_epochs": epoch, "best_validation_mrr": best_mrr,
                   "test": "not_evaluated", "history": history})
        log(run_dir, f"epoch {epoch:02d} | val MRR={m['mrr']:.4f} | Acc@1={m['acc1']:.4f} | Hit@5={m['hit5']:.4f} | Hit@10={m['hit10']:.4f} | {row['seconds']:.1f}s" + (" | BEST" if improved else ""))
        if stale >= config["patience"]:
            log(run_dir, f"Early stopping: {stale} epochs without a better validation MRR.")
            break
    checkpoint = torch.load(run_dir / "best.pt", map_location=selected_device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    validation, records = evaluate(model, data.examples["validation"], data, config["batch_size"],
                                    selected_device, config["bootstrap"], config["seed"])
    result = {"status": "completed; one seed, exploratory", "model": config["model_label"],
              "selected_epoch": checkpoint["epoch"], "trained_epochs": history[-1]["epoch"],
              "selection_metric": "validation.micro.mrr", "validation": validation,
              "test": "not_evaluated_use_evaluate-test", "history": history}
    write_json(run_dir / "metrics.json", result)
    predictions_csv(run_dir / "validation_predictions.csv", records)
    log(run_dir, f"Selected epoch {checkpoint['epoch']} | validation MRR={validation['micro']['mrr']:.4f} | saved to {run_dir}")
    return result


def evaluate_test(data_dir, run_dir, device="auto", threads=4):
    run_dir = Path(run_dir)
    config = read_json(run_dir / "config.json")
    metrics = read_json(run_dir / "metrics.json")
    if not metrics["status"].startswith("completed"):
        raise ValueError("Training has not completed; resume before evaluating the test.")
    data = BirdData(data_dir, config["max_history"], config["split_seed"])
    verify(data, config)
    checkpoint_hash = sha256(run_dir / "best.pt")
    output = run_dir / "test_metrics.json"
    if output.exists():
        previous = read_json(output)
        if previous["checkpoint_sha256"] != checkpoint_hash:
            raise ValueError("A different checkpoint was already evaluated in this run directory.")
        print(f"Existing test reused: {output}", flush=True)
        return previous
    selected_device = runtime(device, threads)
    checkpoint = torch.load(run_dir / "best.pt", map_location=selected_device, weights_only=True)
    model = build_model(config, len(data.poi_ids)).to(selected_device)
    model.load_state_dict(checkpoint["state_dict"])
    result, records = evaluate(model, data.examples["test"], data, config["batch_size"],
                               selected_device, config["bootstrap"], config["seed"])
    report = {"status": "test evaluated after validation selection; one split/seed, exploratory",
              "model": config["model_label"], "selected_epoch": checkpoint["epoch"],
              "test": result, "checkpoint_sha256": checkpoint_hash,
              "processed_input_sha256": data.hashes, "split_seed": data.split_seed,
              "split_file_sha256": data.split_file_hash}
    predictions_csv(run_dir / "test_predictions.csv", records)
    write_json(output, report)
    return report


def baselines(data_dir, run_root, bootstrap=1000, split_seed=2022):
    """Compute three parameter-free baselines ONCE per CLL split, for both folds.

    Markov-1 and Markov-2 are the same 0.001 backoff estimators as the STKG
    experiment (full 359-POI ranking; counts from train visitors only).
    """
    data = BirdData(data_dir, split_seed=split_seed)
    output = Path(run_root)
    output.mkdir(parents=True, exist_ok=True)
    pop, one, two = Counter(), defaultdict(Counter), defaultdict(Counter)
    for row in data.examples["train"]:
        h, target = row["history"], row["target"]
        pop[target] += 1
        one[h[-1]][target] += 1
        if len(h) >= 2:
            two[tuple(h[-2:])][target] += 1
    n_items = len(data.poi_ids)
    base = np.array([pop.get(i, 0) + 0.001 for i in range(1, n_items + 1)], dtype=float)
    base /= base.sum()

    def distribution(counts, backoff):
        if not counts:
            return backoff
        a = backoff * 0.001
        for key, count in counts.items():
            a[key - 1] += count
        return a / a.sum()

    results = {"split_seed": split_seed, "split": data.split,
               "split_file_sha256": data.split_file_hash,
               "processed_input_sha256": data.hashes,
               "scope": "full-catalogue; counts from train only; 0.001 backoff smoothing",
               "train_transition_count": sum(pop.values()),
               "selection": "parameter-free; not selected by validation"}
    for method in ("popularity", "markov1", "markov2"):
        results[method] = {}
        for fold in ("validation", "test"):
            rows = data.examples[fold]
            scores = []
            for row in rows:
                h = row["history"]
                first = distribution(one.get(h[-1]), base)
                score = base if method == "popularity" else first
                if method == "markov2" and len(h) >= 2:
                    score = distribution(two.get(tuple(h[-2:])), first)
                scores.append(score)
            records = ranking_records(np.array(scores), np.array([r["target"] - 1 for r in rows]),
                                      [r["visitor"] for r in rows], data.poi_ids)
            results[method][fold] = summarize(records, bootstrap, split_seed)
    write_json(output / "baselines.json", results)
    for fold in ("validation", "test"):
        write_json(output / f"baselines_{fold}.json", {
            "split_seed": split_seed, "split": data.split,
            "split_file_sha256": data.split_file_hash,
            "processed_input_sha256": data.hashes,
            **{name: results[name][fold] for name in ("popularity", "markov1", "markov2")}})
    return results


def report(run_root, fold="validation"):
    root = Path(run_root)
    rows, expected_hashes, expected_split_hash = [], None, None
    baseline_path = root / f"baselines_{fold}.json"
    if baseline_path.exists():
        base = read_json(baseline_path)
        expected_hashes = base["processed_input_sha256"]
        expected_split_hash = base["split_file_sha256"]
        for name in ("popularity", "markov1", "markov2"):
            rows.append({"model": name, "selected_epoch": "", **base[name]["micro"]})
    for run in sorted(root.iterdir()):
        path = run / ("metrics.json" if fold == "validation" else "test_metrics.json")
        if not run.is_dir() or not path.exists():
            continue
        result = read_json(path)
        if fold not in result or not isinstance(result[fold], dict):
            continue
        config = read_json(run / "config.json")
        hashes = config["processed_input_sha256"]
        if expected_hashes is not None and hashes != expected_hashes:
            raise ValueError("Cannot compare runs built from different data")
        if expected_split_hash is not None and config["split_file_sha256"] != expected_split_hash:
            raise ValueError("Cannot compare runs from different visitor partitions")
        expected_hashes = hashes
        expected_split_hash = config["split_file_sha256"]
        rows.append({"model": run.name, "selected_epoch": result["selected_epoch"], **result[fold]["micro"]})
    if not rows:
        raise ValueError(f"No completed {fold} results in {root}")
    target = root / f"comparison_{fold}.csv"
    with target.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "selected_epoch", "examples", "acc1", "hit5", "hit10", "mrr"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n{fold.upper()} | model                  | MRR    | Acc@1  | Hit@5  | Hit@10")
    for row in rows:
        print(f"{'':10} | {row['model']:22} | {row['mrr']:.4f} | {row['acc1']:.4f} | {row['hit5']:.4f} | {row['hit10']:.4f}")
    print(f"Saved: {target}")
    return rows
