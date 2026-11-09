"""Checks frozen CLL split reproduction, neural baselines and Markov reference."""
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from bird_baselines.__main__ import DEFAULTS, ROOT
from bird_baselines.data import BirdData, pack_histories, read_json
from bird_baselines.engine import baselines
from bird_baselines.evaluation import ranking_records
from bird_baselines.models import build_model


class CorrectnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.data = BirdData(ROOT / "data/processed")

    def test_protocol_default(self):
        self.assertEqual((DEFAULTS["batch_size"],DEFAULTS["epochs"],DEFAULTS["patience"]), (64,30,5))
        self.assertEqual(set(["gru4rec","sasrec"]),set(["gru4rec","sasrec"]))
        with self.assertRaises(ValueError):
            build_model(dict(DEFAULTS, model="cl4srec"), 359)

    def test_five_exact_cll_partitions(self):
        sizes = {2022:(5082,988,1083,80),2023:(4955,1122,1076,78),
                 2024:(4766,1168,1219,75),2025:(4942,1075,1136,78),
                 2026:(5098,1098,957,80)}
        for seed,(ntrain,nval,ntest,nbatches) in sizes.items():
            d = BirdData(ROOT / "data/processed", split_seed=seed)
            self.assertEqual((len(d.examples["train"]),len(d.examples["validation"]),
                              len(d.examples["test"])),(ntrain,nval,ntest))
            self.assertEqual(math.ceil(ntrain/64),nbatches)
            self.assertEqual([len(d.split[k]) for k in ("train","validation","test")],[35,8,8])
            self.assertEqual(len(set(sum((v for v in d.split.values()),[]))),51)
            frozen = read_json(ROOT / "splits" / f"split_{seed}.json")
            self.assertEqual(frozen["visitors"],d.split)
        self.assertEqual(self.data.split,read_json(ROOT / "data/processed/split.json")["visitors"])

    def test_prefix_target_causality(self):
        events = read_json(ROOT / "data/processed/trajectories_poi.json")["visitors"]
        for seed in (2022,2024):
            d=BirdData(ROOT / "data/processed", split_seed=seed)
            for fold_rows in d.examples.values():
                for row in fold_rows:
                    visit=events[row["visitor"]]["events"]
                    i=row["event_index"]
                    self.assertEqual(row["history"],
                        [d.index[e["poi_id"]] for e in visit[max(0,i-31):i+1]])
                    self.assertEqual(row["target"],d.index[visit[i+1]["poi_id"]])

    def test_padding_and_candidate_count(self):
        for name in ("gru4rec","sasrec"):
            model=build_model(dict(DEFAULTS,model=name),359).eval()
            x,lengths=pack_histories([[1,2,3],[4]])
            y,lengths2=pack_histories([[1,2,3],[4]],width=8)
            with torch.no_grad():
                scores=model(x,lengths)
                extra=model(y,lengths2)
            self.assertEqual(tuple(scores.shape),(2,359))
            torch.testing.assert_close(scores,extra,atol=2e-6,rtol=2e-5)
            self.assertTrue(torch.isfinite(scores).all())

    def test_future_does_not_affect_prefix(self):
        for name in ("gru4rec","sasrec"):
            model=build_model(dict(DEFAULTS,model=name),359).eval()
            with torch.no_grad():
                a=model.encode_steps(torch.tensor([[1,2,3,4]]))
                b=model.encode_steps(torch.tensor([[1,2,88,99]]))
            torch.testing.assert_close(a[:,:2],b[:,:2],atol=1e-6,rtol=1e-5)

    def test_training_gradients_finite(self):
        for name in ("gru4rec","sasrec"):
            model=build_model(dict(DEFAULTS,model=name),359)
            ids,lengths=pack_histories([[1,2,3],[4],[5,6]])
            loss=torch.nn.functional.cross_entropy(model(ids,lengths),torch.tensor([8,9,10]))
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_full_ranking_stable_ties(self):
        r=ranking_records(np.array([[1.,1.,0.],[0.,2.,3.]]),np.array([1,0]),
                          ["a","b"],["A","B","C"])
        self.assertEqual([x["rank"] for x in r],[2,3])

    def test_campaign_summary_does_not_triple_count_markov(self):
        import csv
        from unittest.mock import patch
        import run_campaign
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "split2022"
            base.mkdir(parents=True)
            method = {"micro": {"examples":988,"acc1":0.3,"hit5":0.5,"hit10":0.7,"mrr":0.4}}
            baseline = {m:{f:method for f in ("validation","test")}
                        for m in ("popularity","markov1","markov2")}
            (base / "baselines.json").write_text(__import__("json").dumps(baseline))
            for seed in (2022,2023,2024):
                dest = base / f"seed{seed}" / "gru4rec"
                dest.mkdir(parents=True)
                (dest / "metrics.json").write_text(__import__("json").dumps({
                    "status":"completed", "selected_epoch":3,
                    "validation":method}))
            with patch.object(run_campaign,"RUNS",root):
                run_campaign.make_summary([2022],[2022,2023,2024],["gru4rec"])
            with (root/"summary/results_per_split.csv").open(encoding="utf-8-sig") as f:
                rows=list(csv.DictReader(f))
            selected={(r["model"],r["fold"]):r for r in rows}
            self.assertEqual(selected[("markov1","validation")]["n_runs"],"1")
            self.assertEqual(selected[("gru4rec","validation")]["n_runs"],"3")
            self.assertEqual(selected[("markov2","validation")]["complete"],"True")

    def test_markov_reproduces_stkg_cll_reference(self):
        # Exact numbers from STKG_CLL_POI/runs/baselines_seed2022/baselines.json
        with tempfile.TemporaryDirectory() as td:
            d=baselines(ROOT / "data/processed",td,bootstrap=0,split_seed=2022)
            self.assertTrue((Path(td)/"baselines.json").exists())
            expected={
              "popularity":{"validation":(.012145748987854251,.042676607094846886),
                            "test":(.009233610341643583,.039604656107258766)},
              "markov1":{"validation":(.3016194331983806,.4644071451981056),
                         "test":(.3157894736842105,.4693326393116761)},
              "markov2":{"validation":(.3350202429149798,.4800275627922073),
                         "test":(.3601108033240997,.4931417172758787)}}
            for method,folds in expected.items():
                for fold,(acc,mrr) in folds.items():
                    self.assertAlmostEqual(d[method][fold]["micro"]["acc1"],acc,places=12)
                    self.assertAlmostEqual(d[method][fold]["micro"]["mrr"],mrr,places=12)


if __name__ == "__main__":
    unittest.main()
