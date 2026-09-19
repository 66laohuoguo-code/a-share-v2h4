"""factor_eval 的单元测试（合成数据，不需要任何数据库）。

这些测试同时是"协议应当如何表现"的规格说明：
  * 可分组门必须把离散计数型信号判为不可用；
  * 控制审计必须发现恒为常数的控制列；
  * 反转向复检必须同时做到 —— 低秩相关不能救一个实质是反转的因子、
    加回被剔除的风格必须能杀掉"伪新因子"、真正的因子必须活下来。
"""

from __future__ import annotations

import math
import os
import unittest

import numpy as np
import pandas as pd

from factor_eval import adapter, gates, protocol, stats


class TestStats(unittest.TestCase):
    def test_newey_west_matches_naive_when_no_autocorrelation(self):
        rng = np.random.default_rng(0)
        values = pd.Series(rng.normal(size=400))
        naive = values.mean() / (values.std(ddof=1) / math.sqrt(len(values)))
        hac = stats.newey_west_t(values, lags=4)
        self.assertAlmostEqual(hac, naive, delta=0.6)

    def test_newey_west_widens_under_positive_autocorrelation(self):
        rng = np.random.default_rng(1)
        base = rng.normal(size=300)
        series = pd.Series(base + 0.8 * np.roll(base, 1))
        naive = series.mean() / (series.std(ddof=1) / math.sqrt(len(series)))
        self.assertLess(abs(stats.newey_west_t(series, 4)), abs(naive) + 1e-9)

    def test_spearman_handles_duplicate_columns(self):
        # 列名重复时 pair[name] 会是 DataFrame —— 必须仍能算
        frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        doubled = pd.concat([frame, frame], axis=1)
        self.assertTrue(math.isfinite(stats.spearman(doubled["x"], frame["x"])))

    def test_bonferroni_bar_grows_with_trials(self):
        self.assertLess(stats.bonferroni_t_bar(1), stats.bonferroni_t_bar(56))
        self.assertLess(stats.bonferroni_t_bar(56), stats.bonferroni_t_bar(196))
        self.assertAlmostEqual(stats.bonferroni_t_bar(56), 3.32, delta=0.02)
        self.assertAlmostEqual(stats.bonferroni_t_bar(196), 3.66, delta=0.02)

    def test_orthogonalise_removes_linear_component(self):
        rng = np.random.default_rng(2)
        panel = pd.DataFrame({
            "date": np.repeat(pd.date_range("2020-01-01", periods=30), 100),
            "code": np.tile(["C%03d" % i for i in range(100)], 30),
            "base": rng.normal(size=3000),
        })
        panel["mixed"] = 2.0 * panel["base"] + rng.normal(scale=0.5, size=3000)
        residual = stats.orthogonalise(panel, "mixed", ["base"])
        self.assertLess(abs(stats.spearman(residual, panel["base"])), 0.1)


class TestGates(unittest.TestCase):
    def setUp(self):
        self.panel, _ = adapter.make_synthetic_panel(n_dates=40, n_codes=120)

    def test_groupability_rejects_tied_counter(self):
        tied = self.panel.copy()
        # 一个只能取 0/1/2 的离散计数信号
        tied["counter"] = (tied["past_return"] > 0).astype(float) + (
            tied["past_return"] > 1.5).astype(float)
        result = gates.groupability(tied, "counter")
        self.assertFalse(result["applicable"])
        self.assertEqual(result["reason"],
                         "inapplicable_signal_too_tied_for_quantiles")

    def test_groupability_accepts_continuous(self):
        self.assertTrue(gates.groupability(self.panel, "honest_factor")["applicable"])

    def test_control_audit_finds_constant_column(self):
        audit = gates.control_audit(self.panel, ["size", "value", "beta"],
                                    era_col="era")
        constants = set(audit["constant_controls"]["control"])
        self.assertIn("beta", constants)          # train 段的 beta 是常数
        self.assertIn("常数", audit["verdict"])


class TestProtocol(unittest.TestCase):
    def setUp(self):
        self.panel, _ = adapter.make_synthetic_panel(n_dates=200, n_codes=150)
        self.styles = ["size", "momentum", "liquidity", "value", "earnings_yield",
                       "growth"]

    def test_reversal_retest_kills_low_correlation_reversal_mimic(self):
        """教训 1：与反转的秩相关低，不代表它不是反转。"""
        table = protocol.reversal_retest(self.panel, ["disguised_reversal"],
                                         "past_return", self.styles)
        raw = table.loc[table["object"] == "raw"].iloc[0]
        residual = table.loc[table["object"] == "res"].iloc[0]
        self.assertLess(abs(raw["rho_with_reversal"]), 0.6)     # 相关确实不高
        self.assertGreater(abs(raw["t_train"]), 5.0)            # 原始看起来很强
        # 但正交掉反转后样本外就没了
        self.assertLess(abs(residual["t_test"]), abs(raw["t_test"]))

    def test_reversal_retest_kills_pure_style_exposure(self):
        """教训 2：正交掉反转还不够，必须把剔除的风格加回来。"""
        table = protocol.reversal_retest(self.panel, ["value_style"],
                                         "past_return", self.styles)
        for _, row in table.iterrows():
            self.assertLess(row["style_t_test"], 3.0)
        self.assertFalse(table["survived"].any())

    def test_reversal_retest_keeps_honest_factor(self):
        table = protocol.reversal_retest(self.panel, ["honest_factor"],
                                         "past_return", self.styles)
        self.assertTrue(table["survived"].any())

    def test_ablation_ladder_is_monotone_in_layers(self):
        layers = {"L0": ["size"], "L1": ["size", "value"],
                  "L2": ["size", "value", "momentum"]}
        table = protocol.ablation_ladder(self.panel, "honest_factor", layers)
        self.assertEqual(list(table["layer"]), ["L0", "L1", "L2"])
        self.assertTrue(table["t"].notna().all())

    def test_quantile_profile_reports_avoidance_alpha(self):
        result = protocol.quantile_profile(self.panel, "honest_factor")
        self.assertIn("top_minus_pool", result)
        self.assertIn("pool_minus_bottom", result)
        self.assertEqual(len(result["profile"]), 5)


class TestAdapter(unittest.TestCase):
    def test_synthetic_panel_has_both_eras(self):
        panel, truth = adapter.make_synthetic_panel(n_dates=20, n_codes=60)
        self.assertEqual(set(panel["era"].unique()), {"train", "test"})
        train, test = adapter.split_eras(panel)
        self.assertEqual(len(train) + len(test), len(panel))
        self.assertFalse(truth.empty)

    def test_load_panel_rejects_missing_columns(self):
        # 刻意不用 tempfile：某些受限环境会拒绝 tempfile 的 0700 chmod
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "_factor_eval_tmp")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "bad.csv")
        try:
            pd.DataFrame({"a": [1], "b": [2]}).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                adapter.load_panel(path)
        finally:
            if os.path.exists(path):
                os.remove(path)
            if os.path.isdir(folder) and not os.listdir(folder):
                os.rmdir(folder)


if __name__ == "__main__":
    unittest.main()

