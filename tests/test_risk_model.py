import math
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

import ashare_risk_model
import build_risk_model_data
import build_weekly_risk_model
import validate_risk_model


class RiskModelFinancialTest(unittest.TestCase):
    def test_ttm_derivation_uses_current_ytd_and_prior_comparables(self):
        rows = [
            {
                "code": "000001",
                "report_period": "2022-03-31",
                "available_date": "2022-04-25",
                "operating_revenue_ytd": 20.0,
                "net_profit_ytd": 2.0,
                "parent_net_profit_ytd": 1.8,
                "operating_cashflow_ytd": 1.0,
                "capital_expenditure_ytd": 0.5,
            },
            {
                "code": "000001",
                "report_period": "2022-12-31",
                "available_date": "2023-03-20",
                "operating_revenue_ytd": 100.0,
                "net_profit_ytd": 10.0,
                "parent_net_profit_ytd": 9.0,
                "operating_cashflow_ytd": 12.0,
                "capital_expenditure_ytd": 4.0,
            },
            {
                "code": "000001",
                "report_period": "2023-03-31",
                "available_date": "2023-04-25",
                "operating_revenue_ytd": 30.0,
                "net_profit_ytd": 3.0,
                "parent_net_profit_ytd": 2.7,
                "operating_cashflow_ytd": 2.0,
                "capital_expenditure_ytd": 0.8,
            },
        ]
        updates = build_risk_model_data.derive_ttm_rows(rows)
        latest = {
            (row[-2], row[-1]): row
            for row in updates
        }[("000001", "2023-03-31")]
        self.assertAlmostEqual(latest[0], 110.0)
        self.assertAlmostEqual(latest[1], 11.0)
        self.assertAlmostEqual(latest[2], 9.9)
        self.assertAlmostEqual(latest[3], 13.0)
        self.assertAlmostEqual(latest[4], 4.3)
        self.assertEqual(latest[5], "2023-04-25")

    def test_financial_store_never_exposes_data_before_available_date(self):
        rows = [
            {
                "code": "000001",
                "report_period": "2023-12-31",
                "available_date": "2024-03-20",
                "ttm_available_date": "2024-03-20",
                "growth_available_date": "2024-03-20",
                "total_assets": 10.0,
                "total_liabilities": 4.0,
                "parent_equity": 5.0,
                "total_equity": 6.0,
                "parent_net_profit_ttm": 1.0,
                "operating_cashflow_ttm": 1.2,
                "revenue_growth": 0.1,
                "earnings_growth": 0.2,
            }
        ]
        store = build_weekly_risk_model.FinancialPointInTimeStore(
            rows, ["2024-03-19", "2024-03-20", "2024-03-21"]
        )
        store.advance("2024-03-19")
        self.assertIsNone(store.get("000001")["financial_report_period"])
        store.advance("2024-03-20")
        self.assertIsNone(store.get("000001")["financial_report_period"])
        store.advance("2024-03-21")
        self.assertEqual(
            store.get("000001")["financial_report_period"], "2023-12-31"
        )


class RiskModelStyleTest(unittest.TestCase):
    def make_daily(self, returns):
        dates = pd.bdate_range("2022-01-03", periods=len(returns))
        return pd.DataFrame(
            {
                "trade_date": dates,
                "daily_return": returns,
                "market_return": np.full(len(returns), 0.001),
                "risk_free_return": np.zeros(len(returns)),
                "turnover_float": np.full(len(returns), 1.0),
                "amount": np.full(len(returns), 100_000_000.0),
                "total_market_cap": np.full(len(returns), 10_000_000_000.0),
                "float_market_cap": np.full(len(returns), 8_000_000_000.0),
                "listed_state": np.full(len(returns), "Norm"),
            }
        )

    def test_momentum_skips_the_most_recent_21_trading_days(self):
        base_returns = np.full(300, 0.001)
        changed_returns = base_returns.copy()
        changed_returns[-21:] = 0.08
        base = self.make_daily(base_returns)
        changed = self.make_daily(changed_returns)
        model_date = base["trade_date"].iloc[-1].strftime("%Y-%m-%d")
        config = {
            "beta_window_days": 252,
            "beta_minimum_days": 126,
            "beta_half_life_days": 63,
            "momentum_lookback_days": 252,
            "momentum_skip_days": 21,
            "liquidity_windows_days": [21, 63, 252],
            "maximum_stale_calendar_days": 10,
        }
        base_row = ashare_risk_model.compute_stock_weekly_features(
            base, [model_date], config
        )[0]
        changed_row = ashare_risk_model.compute_stock_weekly_features(
            changed, [model_date], config
        )[0]
        self.assertAlmostEqual(
            base_row["momentum_raw"], changed_row["momentum_raw"], places=12
        )

    def test_factor_regression_recovers_constrained_industry_returns(self):
        rng = np.random.default_rng(20260730)
        count = 180
        industries = np.array(["20", "30", "40"] * (count // 3))
        exposure = pd.DataFrame(
            {
                "model_date": "2024-01-05",
                "code": [f"{index:06d}" for index in range(count)],
                "industry_group": industries,
                "total_market_cap": np.full(count, 1_000_000_000.0),
            }
        )
        style_returns = {}
        for index, factor in enumerate(ashare_risk_model.STYLE_FACTORS):
            exposure[factor] = rng.normal(0.0, 1.0, count)
            style_returns[factor] = (index + 1) * 0.0001
        industry_returns = {"20": 0.001, "30": 0.002, "40": -0.003}
        realized = np.full(count, 0.01)
        for factor, factor_return in style_returns.items():
            realized += exposure[factor].to_numpy() * factor_return
        realized += np.array([industry_returns[value] for value in industries])
        returns = pd.DataFrame(
            {"code": exposure["code"], "weekly_return": realized}
        )

        fitted, specific, _ = ashare_risk_model.fit_factor_returns(
            exposure,
            returns,
            ["20", "30", "40"],
            {"industry_base_group": "40"},
        )
        self.assertAlmostEqual(fitted["MARKET"], 0.01, places=10)
        for factor, expected in style_returns.items():
            self.assertAlmostEqual(fitted[factor], expected, places=10)
        for industry, expected in industry_returns.items():
            self.assertAlmostEqual(
                fitted[f"INDUSTRY:{industry}"], expected, places=10
            )
        self.assertLess(specific["specific_return"].abs().max(), 1e-10)

    def test_newey_west_covariance_is_positive_semidefinite(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.02, size=(80, 12))
        covariance = ashare_risk_model.ewma_newey_west_covariance(returns)
        eigenvalues = np.linalg.eigvalsh(covariance)
        self.assertGreaterEqual(eigenvalues.min(), -1e-12)
        self.assertTrue(np.allclose(covariance, covariance.T))


class RiskModelSchemaTest(unittest.TestCase):
    def test_checkpoint_hashes_ignore_only_the_rolling_model_end_date(self):
        config = {
            "history_start_date": "2019-01-01",
            "model_start_date": "2021-01-01",
            "model_end_date": "2026-07-17",
            "beta_window_days": 252,
            "beta_minimum_days": 126,
            "beta_half_life_days": 63,
            "momentum_lookback_days": 252,
            "momentum_skip_days": 21,
            "liquidity_windows_days": [21, 63, 252],
            "market_return_types": [117, 53],
            "risk_free_benchmark": "NRI01",
        }
        original = build_weekly_risk_model.stage_config_hashes(config)
        extended = dict(config, model_end_date="2026-08-07")
        self.assertEqual(
            original,
            build_weekly_risk_model.stage_config_hashes(extended),
        )
        changed = dict(extended, beta_window_days=126)
        self.assertNotEqual(
            original,
            build_weekly_risk_model.stage_config_hashes(changed),
        )

    def test_legacy_horizon_hashes_are_migrated_without_clearing_rows(self):
        config = {
            "history_start_date": "2019-01-01",
            "model_start_date": "2021-01-01",
            "model_end_date": "2026-07-17",
            "beta_window_days": 252,
            "beta_minimum_days": 126,
            "beta_half_life_days": 63,
            "momentum_lookback_days": 252,
            "momentum_skip_days": 21,
            "liquidity_windows_days": [21, 63, 252],
            "market_return_types": [117, 53],
            "risk_free_benchmark": "NRI01",
        }
        legacy_raw, legacy_model = build_weekly_risk_model.stage_config_hashes(
            config, include_model_end_date=True
        )
        extended = dict(config, model_end_date="2026-08-07")
        stable_raw, stable_model = build_weekly_risk_model.stage_config_hashes(extended)
        conn = sqlite3.connect(":memory:")
        try:
            build_weekly_risk_model.create_model_schema(conn)
            conn.execute(
                "INSERT INTO risk_model_metadata VALUES ('config', ?)",
                (json.dumps(config),),
            )
            conn.execute(
                "INSERT INTO risk_model_build_state VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("raw", "000001", legacy_raw, "complete", 1, "now", None),
            )
            conn.execute(
                "INSERT INTO risk_model_build_state VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("model", "2026-07-17", legacy_model, "complete", 1, "now", None),
            )
            conn.commit()
            migrated = build_weekly_risk_model.migrate_legacy_horizon_hashes(
                conn, extended, stable_raw, stable_model
            )
            hashes = dict(
                conn.execute(
                    "SELECT stage, config_hash FROM risk_model_build_state"
                ).fetchall()
            )
        finally:
            conn.close()
        self.assertTrue(migrated)
        self.assertEqual(hashes, {"raw": stable_raw, "model": stable_model})

    def test_sidecar_and_model_schema_can_be_created_together(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "risk.sqlite"
            conn = sqlite3.connect(path)
            try:
                build_risk_model_data.create_schema(conn)
                build_weekly_risk_model.create_model_schema(conn)
                tables = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
            finally:
                conn.close()
        self.assertIn("financial_pit", tables)
        self.assertIn("weekly_exposure", tables)
        self.assertIn("weekly_factor_covariance", tables)

    def test_validator_reads_the_requested_model_end_date(self):
        conn = sqlite3.connect(":memory:")
        try:
            build_weekly_risk_model.create_model_schema(conn)
            conn.execute(
                """
                INSERT INTO risk_model_metadata(key, value)
                VALUES ('raw_end', '"2026-07-24"')
                """
            )
            self.assertEqual(
                validate_risk_model.configured_model_end_date(conn),
                "2026-07-24",
            )
        finally:
            conn.close()

    def test_small_model_pipeline_writes_covariance_and_specific_risk(self):
        rng = np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "risk.sqlite"
            conn = sqlite3.connect(path)
            try:
                build_risk_model_data.create_schema(conn)
                build_weekly_risk_model.create_model_schema(conn)
                codes = [f"{index:06d}" for index in range(90)]
                industries = ("20", "30", "40")
                dates = [
                    value.strftime("%Y-%m-%d")
                    for value in pd.date_range(
                        "2024-01-05", periods=10, freq="W-FRI"
                    )
                ]
                calendar = [
                    value.strftime("%Y-%m-%d")
                    for value in pd.bdate_range("2023-12-20", dates[-1])
                ]
                for index, code in enumerate(codes):
                    industry = industries[index % len(industries)]
                    conn.execute(
                        """
                        INSERT INTO industry_history
                        VALUES (?, 'TEST', 'TEST', '2023-01-01', ?, ?, '', 'test')
                        """,
                        (code, industry + "0000", industry),
                    )
                    conn.execute(
                        """
                        INSERT INTO financial_pit (
                            code, name, report_period, actual_disclosure_date,
                            available_date, total_assets, total_liabilities,
                            parent_equity, total_equity, parent_net_profit_ttm,
                            operating_cashflow_ttm, ttm_available_date,
                            revenue_growth, earnings_growth,
                            growth_available_date, is_corrected
                        ) VALUES (
                            ?, ?, '2023-09-30', '2023-10-20',
                            '2023-10-20', ?, ?, ?, ?, ?, ?, '2023-10-20',
                            ?, ?, '2023-10-20', 0
                        )
                        """,
                        (
                            code,
                            code,
                            10_000_000_000.0 + index * 1_000_000.0,
                            4_000_000_000.0 + index * 100_000.0,
                            5_000_000_000.0 + index * 500_000.0,
                            6_000_000_000.0 + index * 500_000.0,
                            500_000_000.0 + index * 20_000.0,
                            550_000_000.0 + index * 20_000.0,
                            0.05 + index * 0.0001,
                            0.04 + index * 0.0001,
                        ),
                    )
                    for date_index, model_date in enumerate(dates):
                        cap = 5_000_000_000.0 + index * 10_000_000.0
                        conn.execute(
                            """
                            INSERT INTO weekly_raw_exposure
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'Norm')
                            """,
                            (
                                model_date,
                                code,
                                model_date,
                                float(rng.normal(0.001, 0.02)),
                                cap,
                                cap * 0.8,
                                float(rng.normal(1.0, 0.2)),
                                float(rng.normal(0.1, 0.2)),
                                float(abs(rng.normal(0.25, 0.05))),
                                float(rng.normal(-0.5, 0.3)),
                                100_000_000.0,
                                300 + date_index,
                            ),
                        )
                conn.commit()
                config = {
                    "model_start_date": dates[0],
                    "model_end_date": dates[-1],
                    "industry_classification_name": "TEST",
                    "industry_group_digits": 2,
                    "industry_base_group": "40",
                    "minimum_history_days": 20,
                    "minimum_average_amount": 1.0,
                    "exclude_special_treatment": True,
                    "winsor_mad_width": 5.0,
                    "minimum_factor_history_weeks": 4,
                    "factor_covariance_window_weeks": 20,
                    "factor_covariance_half_life_weeks": 4,
                    "factor_covariance_shrinkage": 0.1,
                    "newey_west_lags": 1,
                    "minimum_specific_history_weeks": 3,
                    "specific_risk_half_life_weeks": 3,
                    "specific_risk_shrinkage": 0.2,
                }
                with contextlib.redirect_stdout(io.StringIO()):
                    build_weekly_risk_model.build_model(
                        conn, dates, calendar, config, "test-hash"
                    )
                covariance_rows = conn.execute(
                    "SELECT COUNT(*) FROM weekly_factor_covariance"
                ).fetchone()[0]
                specific_rows = conn.execute(
                    "SELECT COUNT(*) FROM weekly_specific_risk"
                ).fetchone()[0]
                latest_date = conn.execute(
                    "SELECT MAX(model_date) FROM weekly_risk_diagnostics"
                ).fetchone()[0]
            finally:
                conn.close()
        self.assertGreater(covariance_rows, 0)
        self.assertGreater(specific_rows, 0)
        self.assertEqual(latest_date, dates[-1])


if __name__ == "__main__":
    unittest.main()
