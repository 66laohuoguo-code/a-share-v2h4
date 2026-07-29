import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import ashare_utils
import build_csmar_database
import database_status
import factor_rank_backtest
import factor_rank_backtest_v2h
import import_csmar_forward_quotation
import update_csmar_company_metadata
import weekly_rebalance_v2h


class TradingRulesTest(unittest.TestCase):
    def test_historical_cost_schedule(self):
        self.assertAlmostEqual(
            ashare_utils.mandatory_trade_cost_rate("SELL", "2021-01-05", "000001"),
            0.0010887,
        )
        self.assertAlmostEqual(
            ashare_utils.mandatory_trade_cost_rate("SELL", "2024-01-05", "000001"),
            0.0005641,
        )

    def test_broker_commission_rate_and_minimum_are_applied_per_order(self):
        buy_small = ashare_utils.mandatory_trade_cost(
            1_000.0,
            "BUY",
            "2024-01-05",
            "000001",
            broker_commission_rate=0.0003,
            broker_minimum_commission=5.0,
        )
        sell_large = ashare_utils.mandatory_trade_cost(
            20_000.0,
            "SELL",
            "2024-01-05",
            "000001",
            broker_commission_rate=0.0003,
            broker_minimum_commission=5.0,
        )

        self.assertAlmostEqual(buy_small, 5.0641)
        self.assertAlmostEqual(sell_large, 17.282)

    def test_official_strategy_records_personal_broker_costs(self):
        args = factor_rank_backtest_v2h.parse_args(
            ["--strategy-config", "config/v2h4_strategy.json"]
        )
        self.assertEqual(args.broker_commission_rate, 0.0003)
        self.assertEqual(args.broker_minimum_commission, 5.0)
        self.assertEqual(args.risk_reduction_min_trade_weight, 0.01)
        self.assertEqual(args.risk_increase_min_trade_weight, 0.01)

    def test_10w_causal_strategy_only_rebalances_from_the_prior_week(self):
        args = factor_rank_backtest_v2h.parse_args(
            ["--strategy-config", "config/v2h4_strategy_10w_weekly_causal.json"]
        )
        dates = ["2020-12-31", "2021-01-04", "2021-01-05"]
        date_to_index = {date: index for index, date in enumerate(dates)}

        self.assertEqual(args.rebalance_schedule, "week_end")
        self.assertEqual(args.risk_rebalance_schedule, "scheduled_only")
        self.assertTrue(
            ashare_utils.should_rebalance_on_date(
                args.rebalance_schedule,
                0,
                args.rebalance_every_n_days,
                dates,
                date_to_index,
                "2020-12-31",
            )
        )
        self.assertFalse(
            factor_rank_backtest_v2h.should_run_unscheduled_risk_rebalance(
                args, 0.80, 0.55
            )
        )

    def test_board_order_sizes(self):
        self.assertEqual(ashare_utils.buy_order_size_rules("000001"), (100, 100))
        self.assertEqual(ashare_utils.buy_order_size_rules("688001"), (200, 1))
        self.assertEqual(ashare_utils.buy_order_size_rules("830001"), (100, 1))

    def test_indicative_prices_are_rounded_to_the_a_share_tick(self):
        self.assertEqual(ashare_utils.round_price_to_tick(10.001, "BUY"), 10.01)
        self.assertEqual(ashare_utils.round_price_to_tick(10.009, "SELL"), 10.00)

    def test_new_bse_codes_use_the_thirty_percent_price_limit(self):
        self.assertEqual(factor_rank_backtest.price_limit_rate("920001"), 0.30)

    def test_entry_exit_trade_floors_are_separate_from_adjustments(self):
        common = {
            "portfolio_value": 566_446.71,
            "min_trade_value": 20_000.0,
            "min_trade_weight": 0.0,
            "entry_exit_min_trade_value": 2_000.0,
            "entry_exit_min_trade_weight": 0.0,
        }
        entry_floor, entry_type = ashare_utils.trade_value_floor(
            current_shares=0, target_shares=1000, **common
        )
        exit_floor, exit_type = ashare_utils.trade_value_floor(
            current_shares=1000, target_shares=0, **common
        )
        adjust_floor, adjust_type = ashare_utils.trade_value_floor(
            current_shares=1000, target_shares=1200, **common
        )

        self.assertEqual((entry_floor, entry_type), (2_000.0, "ENTRY"))
        self.assertEqual((exit_floor, exit_type), (0.0, "EXIT"))
        self.assertEqual((adjust_floor, adjust_type), (20_000.0, "ADJUST"))

    def test_risk_alignment_uses_separate_lower_buy_and_sell_floors(self):
        sell_floor, sell_mode = ashare_utils.apply_risk_alignment_trade_floor(
            order_floor=1_500.0,
            portfolio_value=20_000.0,
            side="SELL",
            current_equity_weight=0.80,
            target_equity_weight=0.55,
            risk_rebalance_band=0.03,
            risk_reduction_min_trade_weight=0.01,
            risk_increase_min_trade_weight=0.01,
        )
        buy_floor, buy_mode = ashare_utils.apply_risk_alignment_trade_floor(
            order_floor=1_500.0,
            portfolio_value=20_000.0,
            side="BUY",
            current_equity_weight=0.55,
            target_equity_weight=0.80,
            risk_rebalance_band=0.03,
            risk_reduction_min_trade_weight=0.01,
            risk_increase_min_trade_weight=0.01,
        )
        unchanged, inactive_mode = ashare_utils.apply_risk_alignment_trade_floor(
            order_floor=1_500.0,
            portfolio_value=20_000.0,
            side="SELL",
            current_equity_weight=0.57,
            target_equity_weight=0.55,
            risk_rebalance_band=0.03,
            risk_reduction_min_trade_weight=0.01,
            risk_increase_min_trade_weight=0.01,
        )

        self.assertEqual(sell_floor, 200.0)
        self.assertEqual(sell_mode, "REDUCE")
        self.assertEqual(buy_floor, 200.0)
        self.assertEqual(buy_mode, "INCREASE")
        self.assertEqual(unchanged, 1_500.0)
        self.assertEqual(inactive_mode, "")

    def test_percentage_trade_floors_scale_with_portfolio_value(self):
        common = {
            "portfolio_value": 566_446.71,
            "min_trade_value": 0.0,
            "min_trade_weight": 0.02,
            "entry_exit_min_trade_value": 0.0,
            "entry_exit_min_trade_weight": 0.0035,
        }
        entry_floor, _ = ashare_utils.trade_value_floor(
            current_shares=0, target_shares=1000, **common
        )
        adjust_floor, _ = ashare_utils.trade_value_floor(
            current_shares=1000, target_shares=1200, **common
        )

        self.assertAlmostEqual(entry_floor, 1_982.563485)
        self.assertAlmostEqual(adjust_floor, 11_328.9342)

    def test_force_risk_alignment_bypasses_the_per_stock_no_trade_band(self):
        features = pd.DataFrame(
            [
                {
                    "code": "000001",
                    "industry_1": "A",
                    "score_v2": 1.0,
                    "volatility_120": 0.20,
                }
            ]
        )
        args = SimpleNamespace(
            target_count=1,
            min_target_count=1,
            buy_rank=1,
            sell_rank=2,
            max_stock_weight=1.0,
            max_industry_weight=1.0,
            enable_lot_aware_selection=False,
            score_temperature=0.80,
            min_stock_volatility=0.08,
            inverse_vol_power=1.0,
            rebalance_band_weight=1.0,
        )
        ordinary, _ = factor_rank_backtest_v2h.build_targets_v2(
            features,
            {"000001": 100},
            {"000001": 0.80},
            args,
            target_equity_weight=0.55,
        )
        aligned, meta = factor_rank_backtest_v2h.build_targets_v2(
            features,
            {"000001": 100},
            {"000001": 0.80},
            args,
            target_equity_weight=0.55,
            force_risk_alignment=True,
        )

        self.assertAlmostEqual(ordinary["000001"], 0.80)
        self.assertAlmostEqual(aligned["000001"], 0.55)
        self.assertTrue(meta["force_risk_alignment"])

    def test_risk_alignment_modes_preserve_the_best_banded_baseline(self):
        best = factor_rank_backtest_v2h.parse_args(
            [
                "--strategy-config",
                "config/v2h4_small_account_20k_best_20260723.json",
            ]
        )
        candidate = factor_rank_backtest_v2h.parse_args(
            [
                "--strategy-config",
                "config/v2h4_small_account_20k_sparse_weekly.json",
            ]
        )

        self.assertEqual(best.risk_target_alignment, "banded")
        self.assertEqual(best.risk_rebalance_schedule, "daily")
        self.assertEqual(best.risk_alignment_max_orders, 0)
        self.assertFalse(
            factor_rank_backtest_v2h.should_force_risk_alignment(best, 0.80, 0.55)
        )

        self.assertEqual(candidate.risk_target_alignment, "strict")
        self.assertEqual(candidate.risk_rebalance_schedule, "scheduled_only")
        self.assertEqual(candidate.risk_alignment_max_orders, 4)
        self.assertTrue(
            factor_rank_backtest_v2h.should_force_risk_alignment(candidate, 0.80, 0.55)
        )
        self.assertFalse(
            factor_rank_backtest_v2h.should_run_unscheduled_risk_rebalance(
                candidate, 0.80, 0.55
            )
        )

    def test_sparse_risk_alignment_keeps_the_fewest_large_orders_needed(self):
        planned = [
            {"code": "A", "risk_alignment_mode": "REDUCE", "gross_amount": 2_000.0},
            {"code": "B", "risk_alignment_mode": "REDUCE", "gross_amount": 1_800.0},
            {"code": "C", "risk_alignment_mode": "REDUCE", "gross_amount": 1_600.0},
            {"code": "D", "risk_alignment_mode": "REDUCE", "gross_amount": 1_400.0},
            {"code": "E", "risk_alignment_mode": "", "gross_amount": 3_000.0},
        ]

        selected = factor_rank_backtest_v2h.select_sparse_risk_alignment_orders(
            planned,
            current_equity_weight=0.80,
            target_equity_weight=0.55,
            portfolio_value=20_000.0,
            max_orders=4,
        )

        self.assertEqual([item["code"] for item in selected], ["A", "B", "C"])
        self.assertEqual(selected[0]["sparse_risk_orders_dropped"], 2)
        self.assertEqual(selected[0]["risk_alignment_order_cap"], 4)

    def test_sparse_risk_alignment_uses_a_separate_initial_deployment_cap(self):
        planned = [
            {
                "code": str(index),
                "transition_type": "ENTRY",
                "risk_alignment_mode": "INCREASE",
                "gross_amount": 1_000.0,
            }
            for index in range(12)
        ]

        selected = factor_rank_backtest_v2h.select_sparse_risk_alignment_orders(
            planned,
            current_equity_weight=0.0,
            target_equity_weight=0.90,
            portfolio_value=20_000.0,
            max_orders=4,
            initial_max_orders=12,
        )

        self.assertEqual(len(selected), 12)
        self.assertTrue(all(item["sparse_risk_execution"] for item in selected))

    def test_portfolio_lot_rounding_uses_residual_cash_for_an_extra_lot(self):
        shares = ashare_utils.round_portfolio_target_shares(
            {"000001": 0.035, "000002": 0.035},
            portfolio_value=10_000.0,
            prices={"000001": 3.0, "000002": 4.0},
        )
        self.assertEqual(shares["000001"], 100)
        self.assertEqual(shares["000002"], 100)
        self.assertEqual(shares["000001"] * 3.0 + shares["000002"] * 4.0, 700.0)

    def test_lot_aware_selection_skips_unaffordable_ranked_stock(self):
        features = pd.DataFrame(
            [
                {"code": "000001", "close": 5.0, "execution_close": 100.0, "industry_1": "A"},
                {"code": "000002", "close": 5.0, "industry_1": "A"},
                {"code": "000003", "close": 6.0, "industry_1": "B"},
                {"code": "000004", "close": 7.0, "industry_1": "C"},
                {"code": "000005", "close": 8.0, "industry_1": "D"},
                {"code": "000006", "close": 9.0, "industry_1": "E"},
                {"code": "000007", "close": 10.0, "industry_1": "F"},
            ]
        )
        features["execution_close"] = features["execution_close"].fillna(features["close"])
        features["score_v2"] = list(reversed(range(len(features))))
        features["volatility_120"] = 0.20
        args = SimpleNamespace(
            target_count=5,
            sell_rank=10,
            buy_rank=10,
            max_industry_weight=0.40,
            slippage_bps=0.0,
            lot_aware_min_holdings=3,
            enable_lot_aware_selection=True,
            score_temperature=0.80,
            min_stock_volatility=0.08,
            inverse_vol_power=1.0,
            max_stock_weight=0.035,
            rebalance_band_weight=0.0025,
            lot_aware_stock_cap_multiplier=1.25,
            lot_aware_max_stock_weight=0.25,
            lot_aware_max_industry_weight=0.50,
        )

        selected, minimum_weights, meta = (
            factor_rank_backtest_v2h.select_lot_aware_codes(
                features,
                {},
                args,
                target_equity_weight=0.70,
                portfolio_value=10_000.0,
            )
        )

        self.assertNotIn("000001", selected)
        self.assertIn("000002", selected)
        self.assertEqual(len(selected), 5)
        self.assertLessEqual(float(minimum_weights.sum()), 0.70)
        self.assertGreater(meta["skipped_lot_too_expensive"], 0)

        targets, target_meta = factor_rank_backtest_v2h.build_targets_v2(
            features,
            {},
            {},
            args,
            target_equity_weight=0.70,
            portfolio_value=10_000.0,
        )
        self.assertAlmostEqual(sum(targets.values()), 0.70, places=6)
        self.assertLessEqual(target_meta["effective_max_stock_weight"], 0.25)
        for code, weight in targets.items():
            close = float(features.loc[features["code"].eq(code), "close"].iloc[0])
            minimum, _ = ashare_utils.buy_order_size_rules(code)
            self.assertGreaterEqual(weight * 10_000.0 + 1e-8, close * minimum)

    def test_corporate_action_accounting(self):
        holdings = {"003010": 1000}
        rows = {
            "003010": {
                "close": 11.13,
                "daily_return": 0.0046,
                "capital_return": -0.0144,
            }
        }
        cash, actions = factor_rank_backtest.apply_corporate_actions_before_open(
            holdings, rows, {"003010": 15.81}
        )
        self.assertEqual(holdings["003010"], 1400)
        self.assertAlmostEqual(cash, 300.39, places=2)
        self.assertEqual(len(actions), 1)


class DatabaseStatusTest(unittest.TestCase):
    def test_database_status(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.sqlite"
            conn = sqlite3.connect(path)
            conn.execute("CREATE TABLE stock_daily (code TEXT, trade_date TEXT)")
            conn.executemany(
                "INSERT INTO stock_daily VALUES (?, ?)",
                [("000001", "2026-01-05"), ("000002", "2026-01-06")],
            )
            conn.commit()
            conn.close()
            status = database_status.read_status(path)
            self.assertEqual(status["max_date"], "2026-01-06")
            self.assertEqual(status["rows"], 2)
            self.assertEqual(status["stocks"], 2)


class AccountIsolationTest(unittest.TestCase):
    def test_account_state_rejects_a_different_account(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "account_state.json"
            weekly_rebalance_v2h.save_account_state(
                path,
                {"account_id": "account_large", "peak_portfolio_value": 500_000.0},
            )

            state = weekly_rebalance_v2h.load_account_state(path, "account_large")
            self.assertEqual(state["peak_portfolio_value"], 500_000.0)
            with self.assertRaisesRegex(ValueError, "Account state mismatch"):
                weekly_rebalance_v2h.load_account_state(path, "account_small")

    def test_legacy_shared_state_without_account_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "account_state.json"
            path.write_text('{"peak_portfolio_value": 500000}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "has no account_id"):
                weekly_rebalance_v2h.load_account_state(path, "account_small")

    def test_default_state_paths_are_account_specific(self):
        large = weekly_rebalance_v2h.default_account_state_path("account_large")
        small = weekly_rebalance_v2h.default_account_state_path("account_small")
        self.assertNotEqual(large, small)
        self.assertEqual(large.name, "account_state.json")


class WeeklyCapitalStrategyRoutingTest(unittest.TestCase):
    def test_capital_boundaries_select_the_validated_strategy_tiers(self):
        strategy_map = weekly_rebalance_v2h.load_capital_strategy_map(
            Path("config/weekly_capital_strategy_map.json")
        )
        cases = [
            (20_000.0, "small_sparse_12", "v2h4_small_account_20k_best_20260724_sparse_weekly.json"),
            (49_999.99, "small_sparse_12", "v2h4_small_account_20k_best_20260724_sparse_weekly.json"),
            (50_000.0, "core_20", "v2h4_strategy_10w_20stock_concentrated.json"),
            (749_999.99, "core_20", "v2h4_strategy_10w_20stock_concentrated.json"),
            (750_000.0, "capacity_25", "v2h4_strategy_10w_25stock_balanced.json"),
            (10_000_000.0, "capacity_25", "v2h4_strategy_10w_25stock_balanced.json"),
        ]
        for value, expected_tier, expected_file in cases:
            with self.subTest(value=value):
                selected = weekly_rebalance_v2h.select_strategy_for_capital(
                    value, strategy_map
                )
                self.assertEqual(selected["tier"], expected_tier)
                self.assertEqual(Path(selected["strategy_config"]).name, expected_file)

    def test_weekly_cli_defaults_to_automatic_selection_but_allows_manual_override(self):
        automatic = weekly_rebalance_v2h.parse_args(
            ["--account-id", "account_a", "--positions", "positions.csv"]
        )
        self.assertIsNone(automatic.strategy_config)
        self.assertEqual(
            automatic.capital_strategy_map,
            Path("config/weekly_capital_strategy_map.json"),
        )

        manual = weekly_rebalance_v2h.parse_args(
            [
                "--account-id",
                "account_a",
                "--positions",
                "positions.csv",
                "--strategy-config",
                "config/v2h4_strategy.json",
            ]
        )
        self.assertEqual(manual.strategy_config, Path("config/v2h4_strategy.json"))


class ForwardQuotationImportTest(unittest.TestCase):
    def test_market_value_per_share_reconstructs_unadjusted_close(self):
        columns = {
            "MarketValue": 0,
            "TotalShare": 1,
            "CirculatedMarketValue": 2,
            "CirculatedShare": 3,
        }
        raw = [8_682.0, 100.0, 6_077.4, 70.0]
        close = import_csmar_forward_quotation.unadjusted_close_from_row(
            raw, columns
        )
        self.assertAlmostEqual(close, 86.82)

    def test_market_value_per_share_rejects_conflicting_price_bases(self):
        columns = {
            "MarketValue": 0,
            "TotalShare": 1,
            "CirculatedMarketValue": 2,
            "CirculatedShare": 3,
        }
        raw = [8_682.0, 100.0, 7_000.0, 70.0]

        close = import_csmar_forward_quotation.unadjusted_close_from_row(raw, columns)

        self.assertIsNone(close)

    def test_a_share_value_is_preferred_over_conflicting_company_value(self):
        columns = {
            "MarketValue": 0,
            "TotalShare": 1,
            "CirculatedMarketValue": 2,
            "CirculatedShare": 3,
            "AValue": 4,
            "ACirculatedShare": 5,
        }
        raw = [539.0, 100.0, 492.0, 100.0, 427.0, 70.0]

        close = import_csmar_forward_quotation.unadjusted_close_from_row(
            raw, columns, code="000539"
        )

        self.assertAlmostEqual(close, 6.10)

    def test_schema_migration_adds_raw_prices_and_invalidates_file_cache(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE stock_daily (code TEXT, trade_date TEXT)")
        conn.execute(
            """
            CREATE TABLE source_file_state (
                source_file TEXT PRIMARY KEY,
                file_size INTEGER,
                file_mtime_ns INTEGER,
                imported_at TEXT
            )
            """
        )
        conn.execute("INSERT INTO source_file_state VALUES ('old.xlsx', 1, 1, 'now')")

        import_csmar_forward_quotation.ensure_tables_exist(conn, apply=True)

        columns = {row[1] for row in conn.execute("PRAGMA table_info(stock_daily)")}
        self.assertTrue(set(import_csmar_forward_quotation.RAW_PRICE_COLUMNS) <= columns)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM source_file_state").fetchone()[0], 0)
        conn.close()


class FullCsmarBuildTest(unittest.TestCase):
    def test_v2_checkpoint_round_trip_restores_pending_factor_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "sample.sqlite"
            sqlite3.connect(database).close()
            checkpoint = root / "checkpoint.json.gz"
            args = SimpleNamespace(
                database=database,
                output_dir=root / "output",
                checkpoint_file=checkpoint,
                checkpoint_every_n_days=5,
                resume=True,
                strategy_name="checkpoint_test",
                start_date="2026-01-05",
                end_date="2026-01-09",
                initial_cash=10_000.0,
            )
            fingerprint = factor_rank_backtest_v2h.checkpoint_fingerprint(args)
            weighter = factor_rank_backtest_v2h.RollingICWeighter(
                args, ["low_beta_score"]
            )
            weighter.add_snapshot(
                "2026-01-05",
                "2026-01-06",
                "2026-01-09",
                pd.DataFrame([{"code": "000001", "low_beta_score": 0.75}]),
            )
            weighter.ic_rows.append({"decision_date": "2026-01-05", "low_beta_score": 0.1})
            state = factor_rank_backtest_v2h.make_checkpoint_state(
                fingerprint,
                1,
                ["2026-01-05", "2026-01-06"],
                {"000001": 100},
                9_000.0,
                {"000001": 10.0},
                10_000.0,
                10_100.0,
                [{"trade_date": "2026-01-05"}],
                [{"code": "000001", "side": "BUY"}],
                weighter,
            )

            factor_rank_backtest_v2h.save_checkpoint(checkpoint, state, args)
            loaded = factor_rank_backtest_v2h.load_checkpoint(
                checkpoint, fingerprint, 2
            )
            restored = factor_rank_backtest_v2h.RollingICWeighter(
                args, ["low_beta_score"]
            )
            factor_rank_backtest_v2h.restore_weighter(restored, loaded["weighter"])

            self.assertEqual(loaded["next_offset"], 1)
            self.assertEqual(loaded["holdings"], {"000001": 100})
            self.assertEqual(restored.pending[0].exit_date, "2026-01-09")
            self.assertEqual(restored.pending[0].frame.iloc[0]["code"], "000001")
            self.assertTrue(
                factor_rank_backtest_v2h.checkpoint_metadata_path(checkpoint).exists()
            )

            changed_args = SimpleNamespace(**vars(args))
            changed_args.initial_cash = 20_000.0
            with self.assertRaisesRegex(ValueError, "does not match"):
                factor_rank_backtest_v2h.load_checkpoint(
                    checkpoint,
                    factor_rank_backtest_v2h.checkpoint_fingerprint(changed_args),
                    2,
                )

            factor_rank_backtest_v2h.clear_checkpoint(checkpoint)
            self.assertFalse(checkpoint.exists())

    def test_company_industry_update_fills_unknown_without_reclassifying_known_stock(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE stock_daily (code TEXT, industry_1 TEXT, industry_2 TEXT)")
        conn.execute("CREATE TABLE stock_meta (code TEXT, industry_1 TEXT, industry_2 TEXT)")
        rows = [
            ("000001", "UNKNOWN", "UNKNOWN"),
            ("000002", "K", "K70"),
        ]
        conn.executemany("INSERT INTO stock_daily VALUES (?,?,?)", rows)
        conn.executemany("INSERT INTO stock_meta VALUES (?,?,?)", rows)
        companies = {
            "000001": {"code": "000001", "industry_1": "J", "industry_2": "J66"},
            "000002": {"code": "000002", "industry_1": "C", "industry_2": "C39"},
        }

        update_csmar_company_metadata.create_mapping_table(conn, companies)
        updated = update_csmar_company_metadata.update_industries(
            conn, "stock_daily", "missing"
        )
        update_csmar_company_metadata.update_industries(conn, "stock_meta", "missing")

        self.assertEqual(updated, 1)
        self.assertEqual(
            conn.execute(
                "SELECT industry_1, industry_2 FROM stock_daily WHERE code='000001'"
            ).fetchone(),
            ("J", "J66"),
        )
        self.assertEqual(
            conn.execute(
                "SELECT industry_1, industry_2 FROM stock_daily WHERE code='000002'"
            ).fetchone(),
            ("K", "K70"),
        )
        conn.close()

    def test_schema_finalization_creates_model_views(self):
        conn = sqlite3.connect(":memory:")
        build_csmar_database.create_schema(conn)
        record = {column: None for column in build_csmar_database.STOCK_DAILY_COLUMNS}
        record.update(
            {
                "code": "000001",
                "name": "Sample",
                "trade_date": "2026-07-17",
                "close": 10.0,
                "raw_close": 10.0,
                "daily_return": 0.01,
                "capital_return": 0.01,
                "listed_state": "Norm",
                "currency": "CNY",
                "industry_1": "J66",
                "industry_2": "J66",
                "imported_at": "now",
            }
        )
        build_csmar_database.insert_daily_batch(conn, [record])
        build_csmar_database.finalize_database(conn)

        self.assertEqual(conn.execute("SELECT COUNT(*) FROM stock_meta").fetchone()[0], 1)
        self.assertEqual(conn.execute("SELECT raw_close FROM latest_prices").fetchone()[0], 10.0)
        conn.close()

    def test_daily_record_maps_raw_prices_returns_and_turnover(self):
        headers = [
            "Stkcd", "Trddt", "Opnprc", "Hiprc", "Loprc", "Clsprc",
            "Dnshrtrd", "Dnvaltrd", "Dsmvosd", "Dsmvtll", "Dretwd",
            "Dretnd", "Adjprcwd", "Adjprcnd", "Markettype", "Trdsta",
            "PreClosePrice", "ChangeRatio",
        ]
        row = [
            "000001", "2019-01-02", "9.39", "9.42", "9.16", "9.19",
            "53938632", "498695109.66", "157794567.84", "157796080.45",
            "-0.020256", "-0.020256", "953.465954", "761.264196",
            "4", "1", "9.38", "-0.020256",
        ]
        source_root = Path("source")
        source_file = source_root / "TRD_Dalyr.xlsx"
        metadata = {
            "000001": {"name": "Ping An Bank", "industry_1": "J66", "industry_2": "J66"}
        }

        record, gap = build_csmar_database.daily_record(
            row,
            {name: index for index, name in enumerate(headers)},
            source_file,
            "sheet1",
            source_root,
            "now",
            metadata,
        )

        self.assertEqual(record["code"], "000001")
        self.assertEqual(record["close"], 9.19)
        self.assertEqual(record["raw_close"], 9.19)
        self.assertEqual(record["daily_return"], -0.020256)
        self.assertEqual(record["capital_return"], -0.020256)
        self.assertEqual(record["listed_state"], "Norm")
        self.assertEqual(record["industry_1"], "J66")
        self.assertAlmostEqual(
            record["turnover_total"],
            53938632 * 9.19 / (157796080.45 * 1000) * 100,
        )
        self.assertLess(gap, 5e-6)

    def test_incremental_anchor_is_latest_close_before_source_start(self):
        conn = sqlite3.connect(":memory:")
        conn.execute(
            """
            CREATE TABLE stock_daily (
                code TEXT,
                name TEXT,
                trade_date TEXT,
                close REAL,
                listed_state TEXT,
                currency TEXT,
                industry_1 TEXT,
                industry_2 TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO stock_daily VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("000001", "Sample", "2026-03-31", 10.0, "Norm", "CNY", "A", "A01"),
                ("000001", "Sample", "2026-07-03", 12.5, "Norm", "CNY", "A", "A01"),
                ("000001", "Sample", "2026-07-06", 13.0, "Norm", "CNY", "A", "A01"),
            ],
        )

        meta = import_csmar_forward_quotation.latest_database_meta_before(
            conn, "2026-07-06"
        )

        self.assertEqual(meta["000001"]["anchor_date"], "2026-07-03")
        self.assertEqual(meta["000001"]["last_close"], 12.5)
        conn.close()


if __name__ == "__main__":
    unittest.main()
