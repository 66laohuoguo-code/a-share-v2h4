import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import ashare_utils
import build_csmar_database
import build_weekly_risk_model
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

    def test_2010_costs_use_exchange_specific_historical_rates(self):
        shanghai = ashare_utils.mandatory_trade_cost(
            10_000.0,
            "BUY",
            "2010-01-05",
            "600000",
            broker_commission_rate=0.0,
            shares=1_000,
        )
        shenzhen = ashare_utils.mandatory_trade_cost(
            10_000.0,
            "BUY",
            "2010-01-05",
            "000001",
            broker_commission_rate=0.0,
            shares=1_000,
        )
        self.assertAlmostEqual(shanghai, 2.0)
        self.assertAlmostEqual(shenzhen, 1.875)

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

    def test_historical_chinext_and_st_price_limits(self):
        self.assertEqual(
            factor_rank_backtest.price_limit_rate("300001", "2019-01-04"),
            0.10,
        )
        self.assertEqual(
            factor_rank_backtest.price_limit_rate("300001", "2021-01-04"),
            0.20,
        )
        self.assertEqual(
            factor_rank_backtest.price_limit_rate(
                "600000", "2019-01-04", "ST"
            ),
            0.05,
        )

    def test_actual_csmar_limit_prices_override_code_fallback(self):
        args = SimpleNamespace(
            disable_limit_trade_filter=False,
            limit_trade_buffer=0.0,
        )
        row = {
            "trade_date": "2019-01-04",
            "prev_close": 10.0,
            "open": 11.5,
            "limit_up": 12.0,
            "limit_down": 8.0,
            "listed_state": "Norm",
        }
        self.assertFalse(
            factor_rank_backtest.blocked_by_price_limit(
                "300001", row, "BUY", args
            )
        )
        row["open"] = 12.0
        self.assertTrue(
            factor_rank_backtest.blocked_by_price_limit(
                "300001", row, "BUY", args
            )
        )

    def test_csmar_no_limit_day_is_not_filtered(self):
        args = SimpleNamespace(
            disable_limit_trade_filter=False,
            limit_trade_buffer=0.0,
        )
        row = {
            "trade_date": "2019-01-04",
            "prev_close": 10.0,
            "open": 20.0,
            "listed_state": "Norm",
            "no_price_limit": 1,
        }
        self.assertFalse(
            factor_rank_backtest.blocked_by_price_limit(
                "600000", row, "BUY", args
            )
        )

    def test_risk_model_industry_schedule_is_sorted_and_causal(self):
        schedule = build_weekly_risk_model.industry_classification_schedule(
            {
                "industry_classification_schedule": [
                    {
                        "classification_name": "new",
                        "effective_date": "2012-10-26",
                    },
                    {
                        "classification_name": "old",
                        "effective_date": "1900-01-01",
                    },
                ]
            }
        )
        self.assertEqual(
            schedule,
            [
                {
                    "classification_name": "old",
                    "effective_date": "1900-01-01",
                },
                {
                    "classification_name": "new",
                    "effective_date": "2012-10-26",
                },
            ],
        )

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

    def test_replacement_guard_defers_unfunded_sells_near_the_target_equity(self):
        planned = [
            {"code": "A", "side": "SELL", "gross_amount": 3_000.0},
            {"code": "B", "side": "SELL", "gross_amount": 2_000.0},
        ]

        selected, diagnostics = (
            factor_rank_backtest_v2h.balance_executable_replacement_orders(
                planned,
                current_equity_weight=0.5529,
                target_equity_weight=0.55,
                portfolio_value=100_000.0,
            )
        )

        self.assertEqual(selected, [])
        self.assertEqual(diagnostics["deferred_replacement_sell_count"], 2)
        self.assertAlmostEqual(diagnostics["desired_net_buy_gross"], -290.0)

    def test_replacement_guard_keeps_balanced_switch_orders(self):
        planned = [
            {"code": "BUY", "side": "BUY", "gross_amount": 3_000.0},
            {"code": "SELL1", "side": "SELL", "gross_amount": 1_000.0},
            {"code": "SELL2", "side": "SELL", "gross_amount": 2_000.0},
        ]

        selected, diagnostics = (
            factor_rank_backtest_v2h.balance_executable_replacement_orders(
                planned,
                current_equity_weight=0.55,
                target_equity_weight=0.55,
                portfolio_value=100_000.0,
            )
        )

        self.assertEqual({item["code"] for item in selected}, {"BUY", "SELL1", "SELL2"})
        self.assertFalse(diagnostics["replacement_guard_applied"])

    def test_replacement_guard_allows_required_net_risk_reduction(self):
        planned = [
            {"code": "A", "side": "SELL", "gross_amount": 10_000.0},
            {"code": "B", "side": "SELL", "gross_amount": 15_000.0},
            {"code": "C", "side": "SELL", "gross_amount": 5_000.0},
        ]

        selected, diagnostics = (
            factor_rank_backtest_v2h.balance_executable_replacement_orders(
                planned,
                current_equity_weight=0.80,
                target_equity_weight=0.55,
                portfolio_value=100_000.0,
            )
        )

        self.assertAlmostEqual(
            sum(item["gross_amount"] for item in selected), 25_000.0
        )
        self.assertEqual(diagnostics["deferred_replacement_sell_count"], 1)

    def test_replacement_guard_accepts_a_lot_overshoot_when_it_is_closer(self):
        planned = [
            {"code": "A", "side": "SELL", "gross_amount": 1_588.0},
        ]

        selected, diagnostics = (
            factor_rank_backtest_v2h.balance_executable_replacement_orders(
                planned,
                current_equity_weight=0.5445,
                target_equity_weight=0.4797,
                portfolio_value=20_121.88,
            )
        )

        self.assertEqual([item["code"] for item in selected], ["A"])
        self.assertEqual(diagnostics["deferred_replacement_sell_count"], 0)

    def test_unmarketable_buy_does_not_fund_a_replacement_sell(self):
        planned = [
            {
                "code": "BUY",
                "side": "BUY",
                "gross_amount": 3_000.0,
                "auction_marketable": False,
            },
            {
                "code": "SELL",
                "side": "SELL",
                "gross_amount": 3_000.0,
                "auction_marketable": True,
            },
        ]

        selected, diagnostics = (
            factor_rank_backtest_v2h.balance_executable_replacement_orders(
                planned,
                current_equity_weight=0.55,
                target_equity_weight=0.55,
                portfolio_value=100_000.0,
            )
        )

        self.assertEqual([item["code"] for item in selected], ["BUY"])
        self.assertEqual(diagnostics["deferred_replacement_sell_count"], 1)

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
            (20_000.0, "v22s_20k_entry_weight", "v22s_20k_entry_weight_monthly_06_official.json"),
            (49_999.99, "v22s_20k_entry_weight", "v22s_20k_entry_weight_monthly_06_official.json"),
            (50_000.0, "v22r3_100k", "v22r3_weekly_100k_official.json"),
            (299_999.99, "v22r3_100k", "v22r3_weekly_100k_official.json"),
            (300_000.0, "v22r3_560k", "v22r3_weekly_560k_official.json"),
            (749_999.99, "v22r3_560k", "v22r3_weekly_560k_official.json"),
            (750_000.0, "v22r3_1m", "v22r3_weekly_1m_official.json"),
            (1_000_000.0, "v22r3_1m", "v22r3_weekly_1m_official.json"),
            (10_000_000.0, "v22r3_1m", "v22r3_weekly_1m_official.json"),
        ]
        for value, expected_tier, expected_file in cases:
            with self.subTest(value=value):
                selected = weekly_rebalance_v2h.select_strategy_for_capital(
                    value, strategy_map
                )
                self.assertEqual(selected["tier"], expected_tier)
                self.assertEqual(Path(selected["strategy_config"]).name, expected_file)

    def test_deployed_20k_strategy_uses_the_frozen_monthly_industry_signal(self):
        strategy = weekly_rebalance_v2h.strategy_args_from_config(
            Path("config/v22s_20k_entry_weight_monthly_06_official.json")
        )
        self.assertFalse(factor_rank_backtest_v2h.uses_v31_alpha_features(strategy))
        self.assertEqual(strategy.v22_industry_satellite_application, "entry_and_weight")
        self.assertEqual(strategy.v22_industry_satellite_schedule, "monthly")
        self.assertAlmostEqual(strategy.v22_industry_satellite_max_weight, 0.06)
        self.assertEqual(strategy.risk_overlay_mode, "variance_blend")

    def test_monthly_satellite_signal_date_matches_first_week_end_of_month(self):
        dates = [
            "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
            "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
            "2024-01-15", "2024-01-16", "2024-01-17", "2024-01-18", "2024-01-19",
            "2024-01-22", "2024-01-23", "2024-01-24", "2024-01-25", "2024-01-26",
            "2024-01-29", "2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02",
        ]
        date_to_index = {date: index for index, date in enumerate(dates)}
        self.assertEqual(
            weekly_rebalance_v2h.live_monthly_satellite_signal_date(
                dates, date_to_index, "2024-01-26"
            ),
            "2024-01-05",
        )
        self.assertEqual(
            weekly_rebalance_v2h.live_monthly_satellite_signal_date(
                dates, date_to_index, "2024-02-02"
            ),
            "2024-02-02",
        )
        self.assertEqual(
            weekly_rebalance_v2h.live_monthly_satellite_signal_date(
                dates, date_to_index, "2024-01-03"
            ),
            "2024-01-03",
        )

    def test_deployed_v22r3_configs_require_weekly_v31_alpha_data(self):
        for name in (
            "v22r3_weekly_100k_official.json",
            "v22r3_weekly_560k_official.json",
            "v22r3_weekly_1m_official.json",
        ):
            with self.subTest(config=name):
                strategy = weekly_rebalance_v2h.strategy_args_from_config(
                    Path("config") / name
                )
                self.assertTrue(
                    factor_rank_backtest_v2h.uses_v31_alpha_features(strategy)
                )
                self.assertAlmostEqual(strategy.v31_alpha_tilt_weight, 0.10)
                self.assertTrue(strategy.v31_enforce_unknown_industry_cap)
                self.assertEqual(
                    weekly_rebalance_v2h.required_v31_live_columns(strategy),
                    ("v31_earnings_yield_raw", "v31_quality_raw"),
                )

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

        risk = weekly_rebalance_v2h.parse_args(
            [
                "--account-id",
                "account_a",
                "--positions",
                "positions.csv",
                "--risk-model-database",
                "risk.sqlite",
                "--risk-calibration-schedule",
                "calibration.csv",
            ]
        )
        self.assertEqual(risk.risk_model_database, Path("risk.sqlite"))
        self.assertEqual(
            risk.risk_calibration_schedule,
            Path("calibration.csv"),
        )

        reset = weekly_rebalance_v2h.parse_args(
            [
                "--account-id",
                "account_a",
                "--positions",
                "positions.csv",
                "--reset-peak-to-current",
            ]
        )
        self.assertTrue(reset.reset_peak_to_current)


class ForwardQuotationImportTest(unittest.TestCase):
    def test_forward_turnover_is_stored_in_the_historical_percent_unit(self):
        total = import_csmar_forward_quotation.turnover_percent(
            volume=114_093_292.0,
            price=11.10,
            market_value=215_388_000_000.0,
            fallback_decimal=0.00588,
        )
        fallback = import_csmar_forward_quotation.turnover_percent(
            volume=None,
            price=11.10,
            market_value=None,
            fallback_decimal=0.00588,
        )
        self.assertAlmostEqual(total, 0.588, places=3)
        self.assertAlmostEqual(fallback, 0.588)

    def test_forward_market_values_prefer_a_share_float_value(self):
        columns = {
            "MarketValue": 0,
            "CirculatedMarketValue": 1,
            "AValue": 2,
        }
        total, floating = (
            import_csmar_forward_quotation.a_share_market_values_from_row(
                [1000.0, 800.0, 700.0], columns, code="000001"
            )
        )
        self.assertEqual(total, 1000.0)
        self.assertEqual(floating, 700.0)

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

    def test_partial_build_bootstraps_completed_daily_files(self):
        conn = sqlite3.connect(":memory:")
        try:
            build_csmar_database.create_schema(conn)
            record = {
                column: None for column in build_csmar_database.STOCK_DAILY_COLUMNS
            }
            record.update(
                {
                    "code": "000001",
                    "trade_date": "2012-01-04",
                    "close": 10.0,
                    "daily_return": 0.0,
                    "capital_return": 0.0,
                    "no_price_limit": 0,
                    "source_file": "daily\\part1.xlsx",
                    "imported_at": "2026-08-03T00:00:00+08:00",
                }
            )
            build_csmar_database.insert_daily_batch(conn, [record])
            conn.commit()
            build_csmar_database.ensure_resume_schema(conn)
            state = conn.execute(
                "SELECT status, written_rows FROM csmar_daily_import_state "
                "WHERE source_file='daily\\part1.xlsx'"
            ).fetchone()
            self.assertEqual(state, ("complete", 1))
        finally:
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
