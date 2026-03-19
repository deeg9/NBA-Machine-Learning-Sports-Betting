import math
import unittest

from src.Utils.StoikovExecution import (
    estimate_volatility,
    reservation_price,
    optimal_limit_price,
)


class TestEstimateVolatility(unittest.TestCase):
    def test_analytical_at_50c(self):
        vol = estimate_volatility(50)
        self.assertAlmostEqual(vol, 0.5, places=2)

    def test_analytical_at_extreme(self):
        vol_90 = estimate_volatility(90)
        vol_50 = estimate_volatility(50)
        self.assertLess(vol_90, vol_50)

    def test_with_price_history(self):
        # Stable prices → low vol
        history = [50, 50, 51, 50, 50]
        vol = estimate_volatility(50, price_history=history)
        self.assertLess(vol, 0.1)

    def test_short_history_uses_analytical(self):
        # Less than 3 points → fallback to analytical
        vol = estimate_volatility(50, price_history=[50, 51])
        self.assertAlmostEqual(vol, 0.5, places=2)


class TestReservationPrice(unittest.TestCase):
    def test_zero_inventory_equals_fair_value(self):
        res = reservation_price(
            fair_value_cents=60.0,
            inventory=0,
            gamma=0.1,
            sigma=0.5,
            time_remaining=0.5,
        )
        self.assertAlmostEqual(res, 60.0, places=5)

    def test_positive_inventory_lowers_price(self):
        # Long YES → willing to sell cheaper (reservation drops)
        res = reservation_price(
            fair_value_cents=60.0,
            inventory=5,
            gamma=0.1,
            sigma=0.5,
            time_remaining=0.5,
        )
        self.assertLess(res, 60.0)

    def test_negative_inventory_raises_price(self):
        # Short YES → willing to buy higher (reservation rises)
        res = reservation_price(
            fair_value_cents=60.0,
            inventory=-5,
            gamma=0.1,
            sigma=0.5,
            time_remaining=0.5,
        )
        self.assertGreater(res, 60.0)

    def test_time_zero_no_penalty(self):
        res = reservation_price(
            fair_value_cents=60.0,
            inventory=10,
            gamma=0.1,
            sigma=0.5,
            time_remaining=0.0,
        )
        self.assertAlmostEqual(res, 60.0, places=5)


class TestOptimalLimitPrice(unittest.TestCase):
    def test_output_in_valid_range(self):
        result = optimal_limit_price(
            model_prob=0.6,
            current_position=0,
            time_to_close_hours=5.0,
            orderbook={"yes_bid": 58, "yes_ask": 62, "no_bid": 38, "no_ask": 42},
            side="yes",
        )
        self.assertGreaterEqual(result["optimal_price"], 1)
        self.assertLessEqual(result["optimal_price"], 99)

    def test_max_improvement_cap(self):
        result = optimal_limit_price(
            model_prob=0.6,
            current_position=0,
            time_to_close_hours=10.0,
            orderbook={"yes_bid": 58, "yes_ask": 62, "no_bid": 38, "no_ask": 42},
            side="yes",
            gamma=1.0,  # very conservative → large improvement
            max_improvement=3,
        )
        naive_price = 62  # yes_ask
        self.assertGreaterEqual(result["optimal_price"], naive_price - 3)

    def test_near_close_bypasses_stoikov(self):
        result = optimal_limit_price(
            model_prob=0.6,
            current_position=0,
            time_to_close_hours=0.05,  # 3 minutes
            orderbook={"yes_bid": 58, "yes_ask": 62, "no_bid": 38, "no_ask": 42},
            side="yes",
        )
        self.assertEqual(result["improvement_cents"], 0)
        self.assertAlmostEqual(result["fill_probability"], 1.0)

    def test_no_orderbook_uses_model(self):
        result = optimal_limit_price(
            model_prob=0.6,
            current_position=0,
            time_to_close_hours=5.0,
            orderbook={},
            side="yes",
        )
        # Should still produce valid output
        self.assertGreaterEqual(result["optimal_price"], 1)
        self.assertLessEqual(result["optimal_price"], 99)

    def test_fill_probability_decreases_with_improvement(self):
        result_close = optimal_limit_price(
            model_prob=0.6,
            current_position=0,
            time_to_close_hours=5.0,
            orderbook={"yes_bid": 58, "yes_ask": 62},
            side="yes",
            gamma=0.01,
            max_improvement=5,
        )
        result_far = optimal_limit_price(
            model_prob=0.6,
            current_position=0,
            time_to_close_hours=5.0,
            orderbook={"yes_bid": 58, "yes_ask": 62},
            side="yes",
            gamma=0.5,
            max_improvement=5,
        )
        # More aggressive gamma → more improvement → lower fill prob
        if result_far["improvement_cents"] > result_close["improvement_cents"]:
            self.assertLessEqual(
                result_far["fill_probability"],
                result_close["fill_probability"],
            )

    def test_no_side(self):
        result = optimal_limit_price(
            model_prob=0.4,
            current_position=0,
            time_to_close_hours=5.0,
            orderbook={"yes_bid": 38, "yes_ask": 42, "no_bid": 58, "no_ask": 62},
            side="no",
        )
        self.assertGreaterEqual(result["optimal_price"], 1)
        self.assertLessEqual(result["optimal_price"], 99)


if __name__ == "__main__":
    unittest.main()
