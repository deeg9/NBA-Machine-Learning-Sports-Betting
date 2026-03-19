import unittest

from src.Utils.BayesianUpdater import (
    bayesian_update,
    line_movement_lr,
    injury_lr,
    reverse_line_lr,
    update_probability,
)


class TestBayesianUpdate(unittest.TestCase):
    def test_lr_one_returns_prior(self):
        self.assertAlmostEqual(bayesian_update(0.5, 1.0), 0.5, places=4)
        self.assertAlmostEqual(bayesian_update(0.7, 1.0), 0.7, places=4)

    def test_lr_above_one_increases(self):
        posterior = bayesian_update(0.5, 1.5)
        self.assertGreater(posterior, 0.5)

    def test_lr_below_one_decreases(self):
        posterior = bayesian_update(0.5, 0.5)
        self.assertLess(posterior, 0.5)

    def test_clamping_near_zero(self):
        # Prior at 0 gets clamped to 0.01
        posterior = bayesian_update(0.0, 0.5)
        self.assertGreater(posterior, 0.0)

    def test_clamping_near_one(self):
        posterior = bayesian_update(1.0, 2.0)
        self.assertLess(posterior, 1.0)

    def test_negative_lr_returns_prior(self):
        self.assertAlmostEqual(bayesian_update(0.5, -1.0), 0.5, places=4)


class TestLineMovementLR(unittest.TestCase):
    def test_no_change_returns_one(self):
        self.assertEqual(line_movement_lr(50, 50), 1.0)

    def test_price_up_favors_home(self):
        lr = line_movement_lr(55, 50)
        self.assertGreater(lr, 1.0)

    def test_price_down_hurts_home(self):
        lr = line_movement_lr(45, 50)
        self.assertLess(lr, 1.0)

    def test_strength_amplifies(self):
        lr_normal = line_movement_lr(55, 50, strength=1.0)
        lr_strong = line_movement_lr(55, 50, strength=2.0)
        self.assertGreater(lr_strong - 1.0, lr_normal - 1.0)


class TestInjuryLR(unittest.TestCase):
    def test_no_injuries_returns_one(self):
        self.assertEqual(injury_lr(0, is_home=True), 1.0)

    def test_home_injuries_hurt_home(self):
        lr = injury_lr(1, is_home=True)
        self.assertLess(lr, 1.0)

    def test_away_injuries_help_home(self):
        lr = injury_lr(1, is_home=False)
        self.assertGreater(lr, 1.0)

    def test_multiple_injuries_compound(self):
        lr_one = injury_lr(1, is_home=True)
        lr_two = injury_lr(2, is_home=True)
        self.assertLess(lr_two, lr_one)


class TestReverseLineLR(unittest.TestCase):
    def test_no_movement_returns_one(self):
        self.assertEqual(reverse_line_lr(0, "yes"), 1.0)

    def test_reverse_line_detected(self):
        # Price goes up but volume on NO side → sharp money on YES
        lr = reverse_line_lr(5, "no")
        self.assertGreater(lr, 1.0)

    def test_aligned_movement_returns_one(self):
        # Price up, volume on YES → no reverse line signal
        lr = reverse_line_lr(5, "yes")
        self.assertEqual(lr, 1.0)


class TestUpdateProbability(unittest.TestCase):
    def test_no_signals_returns_prior(self):
        posterior, log = update_probability(0.6, None, None, None)
        self.assertAlmostEqual(posterior, 0.6, places=2)

    def test_max_shift_cap(self):
        # Aggressive signals that would shift more than 15%
        snapshot = {"current_price": 80, "previous_price": 50, "yes_volume": 100, "no_volume": 10}
        injury = {"home_out_delta": 0, "away_out_delta": 3}
        config = {"use_line_movement": True, "use_injuries": True,
                  "use_reverse_line": True, "line_strength": 1.0,
                  "injury_strength": 1.0, "max_total_shift": 0.15}

        posterior, log = update_probability(0.5, snapshot, injury, config)
        self.assertLessEqual(abs(posterior - 0.5), 0.151)  # within cap + rounding

    def test_disabled_signals_no_effect(self):
        snapshot = {"current_price": 70, "previous_price": 50}
        config = {"use_line_movement": False, "use_injuries": False,
                  "use_reverse_line": False, "max_total_shift": 0.15}
        posterior, log = update_probability(0.5, snapshot, None, config)
        self.assertAlmostEqual(posterior, 0.5, places=2)

    def test_backtest_mode_skips_market_signals(self):
        snapshot = {"current_price": 70, "previous_price": 50, "yes_volume": 100, "no_volume": 10}
        posterior, log = update_probability(0.5, snapshot, None, None, backtest_mode=True)
        # Market signals skipped → no change
        signal_names = [l.get("signal") for l in log]
        self.assertNotIn("line_movement", signal_names)
        self.assertNotIn("reverse_line", signal_names)

    def test_update_log_records_signals(self):
        snapshot = {"current_price": 55, "previous_price": 50}
        posterior, log = update_probability(0.5, snapshot, None, None)
        self.assertTrue(len(log) > 0)
        self.assertEqual(log[0]["signal"], "line_movement")


if __name__ == "__main__":
    unittest.main()
