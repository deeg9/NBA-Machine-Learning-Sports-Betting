import math
import unittest

from src.Utils.KLDivergence import (
    kl_divergence_binary,
    symmetric_kl,
    detect_mispricing,
    scan_cross_market,
)


class TestKLDivergenceBinary(unittest.TestCase):
    def test_identical_distributions_returns_zero(self):
        self.assertAlmostEqual(kl_divergence_binary(0.5, 0.5), 0.0, places=6)
        self.assertAlmostEqual(kl_divergence_binary(0.8, 0.8), 0.0, places=6)

    def test_known_value(self):
        # KL(0.5 || 0.8) = 0.5*ln(0.5/0.8) + 0.5*ln(0.5/0.2)
        expected = 0.5 * math.log(0.5 / 0.8) + 0.5 * math.log(0.5 / 0.2)
        result = kl_divergence_binary(0.5, 0.8)
        self.assertAlmostEqual(result, expected, places=5)

    def test_always_non_negative(self):
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for q in [0.1, 0.3, 0.5, 0.7, 0.9]:
                self.assertGreaterEqual(kl_divergence_binary(p, q), 0.0)

    def test_clamping_near_zero(self):
        # Should not raise even with extreme values
        result = kl_divergence_binary(0.0, 1.0)
        self.assertTrue(math.isfinite(result))
        result = kl_divergence_binary(1.0, 0.0)
        self.assertTrue(math.isfinite(result))

    def test_asymmetric(self):
        kl_pq = kl_divergence_binary(0.2, 0.6)
        kl_qp = kl_divergence_binary(0.6, 0.2)
        self.assertNotAlmostEqual(kl_pq, kl_qp, places=3)


class TestSymmetricKL(unittest.TestCase):
    def test_symmetric(self):
        self.assertAlmostEqual(
            symmetric_kl(0.3, 0.7),
            symmetric_kl(0.7, 0.3),
            places=10,
        )

    def test_identical_is_zero(self):
        self.assertAlmostEqual(symmetric_kl(0.5, 0.5), 0.0, places=6)


class TestDetectMispricing(unittest.TestCase):
    def test_returns_sorted_by_divergence(self):
        games = [
            {"ticker": "A", "model_prob": 0.6, "market_price_cents": 55},
            {"ticker": "B", "model_prob": 0.8, "market_price_cents": 50},  # bigger gap
            {"ticker": "C", "model_prob": 0.5, "market_price_cents": 50},  # no gap
        ]
        results = detect_mispricing(games, threshold=0.0)
        self.assertEqual(results[0]["ticker"], "B")

    def test_threshold_filters(self):
        games = [
            {"ticker": "A", "model_prob": 0.51, "market_price_cents": 50},
        ]
        results = detect_mispricing(games, threshold=0.1)
        self.assertEqual(len(results), 0)

    def test_direction_field(self):
        games = [
            {"ticker": "A", "model_prob": 0.7, "market_price_cents": 50},
        ]
        results = detect_mispricing(games, threshold=0.0)
        self.assertEqual(results[0]["direction"], "model_higher")

        games = [
            {"ticker": "B", "model_prob": 0.3, "market_price_cents": 50},
        ]
        results = detect_mispricing(games, threshold=0.0)
        self.assertEqual(results[0]["direction"], "market_higher")


class TestScanCrossMarket(unittest.TestCase):
    def test_empty_with_single_market_type(self):
        markets = [
            {"ticker": "A", "market_type": "moneyline", "implied_prob": 0.6,
             "home_team": "Lakers", "away_team": "Celtics"},
        ]
        self.assertEqual(scan_cross_market(markets), [])

    def test_detects_inconsistency(self):
        markets = [
            {"ticker": "A-ML", "market_type": "moneyline", "implied_prob": 0.6,
             "home_team": "Lakers", "away_team": "Celtics"},
            {"ticker": "A-SP", "market_type": "spread", "implied_prob": 0.8,
             "home_team": "Lakers", "away_team": "Celtics"},
        ]
        results = scan_cross_market(markets)
        self.assertEqual(len(results), 1)
        self.assertGreater(results[0]["kl_divergence"], 0)


if __name__ == "__main__":
    unittest.main()
