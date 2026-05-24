import unittest
import pandas as pd
import tempfile
import os
from dutcher import dutch
from brier_monitor import check_brier, check_ece

class TestACE(unittest.TestCase):
    def test_dutch_no_edge(self):
        # Model = 0.60 combined, Book = 0.75 combined
        model_probs = {"2-0": 0.40, "2-1": 0.20}
        book_odds = {"2-0": 2.0, "2-1": 4.0}
        
        result = dutch(model_probs, book_odds, bankroll=1000)
        self.assertEqual(result["decision"], "NO BET")
        self.assertLessEqual(result["edge"], 0)

    def test_dutch_with_edge(self):
        # Model = 0.80 combined, Book = 0.60 combined
        model_probs = {"2-0": 0.50, "2-1": 0.30}  
        book_odds = {"2-0": 2.50, "2-1": 5.00}
        
        result = dutch(model_probs, book_odds, bankroll=1000, kelly_fraction=0.25)
        self.assertEqual(result["decision"], "BET")
        self.assertGreater(result["edge"], 0.19)
        self.assertIn("stakes", result)
        
        # Verify Equal Profit Dutching
        st_2_0 = result["stakes"]["2-0"]
        st_2_1 = result["stakes"]["2-1"]
        ret_2_0 = st_2_0 * 2.5
        ret_2_1 = st_2_1 * 5.0
        self.assertAlmostEqual(ret_2_0, ret_2_1, delta=0.5)

    def test_brier_fails(self):
        df = pd.DataFrame({
            'match': [1, 2, 3],
            'predicted_prob': [0.1, 0.2, 0.05], # Terrible prediction for actual outcomes (1.0)
            'won': [1, 1, 1]
        })
        f = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
        path = f.name
        f.close()
        df.to_csv(path, index=False)

        with self.assertRaises(SystemExit):
            check_brier(path, window=5, threshold=0.25)
        os.remove(path)

    def test_ece_fails(self):
        # 90% confidence but 0% accuracy
        df = pd.DataFrame({
            'predicted_prob': [0.9] * 50,
            'won': [0] * 50
        })
        f = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
        path = f.name
        f.close()
        df.to_csv(path, index=False)

        with self.assertRaises(SystemExit):
            check_ece(path, window=50, threshold=0.15)
        os.remove(path)

if __name__ == '__main__':
    unittest.main()
