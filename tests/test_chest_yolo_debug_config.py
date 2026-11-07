import unittest
from pathlib import Path

from tools.chest_yolo_debug_config import load_config, minimum_confidence, selected_indices


class ChestYoloDebugConfigTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config(Path(__file__).parents[1] / 'config/chest_yolo_debug.yaml')

    def test_person_requires_at_least_55_percent_without_raising_bottle_threshold(self):
        self.assertEqual(
            selected_indices(
                {0: 'person', 39: 'bottle'},
                [0, 0, 0, 39, 39],
                [0.5499, 0.55, 0.9, 0.3999, 0.4],
                self.config['confidence'],
            ),
            [1, 2, 4],
        )

    def test_inference_keeps_candidates_for_a_lower_per_class_override(self):
        thresholds = {'default': 0.40, 'per_class': {'person': 0.55, 'cup': 0.25}}
        self.assertEqual(minimum_confidence(thresholds), 0.25)
        self.assertEqual(selected_indices({0: 'person', 41: 'cup'}, [0, 41], [0.5, 0.3], thresholds), [1])


if __name__ == '__main__':
    unittest.main()
