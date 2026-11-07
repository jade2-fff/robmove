"""Offline checks for the retained Python perception configuration."""
import builtins
from pathlib import Path
import runpy
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class PerceptionConfigurationTest(unittest.TestCase):
    def load_config(self):
        original_import = builtins.__import__

        def reject_sdk(name, *args, **kwargs):
            if name == 'galbot_sdk' or name.startswith('galbot_sdk.'):
                raise AssertionError('Reading perception configuration must not import a device SDK')
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=reject_sdk):
            return runpy.run_path(str(ROOT / 'pick_place_pipeline/config.py'))

    def test_configuration_loads_without_sdk_and_resolves_retained_assets(self):
        config = self.load_config()['cfg']
        config.perception.validate_foundationpose_candidates()
        self.assertTrue((ROOT / config.perception.mask_weights).is_file())
        self.assertEqual(len(config.perception.foundationpose_objects), 5)
        for item in config.perception.foundationpose_objects.values():
            self.assertTrue((ROOT / 'pick_place_pipeline' / item.mesh_path).is_file())

    def test_perception_configuration_exposes_no_actuator_controls(self):
        config = self.load_config()['cfg']
        for attribute in ('robot', 'navigation', 'task', 'arm_name', 'gripper_name', 'joint_speed'):
            with self.subTest(attribute=attribute):
                self.assertFalse(hasattr(config, attribute))


if __name__ == '__main__':
    unittest.main()
