"""Perception-only CLI tests; no SDK, cameras or model inference required."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

BINARY = str(Path(sys.argv.pop(1)).resolve())
ROOT = Path(sys.argv.pop(1)).resolve()


class CliTest(unittest.TestCase):
    def run_cli(self, *args, cwd=None):
        return subprocess.run([BINARY, *args], text=True, capture_output=True, cwd=cwd, timeout=10)

    def records(self, process):
        return [json.loads(line) for line in process.stdout.splitlines()]

    def test_replay_preserves_all_observations_and_candidate_poses(self):
        source = ROOT / 'examples/replay.json'
        expected = json.loads(source.read_text())['frames']
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_cli('--replay', str(source), '--json', cwd=tmp)
        self.assertEqual(result.returncode, 0, result.stderr)
        records = self.records(result)
        self.assertEqual(len(records[:-1]), len(expected))
        for index, (observed, original) in enumerate(zip(records[:-1], expected)):
            self.assertEqual(observed['type'], 'observation')
            self.assertEqual(observed['index'], index)
            self.assertEqual(observed['status'], original['status'])
            self.assertEqual(observed['candidates'], original.get('candidates', []))
        self.assertEqual(records[-1]['frames'], 4)
        self.assertEqual(records[-1]['detections'], 2)

    def test_demo_emits_observations_without_execution_events(self):
        result = self.run_cli('--demo', '--json')
        self.assertEqual(result.returncode, 0, result.stderr)
        records = self.records(result)
        self.assertEqual({item['type'] for item in records}, {'observation', 'summary'})
        self.assertTrue(all(item['backend'] == 'replay' for item in records))
        self.assertEqual(records[-1]['frames'], 4)
        self.assertEqual(records[-1]['detections'], 1)

    def test_removed_motion_options_are_rejected_without_writing_a_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / 'commands.json'
            for args in [('--demo', '--mode', 'single'), ('--demo', '--trace', str(trace)),
                         ('--demo', '--realtime')]:
                with self.subTest(args=args):
                    result = self.run_cli(*args)
                    self.assertEqual(result.returncode, 1, result.stderr)
                    self.assertIn('unknown option', result.stderr)
            self.assertFalse(trace.exists())

    def test_empty_stream_and_unavailable_frames_do_not_imply_valid_perception(self):
        for frames in [[], [{'status': 'unavailable', 'reason': 'no depth'}, {'status': 'empty'}]]:
            with self.subTest(frames=frames), tempfile.TemporaryDirectory() as tmp:
                source = Path(tmp) / 'frames.json'
                source.write_text(json.dumps({'frames': frames}))
                result = self.run_cli('--replay', str(source), '--json')
                self.assertEqual(result.returncode, 2, result.stderr)
                summary = self.records(result)[-1]
                self.assertEqual(summary['frames'], len(frames))
                self.assertEqual(summary['unavailable_frames'], int(bool(frames)))

    def test_motion_configuration_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config.json'
            for section in ('robot', 'navigation', 'task'):
                config.write_text(json.dumps({section: {}}))
                with self.subTest(section=section):
                    self.assertEqual(self.run_cli('--config', str(config), '--validate-config').returncode, 1)

    def test_invalid_inputs_return_startup_error(self):
        for args in [('--robot',), ('--demo', '--replay', 'x'), ('--replay',),
                     ('--demo', '--demo'), ('--config', '/missing-config.json', '--validate-config')]:
            with self.subTest(args=args):
                self.assertEqual(self.run_cli(*args).returncode, 1)

    def test_config_check_and_help_need_no_devices_or_models(self):
        result = self.run_cli('--config', str(ROOT / 'config/default.json'), '--validate-config', '--json')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.records(result)[0]['valid'])
        self.assertEqual(self.run_cli('--help').returncode, 0)


if __name__ == '__main__':
    unittest.main()
