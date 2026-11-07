"""End-to-end tests; stdlib only, no SDK or Python inference dependencies."""
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

BINARY = str(Path(sys.argv.pop(1)).resolve())
ROOT = Path(sys.argv.pop(1)).resolve()


class CliTest(unittest.TestCase):
    def run_cli(self, *args, cwd=None):
        return subprocess.run([BINARY, *args], text=True, capture_output=True, cwd=cwd, timeout=10)

    def summary(self, process):
        return json.loads(process.stdout.splitlines()[-1])

    def test_demo_single_and_trace_are_explicitly_simulated(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.json"
            result = self.run_cli("--demo", "--json", "--trace", str(trace), cwd=tmp)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(self.summary(result)["backend"], "simulated")
            self.assertEqual(self.summary(result)["placed_count"], 1)
            commands = json.loads(trace.read_text())
            self.assertEqual(commands[-1]["operation"], "shutdown")
            self.assertTrue(all(c["collision_check"] for c in commands if c["operation"] == "move_pose"))

    def test_cyclic_demo_with_external_config_from_another_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_cli("--demo", "--mode", "cyclic", "--config", str(ROOT / "config/simulation.json"), "--json", cwd=tmp)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(self.summary(result)["placed_count"], 1)

    def test_empty_file_is_not_a_confirmed_empty_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            replay = Path(tmp) / "empty-stream.json"
            replay.write_text('{"schema_version":1,"frames":[]}')
            result = self.run_cli("--replay", str(replay), "--json")
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertTrue(self.summary(result)["held"])

    def test_invalid_inputs_return_startup_error(self):
        for args in [("--mode", "typo"), ("--robot",), ("--demo", "--replay", "x"),
                     ("--demo", "--mode", "cyclic"), ("--config", "/missing-config.json", "--validate-config")]:
            with self.subTest(args=args):
                self.assertEqual(self.run_cli(*args).returncode, 1)

    def test_config_check_and_help_need_no_robot_or_models(self):
        result = self.run_cli("--config", str(ROOT / "config/default.json"), "--validate-config", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.summary(result)["valid"])
        self.assertEqual(self.run_cli("--help").returncode, 0)

    @unittest.skipIf(sys.platform == "win32", "POSIX signal test")
    def test_interrupt_holds_and_returns_130(self):
        process = subprocess.Popen([BINARY, "--demo", "--realtime", "--json"], text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            time.sleep(0.1)
            process.send_signal(signal.SIGINT)
            output, error = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 130, error)
            summary = json.loads(output.splitlines()[-1])
            self.assertTrue(summary["held"])
            self.assertTrue(summary["cancelled"])
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


if __name__ == "__main__":
    unittest.main()
