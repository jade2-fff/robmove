"""Compare identical local planning math; excludes inference, I/O and robot motion."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time

import numpy as np
from reference_oracle import planner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--python-iterations", type=int, default=10000)
    parser.add_argument("--native-iterations", type=int, default=500000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.python_iterations, args.native_iterations, args.repeats) < 1:
        parser.error("counts must be positive")
    legacy = planner()
    pose = np.eye(4)
    pose[:3, 3] = [0.6, -0.1, 0.4]
    for _ in range(100):
        legacy._plan_foundationpose_grasp(pose, "wulongcha")
    python_samples, native_samples = [], []
    for _ in range(args.repeats):
        checksum = 0.0
        start = time.perf_counter_ns()
        for index in range(args.python_iterations):
            pose[0, 3] = 0.6 + (index % 1024) * 1e-6
            plan, _top_down = legacy._plan_foundationpose_grasp(pose, "wulongcha")
            checksum += plan.grasp[0]
        python_samples.append((time.perf_counter_ns() - start) / args.python_iterations)
        expected = sum(0.42 + (i % 1024) * 1e-6 for i in range(args.python_iterations))
        if abs(checksum - expected) > 1e-6:
            raise RuntimeError("legacy benchmark checksum mismatch")
        native = json.loads(subprocess.check_output([str(args.binary.resolve()), "--iterations", str(args.native_iterations)], text=True, timeout=60))
        if native["build_mode"] != "release":
            raise RuntimeError("use the release benchmark binary")
        expected_native = sum(0.42 + (i % 1024) * 1e-6 for i in range(args.native_iterations))
        if abs(native["checksum"] - expected_native) > 1e-6:
            raise RuntimeError("native benchmark checksum mismatch")
        native_samples.append(native["ns_per_plan"])
    result = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "machine": platform.machine(),
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "scope": "CPU grasp-planning kernel only; no SDK, image inference, networking or motion",
        "repeats": args.repeats, "python_iterations_per_repeat": args.python_iterations,
        "native_iterations_per_repeat": args.native_iterations,
        "python_ns_per_plan": python_samples, "cpp_ns_per_plan": native_samples,
        "python_median_ns": statistics.median(python_samples),
        "cpp_median_ns": statistics.median(native_samples),
        "median_speedup": statistics.median(python_samples) / statistics.median(native_samples),
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
