"""Extract pure legacy math for migration verification, without importing SDK code.

Uses only explicitly selected AST nodes from the preserved local Python source.
NumPy and SciPy are optional development dependencies; C++ builds don't need them.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


class NullLogger:
    def error(self, *_args):
        pass


def planner():
    source = ROOT / "pick_place_pipeline/pick_place_task.py"
    tree = ast.parse(source.read_text())
    methods = {"_plan_foundationpose_grasp", "_build_foundationpose_grasp", "_foundationpose_approach",
               "_lift_pose", "_lift_safe_pose"}
    selected = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "GraspPlan":
            selected.append(node)
        if isinstance(node, ast.ClassDef) and node.name == "PickPlaceTask":
            node.name = "LegacyPlanner"
            node.body = [item for item in node.body if isinstance(item, ast.FunctionDef) and item.name in methods]
            selected.append(node)
    namespace = {"np": np, "dataclass": dataclass, "List": List, "Optional": Optional, "Tuple": Tuple,
                 "FoundationPoseObjectConfig": object, "PickTarget": object, "logger": NullLogger(),
                 "rot_to_quat": lambda matrix: Rotation.from_matrix(matrix).as_quat()}
    # The code is the user's local source under review, never network or replay input.
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
    config_source = ROOT / "pick_place_pipeline/config.py"
    config_tree = ast.parse(config_source.read_text())
    config_nodes = [node for node in config_tree.body if isinstance(node, ast.ClassDef)
                    and node.name in {"FoundationPoseObjectConfig", "PerceptionConfig", "RobotConfig"}]
    from dataclasses import field
    import math
    from typing import Dict
    config_namespace = {"dataclass": dataclass, "field": field, "math": math, "Dict": Dict,
                        "List": List, "Tuple": Tuple, "Optional": Optional}
    exec(compile(ast.Module(body=config_nodes, type_ignores=[]), str(config_source), "exec"), config_namespace)
    instance = namespace["LegacyPlanner"]()
    robot = config_namespace["RobotConfig"]()
    instance.config = SimpleNamespace(**vars(robot), perception=config_namespace["PerceptionConfig"]())
    instance.config.debug_mode = False
    instance.perception = None
    return instance


def generate(output: Path):
    legacy = planner()
    rng = np.random.default_rng(20261107)
    materials = list(legacy.config.perception.foundationpose_objects)
    cases = []
    # Include axis-aligned, inverted and nearly horizontal objects as well as random rotations.
    rotations = [Rotation.identity(), Rotation.from_euler("y", 90, degrees=True),
                 Rotation.from_euler("x", 180, degrees=True), Rotation.from_euler("y", 44.9, degrees=True),
                 Rotation.from_euler("y", 45.1, degrees=True)]
    rotations += list(Rotation.random(195, random_state=rng))
    for index, rotation in enumerate(rotations):
        legacy.config.arm_name = "left_arm" if index % 2 else "right_arm"
        legacy.config.work_angle_min_deg = -20 if index % 3 else -10
        legacy.config.work_angle_max_deg = 5 if index % 3 else 10
        position = rng.uniform([0.3, -0.4, 0.2], [1.0, 0.4, 0.8])
        pose = np.eye(4)
        pose[:3, :3] = rotation.as_matrix()
        pose[:3, 3] = position
        material = materials[index % len(materials)]
        result = legacy._plan_foundationpose_grasp(pose, material)
        case = {"class_name": material, "pose": [*position.tolist(), *rotation.as_quat().tolist()],
                "arm": legacy.config.arm_name, "min_angle": legacy.config.work_angle_min_deg,
                "max_angle": legacy.config.work_angle_max_deg}
        if result is None:
            case["expected"] = None
        else:
            plan, top_down = result
            target = SimpleNamespace(class_name=material, plan=plan, top_down=top_down)
            lifted = legacy._lift_pose(plan.grasp)
            case["expected"] = {"pre_grasp": plan.pre_grasp, "grasp": plan.grasp, "top_down": bool(top_down),
                                "lift_safe": legacy._lift_safe_pose(target, lifted)}
        cases.append(case)
    sources = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in [ROOT / "pick_place_pipeline/pick_place_task.py", ROOT / "pick_place_pipeline/config.py"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"seed": 20261107, "source_sha256": sources, "cases": cases}, indent=2) + "\n")
    print(f"Wrote {len(cases)} legacy reference cases to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "tests/fixtures/planning_golden.json")
    generate(parser.parse_args().output)
