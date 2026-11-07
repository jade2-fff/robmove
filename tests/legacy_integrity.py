"""Verify retained source and assets against the current migration manifest."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
manifest = json.loads((root / "migration_manifest.json").read_text())
for item in manifest["files"]:
    path = root / item["path"]
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != item["sha256"]:
        raise SystemExit(f"Manifest file changed: {item['path']}")
golden = json.loads((root / "tests/fixtures/planning_golden.json").read_text())
for name, expected in golden["source_sha256"].items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
        raise SystemExit(f"Reference fixture is stale: {name}")
print(f"Verified {len(manifest['files'])} retained files and reference-source hashes")
