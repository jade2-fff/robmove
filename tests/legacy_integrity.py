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
for item in manifest.get("removed_files", []):
    if (root / item["path"]).exists():
        raise SystemExit(f"Removed file reappeared: {item['path']}")
print(f"Verified {len(manifest['files'])} retained source and asset files")
