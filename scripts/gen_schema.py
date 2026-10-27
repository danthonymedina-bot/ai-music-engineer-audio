"""Regenerate schema/edit_model.json from app/edit/model.py (the Python model is the single source of truth).

    python scripts/gen_schema.py          # write
    python scripts/gen_schema.py --check  # exit 1 if the committed file is stale (used by tests)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.edit.model import SCHEMA_VERSION, EditHistory, Project  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "schema" / "edit_model.json"


def build() -> dict:
    s = Project.model_json_schema(ref_template="#/$defs/{model}")
    hist = EditHistory.model_json_schema(ref_template="#/$defs/{model}")
    defs = {**s.pop("$defs", {}), **hist.pop("$defs", {})}
    defs["EditHistory"] = hist
    defs["Project"] = s
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://github.com/danthonymedina-bot/ai-music-engineer-audio/schema/edit_model.json",
        "title": "AI Music Engineer edit model",
        "description": "Project JSON sent to POST /edit/render and persisted by the Android app. "
                       "All positions are integer samples at Project.sampleRate; ranges are half-open.",
        "x-schemaVersion": SCHEMA_VERSION,
        "$ref": "#/$defs/Project",
        "$defs": dict(sorted(defs.items())),
    }


def render() -> str:
    return json.dumps(build(), indent=2, sort_keys=False) + "\n"


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(0 if OUT.exists() and OUT.read_text() == render() else 1)
    OUT.write_text(render())
    print(f"wrote {OUT}")
