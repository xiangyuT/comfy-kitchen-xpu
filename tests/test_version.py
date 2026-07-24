import re
from pathlib import Path

import comfy_kitchen


def test_source_version_matches_project_metadata():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    match = re.search(
        r'^version\s*=\s*["\']([^"\']+)["\']\s*$',
        pyproject.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )

    assert match is not None
    assert comfy_kitchen.__version__ == match.group(1)


def test_xpu_wheel_has_distinct_local_version():
    assert comfy_kitchen.__version__ == "0.2.18+xpu.1"
