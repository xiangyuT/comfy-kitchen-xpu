from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest


_BUILDER = (
    Path(__file__).parents[1]
    / "packaging"
    / "xpu_runtime_provider"
    / "build_wheel.py"
)


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "comfy_kitchen_xpu_runtime_wheel_builder", _BUILDER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _source_wheel(path: Path, *, distribution: str = "comfy-kitchen") -> Path:
    dist_info = "comfy_kitchen-0.2.35.dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"{dist_info}/METADATA",
            "Metadata-Version: 2.4\n"
            f"Name: {distribution}\n"
            "Version: 0.2.35\n"
            "Requires-Python: >=3.10\n",
        )
        archive.writestr(
            f"{dist_info}/WHEEL",
            "Wheel-Version: 1.0\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n",
        )
        archive.writestr("comfy_kitchen/__init__.py", "VALUE = 'xpu'\n")
        archive.writestr(
            "comfy_kitchen/backends/xpu/__init__.py", "AVAILABLE = True\n"
        )
        archive.writestr(f"{dist_info}/RECORD", "")
    return path


@pytest.mark.parametrize("xpu_target", ["bmg", "ptl-h", "dg2"])
def test_provider_wheel_has_disjoint_top_level_and_verified_manifest(
    tmp_path, monkeypatch, xpu_target
):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
    builder = _load_builder()
    source = _source_wheel(tmp_path / "comfy_kitchen-0.2.35-py3-none-any.whl")

    provider = builder.build_provider_wheel(
        source_wheel=source,
        output_directory=tmp_path / "dist",
        source_revision="a" * 40,
        torch_version="2.13.0+xpu",
        xpu_target=xpu_target,
    )

    assert provider.name == "comfy_kitchen_xpu_runtime-0.2.35-py3-none-any.whl"
    with zipfile.ZipFile(provider) as archive:
        names = set(archive.namelist())
        assert not any(name.startswith("comfy_kitchen/") for name in names)
        vendored = (
            "comfy_kitchen_xpu_runtime/_vendor/comfy_kitchen/__init__.py"
        )
        assert vendored in names
        assert (
            "comfy_kitchen_xpu_runtime-0.2.35.dist-info/entry_points.txt"
            in names
        )
        manifest = json.loads(
            archive.read("comfy_kitchen_xpu_runtime/provider.json")
        )
        assert manifest["provider_id"] == "comfy_kitchen.xpu"
        assert manifest["canonical_import"] == "comfy_kitchen"
        assert manifest["canonical_distribution"] == {
            "name": "comfy-kitchen",
            "compatible_versions": ["0.2.35"],
        }
        assert manifest["source"]["revision"] == "a" * 40
        assert manifest["source"]["wheel_sha256"] == hashlib.sha256(
            source.read_bytes()
        ).hexdigest()
        assert manifest["runtime"] == {
            "torch_version": "2.13.0+xpu",
            "torch_build": "xpu",
            "xpu_targets": [xpu_target],
            "platforms": ["linux", "win32"],
        }
        vendored_bytes = archive.read(vendored)
        assert manifest["vendored_files"][vendored] == hashlib.sha256(
            vendored_bytes
        ).hexdigest()

        entry_points = archive.read(
            "comfy_kitchen_xpu_runtime-0.2.35.dist-info/entry_points.txt"
        ).decode()
        assert "[comfyui_omnixpu.runtime_providers]" in entry_points
        assert (
            "comfy_kitchen.xpu = "
            "comfy_kitchen_xpu_runtime.provider:get_manifest"
        ) in entry_points

        record = list(
            csv.reader(
                io.StringIO(
                    archive.read(
                        "comfy_kitchen_xpu_runtime-0.2.35.dist-info/RECORD"
                    ).decode()
                )
            )
        )
        recorded_paths = {row[0] for row in record}
        assert recorded_paths == names


def test_provider_wheel_is_reproducible(tmp_path, monkeypatch):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
    builder = _load_builder()
    source = _source_wheel(tmp_path / "comfy_kitchen-0.2.35-py3-none-any.whl")
    arguments = {
        "source_wheel": source,
        "source_revision": "b" * 40,
        "torch_version": "2.13.0+xpu",
        "xpu_target": "ptl-h",
    }

    first = builder.build_provider_wheel(
        output_directory=tmp_path / "first", **arguments
    )
    second = builder.build_provider_wheel(
        output_directory=tmp_path / "second", **arguments
    )

    assert first.read_bytes() == second.read_bytes()


def test_provider_builder_rejects_an_unrelated_distribution(tmp_path):
    builder = _load_builder()
    source = _source_wheel(
        tmp_path / "unrelated-0.2.35-py3-none-any.whl",
        distribution="unrelated",
    )

    with pytest.raises(RuntimeError, match="expected a comfy-kitchen wheel"):
        builder.build_provider_wheel(
            source_wheel=source,
            output_directory=tmp_path / "dist",
            source_revision="c" * 40,
            torch_version="2.13.0+xpu",
            xpu_target="bmg",
        )


@pytest.mark.parametrize("target", ["a770", "unknown", "DG2"])
def test_provider_builder_rejects_unknown_target(tmp_path, target):
    builder = _load_builder()
    source = _source_wheel(tmp_path / "comfy_kitchen-0.2.35-py3-none-any.whl")
    with pytest.raises(ValueError, match="xpu target must be"):
        builder.build_provider_wheel(
            source_wheel=source, output_directory=tmp_path / "dist",
            source_revision="c" * 40, torch_version="2.13.0+xpu",
            xpu_target=target,
        )


@pytest.mark.parametrize("target", ["bmg", "ptl-h", "dg2"])
def test_cli_accepts_supported_target(tmp_path, monkeypatch, target):
    builder = _load_builder()
    monkeypatch.setattr(sys, "argv", [
        str(_BUILDER), "--source-wheel", str(tmp_path / "source.whl"),
        "--output-dir", str(tmp_path / "dist"),
        "--source-revision", "d" * 40, "--torch-version", "2.13.0+xpu",
        "--xpu-target", target,
    ])
    assert builder._parse_args().xpu_target == target
