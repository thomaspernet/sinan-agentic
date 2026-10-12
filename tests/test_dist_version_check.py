"""Tests that Publish refuses to upload a build whose version is not the release tag's.

setuptools-scm falls back to a development version when it cannot see the
release tag, rather than failing, so Publish reads the version back out of the
built wheel and sdist and stops before the upload unless it is the tag's. These
tests run that check against distributions laid out as ``python -m build``
writes them, and hold the workflow to running it between the build and the
upload.
"""

from __future__ import annotations

import io
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECK_SCRIPT = REPO_ROOT / ".github" / "scripts" / "check_dist_version.py"
PUBLISH_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "publish.yml"

DIST_NAME = "sinan_agentic_core"
RELEASE_VERSION = "9.9.9"
RELEASE_TAG = f"v{RELEASE_VERSION}"
# What setuptools-scm yields when it misses the tag and a configuration drops
# the local part: a version PyPI accepts, so only this check stands in its way.
FALLBACK_VERSION = "9.9.8.dev3"


def _metadata(version: str) -> bytes:
    return f"Metadata-Version: 2.4\nName: sinan-agentic-core\nVersion: {version}\n".encode()


def _write_wheel(dist_dir: Path, version: str) -> None:
    wheel = dist_dir / f"{DIST_NAME}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"{DIST_NAME}/__init__.py", "")
        archive.writestr(f"{DIST_NAME}-{version}.dist-info/METADATA", _metadata(version))


def _add_file(archive: tarfile.TarFile, name: str, content: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(content)
    archive.addfile(member, io.BytesIO(content))


def _write_sdist(dist_dir: Path, version: str, egg_info_version: str | None = None) -> None:
    root = f"{DIST_NAME}-{version}"
    with tarfile.open(dist_dir / f"{root}.tar.gz", "w:gz") as archive:
        _add_file(archive, f"{root}/PKG-INFO", _metadata(version))
        _add_file(
            archive,
            f"{root}/{DIST_NAME}.egg-info/PKG-INFO",
            _metadata(egg_info_version or version),
        )


def _check(release_tag: str, dist_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), release_tag, str(dist_dir)],
        capture_output=True,
        text=True,
    )


@pytest.fixture
def dist_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "dist"
    directory.mkdir()
    return directory


def test_build_carrying_the_tag_version_passes(dist_dir: Path) -> None:
    _write_wheel(dist_dir, RELEASE_VERSION)
    _write_sdist(dist_dir, RELEASE_VERSION)

    result = _check(RELEASE_TAG, dist_dir)

    assert result.returncode == 0, result.stdout + result.stderr


def test_wheel_built_as_a_fallback_version_fails_naming_tag_and_version(dist_dir: Path) -> None:
    _write_wheel(dist_dir, FALLBACK_VERSION)
    _write_sdist(dist_dir, RELEASE_VERSION)

    result = _check(RELEASE_TAG, dist_dir)

    assert result.returncode == 1
    assert RELEASE_TAG in result.stdout
    assert FALLBACK_VERSION in result.stdout


def test_sdist_built_as_a_fallback_version_fails(dist_dir: Path) -> None:
    _write_wheel(dist_dir, RELEASE_VERSION)
    _write_sdist(dist_dir, FALLBACK_VERSION)

    result = _check(RELEASE_TAG, dist_dir)

    assert result.returncode == 1
    assert FALLBACK_VERSION in result.stdout


def test_sdist_version_is_read_from_its_top_level_pkg_info(dist_dir: Path) -> None:
    """The egg-info copy nested inside the sdist does not decide its version."""
    _write_wheel(dist_dir, RELEASE_VERSION)
    _write_sdist(dist_dir, FALLBACK_VERSION, egg_info_version=RELEASE_VERSION)

    result = _check(RELEASE_TAG, dist_dir)

    assert result.returncode == 1
    assert FALLBACK_VERSION in result.stdout


def test_build_missing_a_distribution_fails(dist_dir: Path) -> None:
    _write_wheel(dist_dir, RELEASE_VERSION)

    result = _check(RELEASE_TAG, dist_dir)

    assert result.returncode == 1


def _build_steps() -> list[dict[str, Any]]:
    workflow = yaml.safe_load(PUBLISH_WORKFLOW.read_text(encoding="utf-8"))
    steps: list[dict[str, Any]] = workflow["jobs"]["build"]["steps"]
    return steps


def _step_index(steps: list[dict[str, Any]], field: str, needle: str) -> int:
    return next(i for i, step in enumerate(steps) if needle in str(step.get(field, "")))


def test_publish_checks_the_version_between_build_and_upload() -> None:
    steps = _build_steps()
    check_path = str(CHECK_SCRIPT.relative_to(REPO_ROOT))

    build = _step_index(steps, "run", "python -m build")
    check = _step_index(steps, "run", check_path)
    upload = _step_index(steps, "uses", "actions/upload-artifact")

    assert build < check < upload
    assert "github.event.release.tag_name" in steps[check]["env"]["RELEASE_TAG"]
