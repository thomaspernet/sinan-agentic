"""Tests that the package version is the release tag, with nothing to drift.

A GitHub release publishes to PyPI whatever version the build reports. While the
version lived in ``pyproject.toml`` it was set separately from the tag, and a
release cut without a bump was published on GitHub but never reached PyPI. The
version is now derived from the tag by setuptools-scm, so these tests hold both
halves of that: the manifest carries no version string of its own, and a commit
tagged ``vX.Y.Z`` builds as ``X.Y.Z`` under the manifest's own configuration.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"

RELEASE_VERSION = "9.9.9"
RELEASE_TAG = f"v{RELEASE_VERSION}"

# A throwaway repository commits without the developer's identity or signing.
_GIT_CONFIG = (
    "-c",
    "user.name=release-test",
    "-c",
    "user.email=release-test@example.invalid",
    "-c",
    "commit.gpgsign=false",
    "-c",
    "tag.gpgsign=false",
)


def _project_table() -> str:
    """Return the text of the ``[project]`` table of ``pyproject.toml``.

    ``tomllib`` is 3.11+ and this package supports 3.10, so the table is cut out
    of the manifest text: it runs from its header to the next table header.
    """
    manifest = PYPROJECT.read_text(encoding="utf-8")
    found = re.search(r"^\[project\]\n(.*?)(?=^\[)", manifest, flags=re.MULTILINE | re.DOTALL)
    assert found is not None, "pyproject.toml has no [project] table"
    return found.group(1)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *_GIT_CONFIG, *args], cwd=repo, check=True, capture_output=True)


def _scm_version(repo: Path) -> str:
    """Return the version setuptools-scm derives for *repo* from its ``pyproject.toml``."""
    result = subprocess.run(
        [sys.executable, "-m", "setuptools_scm"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


@pytest.fixture
def tagged_repo(tmp_path: Path) -> Path:
    """A repository holding this package's manifest, its one commit tagged ``RELEASE_TAG``."""
    shutil.copy(PYPROJECT, tmp_path / PYPROJECT.name)
    _git(tmp_path, "init", "--quiet")
    _git(tmp_path, "add", PYPROJECT.name)
    _git(tmp_path, "commit", "--quiet", "-m", "release")
    _git(tmp_path, "tag", RELEASE_TAG)
    return tmp_path


def test_manifest_declares_no_static_version() -> None:
    project = _project_table()
    assert not re.search(
        r"^version\s*=", project, flags=re.MULTILINE
    ), "pyproject.toml hard-codes [project].version; the release tag is the version"
    assert re.search(
        r'^dynamic\s*=\s*\[[^\]]*"version"', project, flags=re.MULTILINE
    ), 'pyproject.toml must declare dynamic = ["version"] for setuptools-scm to set it'


def test_tagged_commit_builds_as_the_tag_version(tagged_repo: Path) -> None:
    assert _scm_version(tagged_repo) == RELEASE_VERSION


def test_commit_past_the_tag_does_not_reuse_the_tag_version(tagged_repo: Path) -> None:
    """An untagged commit builds as a development version, never as the last release."""
    _git(tagged_repo, "commit", "--quiet", "--allow-empty", "-m", "after release")

    version = _scm_version(tagged_repo)

    assert version != RELEASE_VERSION
    assert ".dev" in version
