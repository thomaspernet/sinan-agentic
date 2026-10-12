"""Fail unless every built distribution carries the version its release tag names.

Publish builds the package from a release tagged ``vX.Y.Z`` and setuptools-scm
derives the version from that tag. When it cannot see the tag — a shallow
checkout, a tag on another commit — it does not fail: it falls back to a
development version off the nearest earlier tag. This check reads the version
out of each built wheel's and sdist's own metadata and stops the run, before
anything is uploaded, unless it is the tag without its leading ``v``.

Usage: ``python check_dist_version.py <release-tag> <dist-dir>``
"""

from __future__ import annotations

import sys
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath

TAG_PREFIX = "v"
WHEEL_SUFFIX = ".whl"
SDIST_SUFFIX = ".tar.gz"
WHEEL_METADATA_FILE = "METADATA"
WHEEL_METADATA_DIR_SUFFIX = ".dist-info"
SDIST_METADATA_FILE = "PKG-INFO"
VERSION_FIELD = "Version"
# A wheel holds <name>.dist-info/METADATA and an sdist <name>-<version>/PKG-INFO:
# the metadata file sits exactly one directory deep in both archives.
METADATA_DEPTH = 2


def _metadata_version(metadata: str, dist: Path) -> str:
    version = Parser().parsestr(metadata)[VERSION_FIELD]
    if not version:
        raise ValueError(f"{dist.name} carries no {VERSION_FIELD} in its metadata")
    return str(version)


def _is_top_level(member: str, filename: str) -> bool:
    path = PurePosixPath(member)
    return len(path.parts) == METADATA_DEPTH and path.name == filename


def wheel_version(wheel: Path) -> str:
    """Return the version recorded in *wheel*'s ``.dist-info/METADATA``."""
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.namelist():
            if _is_top_level(member, WHEEL_METADATA_FILE) and PurePosixPath(
                member
            ).parent.name.endswith(WHEEL_METADATA_DIR_SUFFIX):
                return _metadata_version(archive.read(member).decode("utf-8"), wheel)
    raise ValueError(f"{wheel.name} holds no {WHEEL_METADATA_DIR_SUFFIX}/{WHEEL_METADATA_FILE}")


def sdist_version(sdist: Path) -> str:
    """Return the version recorded in *sdist*'s top-level ``PKG-INFO``."""
    with tarfile.open(sdist, "r:gz") as archive:
        for member in archive.getmembers():
            if member.isfile() and _is_top_level(member.name, SDIST_METADATA_FILE):
                extracted = archive.extractfile(member)
                if extracted is not None:
                    return _metadata_version(extracted.read().decode("utf-8"), sdist)
    raise ValueError(f"{sdist.name} holds no top-level {SDIST_METADATA_FILE}")


def built_versions(dist_dir: Path) -> dict[str, str]:
    """Return each built distribution's file name mapped to the version it carries.

    A build produces one wheel and one sdist; a directory missing either is an
    error rather than a smaller set to check.
    """
    wheels = sorted(dist_dir.glob(f"*{WHEEL_SUFFIX}"))
    sdists = sorted(dist_dir.glob(f"*{SDIST_SUFFIX}"))
    if not wheels or not sdists:
        raise ValueError(f"{dist_dir} must hold a wheel and an sdist to check")
    versions = {wheel.name: wheel_version(wheel) for wheel in wheels}
    versions.update({sdist.name: sdist_version(sdist) for sdist in sdists})
    return versions


def mismatches(release_tag: str, versions: dict[str, str]) -> list[str]:
    """Return one error line per distribution whose version is not *release_tag*'s."""
    expected = release_tag.removeprefix(TAG_PREFIX)
    return [
        f"Release tag '{release_tag}' expects version '{expected}', "
        f"but {name} was built as '{version}'"
        for name, version in versions.items()
        if version != expected
    ]


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {Path(__file__).name} <release-tag> <dist-dir>", file=sys.stderr)
        return 2
    release_tag, dist_dir = argv[0], Path(argv[1])
    try:
        versions = built_versions(dist_dir)
    except ValueError as error:
        print(f"::error::{error}")
        return 1
    errors = mismatches(release_tag, versions)
    for message in errors:
        print(f"::error::{message}")
    if errors:
        return 1
    for name, version in versions.items():
        print(f"{name}: {version} matches release tag '{release_tag}'")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
