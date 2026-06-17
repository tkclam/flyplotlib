"""Vendor NeuroMechFly v2 assets from flygym into flyplotlib.

This is a *maintainer* script, run only when refreshing the bundled assets. It
downloads the simplified body meshes and the model metadata (rigging, neutral
pose, colors) from the flygym repository and writes them into
``src/flyplotlib/data/neuromechfly/`` in a runtime-friendly form:

* ``meshes/*.stl``    -- one mesh per left/center body segment (simplified where
                         available, full-resolution otherwise). Right-side
                         segments reuse the mirrored left mesh at runtime.
* ``rigging.json``    -- ``{segment: {"pos": [3], "quat": [w, x, y, z]}}``.
* ``neutral_pose.json`` -- ``{dof_name: angle_deg}`` for the default standing
                         pose (left/center only; right derived by mirroring).
* ``colors.json``     -- ``{segment: [r, g, b(, a)]}`` base colors.
* ``NOTICE``          -- provenance + flygym (Apache-2.0) attribution.

flygym is distributed under the Apache License 2.0, which is compatible with
flyplotlib's MIT license as long as attribution is preserved (see ``NOTICE``).

Usage::

    uv run python scripts/vendor_neuromechfly_assets.py            # download
    uv run python scripts/vendor_neuromechfly_assets.py --src PATH # local checkout

Requires ``pyyaml`` (dev-only; the runtime package reads the generated JSON).
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml

REF = "dev-v2.0.3"
RAW = f"https://raw.githubusercontent.com/NeLy-EPFL/flygym/{REF}"
MODEL = "src/flygym/assets/model"

# Output directory (relative to the repo root, i.e. this script's parent's parent).
OUT_DIR = Path(__file__).resolve().parent.parent / "src/flyplotlib/data/neuromechfly"

# --- Anatomy (must stay in sync with flygym.anatomy and neuromechfly.py) -------
SIDES = ["l", "r"]
LEGS = [f"{s}{p}" for s in SIDES for p in "fmh"]
LEG_LINKS = ["coxa", "trochanterfemur", "tibia", *(f"tarsus{i}" for i in "12345")]
ANTENNA_LINKS = ["pedicel", "funiculus", "arista"]
PROBOSCIS_LINKS = ["rostrum", "haustellum"]
ABDOMEN_LINKS = ["abdomen12", *(f"abdomen{i}" for i in "3456")]


def _chain(*names: str) -> list[tuple[str, str]]:
    return [(names[i], names[i + 1]) for i in range(len(names) - 1)]


def connected_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = [("c_thorax", "c_head")]
    pairs += _chain("c_head", *(f"c_{lk}" for lk in PROBOSCIS_LINKS))
    pairs += _chain("c_thorax", *(f"c_{lk}" for lk in ABDOMEN_LINKS))
    pairs += [("c_head", f"{s}_eye") for s in SIDES]
    for s in SIDES:
        pairs += _chain("c_head", *(f"{s}_{lk}" for lk in ANTENNA_LINKS))
    pairs += [("c_thorax", f"{s}_wing") for s in SIDES]
    pairs += [("c_thorax", f"{s}_haltere") for s in SIDES]
    for leg in LEGS:
        pairs += _chain("c_thorax", *(f"{leg}_{lk}" for lk in LEG_LINKS))
    return pairs


def all_segment_names() -> list[str]:
    seen: dict[str, None] = {}
    for a, b in connected_pairs():
        seen.setdefault(a, None)
        seen.setdefault(b, None)
    return list(seen)


def mesh_basenames() -> list[str]:
    """Left/center segment names whose meshes must be vendored (right = mirror)."""
    return [s for s in all_segment_names() if s[0] != "r"]


# --- Fetch helpers -------------------------------------------------------------
def _read(src: Path | None, rel: str) -> bytes:
    if src is not None:
        return (src / rel).read_bytes()
    with urllib.request.urlopen(f"{RAW}/{rel}", timeout=60) as resp:  # noqa: S310
        return resp.read()


def _read_text(src: Path | None, rel: str) -> str:
    return _read(src, rel).decode("utf-8")


# --- Asset generation ----------------------------------------------------------
def vendor_meshes(src: Path | None, out: Path) -> None:
    mesh_out = out / "meshes"
    mesh_out.mkdir(parents=True, exist_ok=True)
    for name in mesh_basenames():
        data = None
        for sub in ("simplified_max2000faces", "fullsize"):
            try:
                data = _read(src, f"{MODEL}/meshes/{sub}/{name}.stl")
                break
            except (FileNotFoundError, urllib.error.HTTPError):
                continue
        if data is None:
            raise FileNotFoundError(f"No mesh found for {name}")
        (mesh_out / f"{name}.stl").write_bytes(data)
        print(f"  mesh {name}.stl ({len(data) // 1024} KB)")


def vendor_rigging(src: Path | None, out: Path) -> None:
    rigging = yaml.safe_load(_read_text(src, f"{MODEL}/rigging.yaml"))
    data = {
        name: {"pos": list(map(float, v["pos"])), "quat": list(map(float, v["quat"]))}
        for name, v in rigging.items()
    }
    (out / "rigging.json").write_text(json.dumps(data, indent=0))
    print(f"  rigging.json ({len(data)} segments)")


def vendor_neutral_pose(src: Path | None, out: Path) -> None:
    pose = yaml.safe_load(_read_text(src, f"{MODEL}/pose/neutral/pitch_roll_yaw.yaml"))
    assert pose["angle_unit"] == "degree", pose["angle_unit"]
    angles = {k: float(v) for k, v in pose["joint_angles"].items()}
    (out / "neutral_pose.json").write_text(json.dumps(angles, indent=0))
    print(f"  neutral_pose.json ({len(angles)} DOFs)")


def vendor_colors(src: Path | None, out: Path) -> None:
    """Derive a base color per segment from flygym's visuals.yaml.

    Each vis-set applies to a wildcard group of segments. We use the texture's
    ``rgb1`` base color when present, else the material ``rgba``.
    """
    visuals = yaml.safe_load(_read_text(src, f"{MODEL}/visuals.yaml"))
    segments = all_segment_names()
    colors: dict[str, list[float]] = {}
    for _set_name, params in visuals.items():
        if not isinstance(params, dict) or "apply_to" not in params:
            continue
        apply_to = params["apply_to"]
        patterns = [apply_to] if isinstance(apply_to, str) else apply_to
        texture = params.get("texture") or {}
        material = params.get("material") or {}
        if "rgb1" in texture:
            color = [float(c) for c in texture["rgb1"]]
            alpha = float(material.get("rgba", [0, 0, 0, 1])[3])
            if alpha != 1.0:
                color = [*color, alpha]
        else:
            color = [float(c) for c in material["rgba"]]
        for pattern in patterns:
            for seg in fnmatch.filter(segments, pattern):
                colors[seg] = color
    missing = [s for s in segments if s not in colors]
    if missing:
        raise RuntimeError(f"No color matched for: {missing}")
    (out / "colors.json").write_text(json.dumps(colors, indent=0))
    print(f"  colors.json ({len(colors)} segments)")


def write_notice(out: Path) -> None:
    (out / "NOTICE").write_text(
        "NeuroMechFly v2 assets vendored from flygym\n"
        f"  source: https://github.com/NeLy-EPFL/flygym (ref {REF})\n"
        f"  path:   {MODEL}\n\n"
        "flygym is distributed under the Apache License, Version 2.0. The\n"
        "bundled meshes (meshes/*.stl) and the derived metadata (rigging.json,\n"
        "neutral_pose.json, colors.json) originate from that project and remain\n"
        "subject to its license and attribution requirements.\n\n"
        "NeuroMechFly v2 reference:\n"
        "  Wang-Chen et al. (2024), NeuroMechFly v2, Nature Methods.\n"
        "  https://doi.org/10.1038/s41592-024-02497-y\n\n"
        "Regenerate with: python scripts/vendor_neuromechfly_assets.py\n"
    )
    print("  NOTICE")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        default=None,
        help="Path to a local flygym checkout (default: download from GitHub).",
    )
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Vendoring NeuroMechFly assets -> {args.out}")
    vendor_meshes(args.src, args.out)
    vendor_rigging(args.src, args.out)
    vendor_neutral_pose(args.src, args.out)
    vendor_colors(args.src, args.out)
    write_notice(args.out)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
