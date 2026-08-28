#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Robin Guiavarch, Télécom Paris (LTCI)

"""Verify that every ``dinov2.*`` import resolves against the frozen 3DINO clone.

This repository is an extension layer: it ships no 3DINO code and resolves
``import dinov2`` from a separate, unmodified clone of AICONSlab/3DINO at a
pinned commit (see the README). This check walks every ``from dinov2.x import
y`` / ``import dinov2.x`` in the package and the two training mains, locates the
module file inside the clone, and confirms the imported names are defined there.
Pure AST — nothing is executed, nothing needs to be installed.

Usage::

    python tests/check_import_wiring.py [--clone /path/to/3DINO]

Default clone location: a ``3DINO`` directory next to this repository folder.
Exit code 0 = every import resolves; 1 = unresolved imports (listed).
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CLONE = REPO.parent / "3DINO"


def module_file(clone: Path, dotted: str) -> Path | None:
    """Map ``dinov2.a.b`` to its file in the clone (``b.py`` or ``b/__init__.py``)."""
    rel = Path(*dotted.split("."))
    candidate = clone / rel.with_suffix(".py")
    if candidate.is_file():
        return candidate
    candidate = clone / rel / "__init__.py"
    if candidate.is_file():
        return candidate
    return None


def top_level_names(path: Path) -> tuple[set, bool]:
    """Names a module binds at top level; the bool flags a ``from x import *``."""
    tree = ast.parse(path.read_text())
    names: set = set()
    star = False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    star = True
                else:
                    names.add(alias.asname or alias.name)
    return names, star


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check dinov2.* imports against the pinned 3DINO clone."
    )
    parser.add_argument(
        "--clone",
        type=Path,
        default=DEFAULT_CLONE,
        help="Path to the upstream 3DINO clone (default: ../3DINO next to this repo).",
    )
    args = parser.parse_args()
    clone = args.clone.resolve()

    if not (clone / "dinov2").is_dir():
        print(f"[import-wiring] ERROR: no dinov2 package under {clone}.")
        print("[import-wiring] Clone AICONSlab/3DINO there first (see the README).")
        return 1

    sources = [REPO / "train.py", REPO / "train_anisotropic.py"]
    sources += sorted((REPO / "sulcus_aware_3DINO").rglob("*.py"))

    problems: list = []
    modules: set = set()
    symbols = 0
    for source in sources:
        tree = ast.parse(source.read_text())
        rel = source.relative_to(REPO)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.split(".")[0] == "dinov2"
            ):
                modules.add(node.module)
                target = module_file(clone, node.module)
                if target is None:
                    problems.append(f"{rel}: module '{node.module}' not found in clone")
                    continue
                names, star = top_level_names(target)
                if star:
                    continue  # module re-exports; accept
                for alias in node.names:
                    symbols += 1
                    if alias.name not in names:
                        problems.append(
                            f"{rel}: '{alias.name}' not defined in '{node.module}'"
                        )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] == "dinov2":
                        modules.add(alias.name)
                        if module_file(clone, alias.name) is None:
                            problems.append(
                                f"{rel}: module '{alias.name}' not found in clone"
                            )

    print(f"[import-wiring] clone           : {clone}")
    print(f"[import-wiring] sources scanned : {len(sources)}")
    print(f"[import-wiring] dinov2 modules  : {len(modules)}")
    print(f"[import-wiring] symbols checked : {symbols}")
    if problems:
        print(f"[import-wiring] FAIL — {len(problems)} unresolved import(s):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("[import-wiring] PASS — every dinov2.* import resolves in the pinned clone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
