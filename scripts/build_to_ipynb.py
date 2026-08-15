"""Regenerate kaggle/src/arc_build/arc-build.ipynb from build.py.

The kernel notebook is a single code cell whose source IS build.py. Authoring in
.py keeps it diffable/editable; this script syncs the .ipynb the CLI uploads, so
the two never drift. Run after every edit to build.py:

    python scripts/build_to_ipynb.py
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
PY   = os.path.join(HERE, "..", "kaggle", "src", "arc_build", "build.py")
NB   = os.path.join(HERE, "..", "kaggle", "src", "arc_build", "arc-build.ipynb")

src = open(PY, encoding="utf-8").read().splitlines(keepends=True)

notebook = {
    "cells": [{
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }],
    "metadata": {
        "kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(NB, "w", encoding="utf-8") as fh:
    json.dump(notebook, fh, indent=1)
print(f"regenerated {os.path.relpath(NB)} from build.py ({len(src)} lines)")
