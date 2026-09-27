"""The results notebook runs top to bottom (skipped unless the notebook extra is installed).

It runs against a small synthetic experiment through ``BACKTEST_CONFIG``, so no
network is needed and the committed notebook is not modified.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")
pytest.importorskip("ipykernel")

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "01_results.ipynb"


def test_committed_notebook_has_no_error_outputs():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert not any(o.get("output_type") == "error" for o in cell.get("outputs", []))


def test_notebook_executes_end_to_end(make_config, monkeypatch):
    config = make_config()
    monkeypatch.setenv("BACKTEST_CONFIG", str(config))
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join([str(REPO / "src"), os.environ.get("PYTHONPATH", "")])
    )
    notebook = nbformat.read(NOTEBOOK, as_version=4)

    client = nbclient.NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK.parent)}},
    )
    client.execute()

    outputs = [o for cell in notebook.cells if cell.cell_type == "code" for o in cell.outputs]
    images = [o for o in outputs if "image/png" in o.get("data", {})]
    assert len(images) == 3
    assert not any(o.get("output_type") == "error" for o in outputs)
    assert (config.parent.parent / "reports" / "figures" / "test_run_equity.png").exists()
