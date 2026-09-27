"""scripts/run_backtest.py: outputs on success, one-line errors and non-zero exits on failure."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_backtest.py"


def load_script():
    spec = importlib.util.spec_from_file_location("run_backtest", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_script(*args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, env=env, check=False
    )


def test_end_to_end_success(make_config):
    config = make_config()

    done = run_script(str(config))

    assert done.returncode == 0, done.stderr
    assert "Comparison window: 2020-01" in done.stdout
    assert "SMA crossover (5/20)" in done.stdout
    reports = config.parent.parent / "reports"
    assert (reports / "test_run_comparison.csv").exists()
    assert (reports / "figures" / "test_run_equity.png").exists()


def test_end_to_end_config_error_is_one_line_and_exit_2(make_config):
    config = make_config(prefill=False, strategy={"fast": 30, "slow": 20})

    done = run_script(str(config))

    assert done.returncode == 2
    assert done.stderr.startswith("config error:")
    assert "shorter than slow" in done.stderr
    assert "Traceback" not in done.stderr


class TestMain:
    @pytest.fixture
    def main(self):
        return load_script().main

    def test_missing_config_file(self, main, tmp_path, capsys):
        assert main([str(tmp_path / "missing.yaml")]) == 2
        assert capsys.readouterr().err.startswith("config error: cannot read config")

    def test_data_error_exits_1(self, main, make_config, capsys):
        config = make_config()
        cache = config.parent.parent / "data" / "cache"
        (cache / "SPY_1d.meta.json").write_text("{corrupt")

        assert main([str(config)]) == 1
        err = capsys.readouterr().err
        assert err.startswith("data error: could not load SPY")
        assert "refresh=True" in err

    def test_backtest_error_exits_1(self, main, make_config, capsys):
        config = make_config(strategy={"fast": 50, "slow": 300})

        assert main([str(config)]) == 1
        assert capsys.readouterr().err.startswith("backtest error: sma_crossover(SPY, 50, 300)")

    def test_help(self, main, capsys):
        with pytest.raises(SystemExit) as info:
            main(["--help"])
        assert info.value.code == 0
        assert "--refresh" in capsys.readouterr().out
