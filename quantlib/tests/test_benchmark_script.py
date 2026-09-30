import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks" / "benchmark.py"


@pytest.mark.slow
def test_benchmark_quick_run_writes_all_sections(tmp_path):
    spec = importlib.util.spec_from_file_location("benchmark", SCRIPT)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)

    output = benchmark.run(quick=True, output=tmp_path / "RESULTS.md")
    text = output.read_text(encoding="utf-8")
    for title, _ in benchmark.SECTIONS:
        assert f"## {title}" in text
    assert "nan" not in text.lower()
