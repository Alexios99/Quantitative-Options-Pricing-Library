"""Execute every Python code block in README.md so the documented examples cannot silently break."""
import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[2] / "README.md"


def _python_blocks():
    text = README.read_text(encoding="utf-8")
    return re.findall(r"```python\n(.*?)```", text, re.S)


BLOCKS = _python_blocks()


def test_readme_has_examples():
    assert len(BLOCKS) >= 1, "README.md contains no ```python blocks"


@pytest.mark.parametrize("index", range(len(BLOCKS)))
def test_readme_example_runs(index, capsys):
    # Each block must run on its own, as a reader would copy-paste it.
    exec(compile(BLOCKS[index], f"README.md[block {index}]", "exec"), {})
    assert capsys.readouterr().out.strip(), f"README block {index} printed nothing"
