"""Execute standalone Python examples embedded in the project documentation."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
import re
import traceback

from matplotlib import pyplot as plt


PYTHON_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def documentation_files(root: Path) -> list[Path]:
    """Return Markdown sources that can contain public examples."""
    return [
        root / "README.md",
        root / "ALGORITHMS.md",
        *sorted((root / "docs").rglob("*.md")),
    ]


def is_standalone_example(code: str) -> bool:
    """Identify snippets that declare their own InterpolatePy imports."""
    return "import interpolatepy" in code or "from interpolatepy" in code


def close_figures() -> None:
    """Release figures created by a snippet."""
    plt.close("all")


def main() -> int:
    """Run each standalone snippet in an isolated global namespace."""
    root = Path(__file__).resolve().parents[1]
    checked = 0
    failures: list[str] = []

    for path in documentation_files(root):
        for block_number, match in enumerate(PYTHON_FENCE.finditer(path.read_text()), 1):
            code = match.group(1)
            if not is_standalone_example(code):
                continue

            checked += 1
            output = io.StringIO()
            try:
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                    exec(
                        compile(code, f"{path}:{block_number}", "exec"),
                        {"__name__": "__documentation_example__"},
                    )
            except Exception:  # noqa: BLE001
                relative_path = path.relative_to(root)
                failures.append(
                    f"{relative_path}: Python block {block_number}\n"
                    f"{output.getvalue()}{traceback.format_exc()}"
                )
            finally:
                close_figures()

    if failures:
        print("\n".join(failures))
        print(f"Failed {len(failures)} of {checked} standalone Python snippets.")
        return 1

    print(f"Passed {checked} standalone Python documentation snippets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
