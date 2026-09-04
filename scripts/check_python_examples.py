"""Run every Python example in a separate headless subprocess."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("auto", "python", "native"),
        default="auto",
        help="Select or verify the backend used by the examples.",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args()


def check_backend(environment: dict[str, str], backend: str, root: Path) -> bool:
    """Verify that native mode actually loaded the compiled extension."""
    if backend != "native":
        return True

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import interpolatepy; raise SystemExit(not interpolatepy.HAS_CPP)",
        ],
        cwd=root,
        env=environment,
        check=False,
    )
    if result.returncode == 0:
        return True
    print("Native backend requested, but interpolatepy.HAS_CPP is false.")
    return False


def main() -> int:
    """Execute every example and report failures with captured output."""
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    examples = sorted((root / "examples").glob("*.py"))
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"

    if args.backend == "python":
        environment["INTERPOLATEPY_NO_CPP"] = "1"
    elif args.backend == "native":
        environment.pop("INTERPOLATEPY_NO_CPP", None)

    failures = 0
    with tempfile.TemporaryDirectory(prefix="interpolatepy-matplotlib-") as config_dir:
        environment["MPLCONFIGDIR"] = config_dir
        if not check_backend(environment, args.backend, root):
            return 1

        for example in examples:
            try:
                result = subprocess.run(
                    [sys.executable, str(example)],
                    cwd=root,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                failures += 1
                print(f"FAIL {example.name}: timed out after {args.timeout:g}s")
                if error.stdout:
                    print(error.stdout)
                if error.stderr:
                    print(error.stderr)
                continue

            if result.returncode == 0:
                print(f"PASS {example.name}")
                continue

            failures += 1
            print(f"FAIL {example.name}: exit {result.returncode}")
            print(result.stdout)
            print(result.stderr)

    print(f"Checked {len(examples)} Python examples; failures: {failures}.")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
