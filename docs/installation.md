# Installation

## Requirements

The Python package declares these runtime requirements in `pyproject.toml`:

| Dependency | Minimum version |
| --- | --- |
| Python | 3.11 |
| NumPy | 1.26 |
| SciPy | 1.11 |

Matplotlib 3.6 or newer is optional and used only by plotting helpers. Published
platform wheels include the native extension and do not need a C++ compiler at
installation time.

## Install from PyPI

Create a virtual environment and install the package with the interpreter that
will run your code:

=== "Linux and macOS"

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install InterpolatePy
    ```

=== "Windows PowerShell"

    ```powershell
    py -3.11 -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install InterpolatePy
    ```

Verify the interpreter, package version, and active backend:

```bash
python -c "import interpolatepy as ip; print(ip.__version__, ip.HAS_CPP)"
```

Install plotting support with `python -m pip install "InterpolatePy[plot]"`.
`HAS_CPP=False` means a compatible native wheel was unavailable or the backend
was explicitly disabled; the Python implementation remains functional.

## Development checkout

InterpolatePy uses [uv](https://docs.astral.sh/uv/) and locks its development
environment in `uv.lock`:

```bash
git clone https://github.com/GiorgioMedico/InterpolatePy.git
cd InterpolatePy
uv sync
```

The default groups are `dev`, `test`, and `examples`. Select additional or
isolated groups as needed:

```bash
# Add the documentation toolchain.
uv sync --group docs

# Reproduce the documentation CI environment only.
uv sync --locked --no-default-groups --group docs

# Install every dependency group.
uv sync --all-groups
```

Useful verification commands are:

```bash
uv run pytest
uv run ruff check .
uv run mypy src/interpolatepy
uv run pyright src/interpolatepy
uv run mkdocs build --clean --strict
```

## Optional C++ backend

The source tree contains two related native targets:

- `interpolatecpp`, a standalone C++20 library;
- `interpolatecpp_py`, a pybind11 extension loaded by the Python package.

Both are built with CMake. CMake fetches Eigen 3.4 and, depending on the build
options, Catch2 and pybind11, so the first configure requires network access.

### Build and activate the Python extension

You need CMake 3.21 or newer, a C++20 compiler, Python development headers, and
a source checkout. From the repository root:

```bash
cmake -S cpp -B build/cpp \
  -DINTERPOLATECPP_BUILD_BINDINGS=ON \
  -DINTERPOLATECPP_BUILD_TESTS=OFF
cmake --build build/cpp --parallel
```

On Linux or macOS, copy the produced extension beside `_backend.py`:

```bash
cp build/cpp/bindings/interpolatecpp_py*.so src/interpolatepy/
python -c "import interpolatepy as ip; print(ip.HAS_CPP)"
```

On Windows, copy the generated `interpolatecpp_py*.pyd` from the selected CMake
configuration into `src/interpolatepy/`. Multi-configuration generators usually
place it under a `Debug` or `Release` subdirectory.

The extension filename must retain its Python ABI suffix. `_backend.py` imports
it as `interpolatepy.interpolatecpp_py`; copying only the standalone
`interpolatecpp` library does not activate Python acceleration.

### Build and test the C++ library

```bash
cmake -S cpp -B build/cpp-tests \
  -DINTERPOLATECPP_BUILD_TESTS=ON \
  -DINTERPOLATECPP_BUILD_BINDINGS=OFF \
  -DINTERPOLATECPP_BUILD_EXAMPLES=ON
cmake --build build/cpp-tests --parallel
ctest --test-dir build/cpp-tests --output-on-failure
```

The CMake options and their defaults are:

| Option | Default | Effect |
| --- | --- | --- |
| `INTERPOLATECPP_BUILD_TESTS` | `ON` | Fetch Catch2 and build C++ tests |
| `INTERPOLATECPP_BUILD_BINDINGS` | `OFF` | Fetch pybind11 and build the Python extension |
| `INTERPOLATECPP_BUILD_EXAMPLES` | `OFF` | Build programs under `cpp/examples/` |

### Force the Python backend

Set the variable before the first import:

=== "Linux and macOS"

    ```bash
    INTERPOLATEPY_NO_CPP=1 python your_program.py
    ```

=== "Windows PowerShell"

    ```powershell
    $env:INTERPOLATEPY_NO_CPP = "1"
    python your_program.py
    ```

Backend selection is process-wide. Changing the variable after importing
`interpolatepy` does not reroute already imported classes.

## Installation troubleshooting

If an import fails, verify that `python` and `pip` refer to the same environment:

```bash
python -m pip show InterpolatePy
python -c "import sys; print(sys.executable)"
```

If the native configure fails:

- confirm `cmake --version` reports 3.21 or newer;
- confirm the selected compiler supports C++20;
- remove only the affected `build/cpp*` directory and configure again after a
  dependency-fetch interruption;
- build with `INTERPOLATECPP_BUILD_TESTS=OFF` when only the Python extension is
  needed.

See [Troubleshooting](troubleshooting.md) for API and numerical issues.
