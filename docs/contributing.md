# Contributing

Contributions to Python code, C++ code, tests, examples, and documentation are
welcome. For a large API or algorithm change, open an issue first so numerical
requirements and backend parity can be agreed before implementation.

## Development setup

Install [uv](https://docs.astral.sh/uv/), then clone and synchronize the locked
environment:

```bash
git clone https://github.com/GiorgioMedico/InterpolatePy.git
cd InterpolatePy
uv sync
uv run pre-commit install
```

The default dependency groups are `dev`, `test`, and `examples`. Add the docs
group for documentation work:

```bash
uv sync --group docs
```

## Before submitting a change

Run checks in proportion to the affected area. The complete Python-side set is:

```bash
uv run pytest
uv run ruff check .
uv run mypy interpolatepy
uv run pre-commit run --all-files
```

For documentation changes:

```bash
uv run mkdocs build --clean --strict
```

For changes to examples, run the affected scripts. A headless smoke check uses:

```bash
MPLBACKEND=Agg INTERPOLATEPY_NO_CPP=1 uv run python examples/your_example.py
```

## Repository map

```text
interpolatepy/            Python package
  _api.py                 backend routing
  _backend.py             native extension detection
  _adapters/              C++-backed Python API adapters
tests/                    pytest suite
examples/                 executable Python demonstrations
cpp/
  include/interpolatecpp/ public C++ headers
  src/                    C++ implementation
  bindings/               pybind11 bindings
  tests/                  Catch2 tests
  examples/               C++ programs
docs/                     MkDocs source
```

## Python changes

Use modern type annotations supported by Python 3.11. Public functions and
classes should have NumPy-style docstrings describing shapes, units, valid
ranges, return semantics, and errors. Keep implementation details private when
they are not part of the compatibility contract.

When adding or changing an algorithm:

1. validate lengths, finite values, and monotonicity at the boundary;
2. test endpoints, interior waypoints, vectorized input where supported, reverse
   motion, degenerate cases, and invalid input;
3. test derivative continuity or bounds numerically;
4. update a runnable example and the relevant tutorial;
5. update both backend routes if a native implementation exists.

Use the package-root API in end-user examples. Direct implementation imports are
appropriate only for backend-specific tests or diagnostics.

## Backend parity

A native algorithm is not complete when only the C++ class exists. Update:

- a public header under `cpp/include/interpolatecpp/`;
- a source file and `INTERPOLATECPP_SOURCES` in `cpp/CMakeLists.txt`;
- Catch2 tests and, when useful, a C++ example;
- a pybind11 binding under `cpp/bindings/`;
- the matching adapter in `interpolatepy/_adapters/`;
- both branches of `interpolatepy/_api.py`;
- package exports and public-API tests.

Adapters should normalize container types, return types, vectorized behavior,
and public parameter names. Tests should call the package-root name so the same
assertions exercise whichever backend is active.

## C++ build and tests

From the repository root:

```bash
cmake -S cpp -B build/cpp-tests \
  -DINTERPOLATECPP_BUILD_TESTS=ON \
  -DINTERPOLATECPP_BUILD_BINDINGS=OFF
cmake --build build/cpp-tests --parallel
ctest --test-dir build/cpp-tests --output-on-failure
```

To build the extension for parity tests:

```bash
cmake -S cpp -B build/cpp-bindings \
  -DINTERPOLATECPP_BUILD_TESTS=OFF \
  -DINTERPOLATECPP_BUILD_BINDINGS=ON
cmake --build build/cpp-bindings --parallel
cp build/cpp-bindings/bindings/interpolatecpp_py*.so interpolatepy/
python -c "import interpolatepy as ip; assert ip.HAS_CPP"
uv run pytest
```

The copy command shown is for Linux/macOS. See [Installation](installation.md)
for Windows and ABI details.

## Tests

Prefer focused tests with explicit numerical tolerances. A useful algorithm test
usually checks:

- exact or near-exact endpoint values;
- interpolation at all required waypoints;
- continuity from both sides of an interior knot;
- derivative limits over a dense sample, when limits are promised;
- scalar and array return shape, where vectorization is supported;
- the active public API rather than private coefficients.

Run a focused file while iterating:

```bash
uv run pytest tests/test_cubic_spline.py -q
```

Then run the full suite before opening a pull request.

## Documentation

The site uses MkDocs Material and mkdocstrings. API pages render NumPy-style
docstrings from the installed source tree. Every code block presented as a
complete example should be executable as written; fragments should be clearly
identified.

Keep these sources synchronized:

- `README.md` for installation and the shortest introduction;
- `ALGORITHMS.md` for the repository-level selection guide;
- `docs/` for the published site;
- public Python docstrings and C++ header comments;
- `docs/changelog.md` for released behavior.

Do not publish unmeasured performance numbers, unsupported platform claims, or
continuity/bound guarantees that tests do not verify.

Preview locally with:

```bash
uv run mkdocs serve
```

The documentation workflow uses `mkdocs build --clean --strict`, so broken
links and mkdocstrings warnings fail CI.

## Example programs

New Python examples should:

- have deterministic inputs;
- place execution under `if __name__ == "__main__"`;
- complete under `MPLBACKEND=Agg`;
- use public imports unless backend internals are the topic;
- avoid requiring an interactive prompt;
- document units and the distinction between time and curve parameters.

Add the script to [Example programs](examples.md). C++ examples must be added to
`cpp/examples/CMakeLists.txt`.

## Pull requests

Keep a pull request focused and describe:

- the problem and numerical behavior being changed;
- compatibility or backend implications;
- tests performed;
- documentation and example updates;
- any intentionally unsupported edge case.

Do not commit generated `site/`, local native extension binaries, build trees,
coverage output, or virtual environments.

InterpolatePy follows semantic versioning. Breaking public API changes require a
major release; backward-compatible features use a minor release; compatible
fixes use a patch release.
