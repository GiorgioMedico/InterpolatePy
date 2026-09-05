# Example programs

The repository contains executable Python and C++ examples. The Python scripts
were checked as programs, not only imported as modules, so their `__main__`
paths and plotting setup are exercised.

## Run Python examples

From a development checkout:

```bash
uv sync
uv run python examples/cubic_spline_ex.py
```

Plots open interactively under a normal Matplotlib backend. For a noninteractive
smoke run:

```bash
MPLBACKEND=Agg uv run python examples/double_s_ex.py
```

### Scalar splines

| Script | Demonstrates |
| --- | --- |
| `cubic_spline_ex.py` | Cubic interpolation and endpoint velocities |
| `c_s_smoothing_ex.py` | Several smoothing values and infinite endpoint weights |
| `c_s_smoot_search_ex.py` | Binary search for a prescribed maximum error |
| `c_s_with_acc1_ex.py` | Virtual-waypoint acceleration constraints and applications |
| `c_s_with_acc2_ex.py` | `SplineParameters` and quintic boundary segments |

### B-splines

| Script | Demonstrates |
| --- | --- |
| `b_spline_ex.py` | Basis values, derivatives, knots, and 2D plotting |
| `b_spline_cubic_ex.py` | Cubic 3D interpolation with chord-length parameters |
| `b_spline_interpolate_ex.py` | Degree 3/4/5 interpolation and derivative constraints |
| `b_spline_approx_ex.py` | Approximation with fewer control points and refinement |
| `b_spline_smooth_ex.py` | 3D curve smoothing and error measures |

### Motion profiles

| Script | Demonstrates |
| --- | --- |
| `double_s_ex.py` | Jerk-limited motion, reverse moves, phase timing, and factory use |
| `trapezoidal_ex.py` | Velocity- or duration-constrained segments and via points |
| `polynomials_ex.py` | Cubic, quintic, seventh-order, and multipoint polynomials |
| `lin_poly_parabolic_ex.py` | Linear paths joined by parabolic blends |
| `linear_ex.py` | Scalar and vector constant-velocity interpolation |

### Quaternion interpolation

| Script | Demonstrates |
| --- | --- |
| `quat_visualization_ex.py` | SLERP/SQUAD comparison and quaternion plots |
| `squad_c2_ex.py` | SLERP, SQUAD, and `SquadC2` comparison |
| `log_quat_new_ex.py` | LQI and mLQI interpolation and diagnostics |
| `lqi_spiral_frenet_ex.py` | LQI fitted to Frenet orientations on a helix |
| `mlqi_spiral_frenet_ex.py` | mLQI fitted to Frenet orientations on a helix |

The logarithmic examples intentionally use some implementation-level state for
diagnostic plots. Run them with the Python backend when inspecting those
details:

```bash
INTERPOLATEPY_NO_CPP=1 uv run python examples/log_quat_new_ex.py
```

### Paths and generic interfaces

| Script | Demonstrates |
| --- | --- |
| `simple_paths_ex.py` | Linear/circular geometry combined with polynomial time laws |
| `frenet_frame_ex.py` | Frenet and tool frames for circles and helices |
| `protocols_ex.py` | Generic sampling through runtime-checkable protocols |
| `main.py` | Minimal installed-version smoke check |

## Run every Python example headlessly

Use the repository checker to run every script in its own process:

```bash
uv run python -m scripts.check_python_examples --backend python
```

After building and copying the native extension into `src/interpolatepy/`, replace
`python` with `native` to verify the adapter-backed API. The checker fails if
native mode is requested but `HAS_CPP` is false.

This can take longer than the unit suite because several scripts construct many
figures. Matplotlib can warn that many figures are open during a headless run;
that warning is not a numerical failure.

## C++ examples

Enable the CMake example target, then build it:

```bash
cmake -S cpp -B build/cpp-examples \
  -DINTERPOLATECPP_BUILD_EXAMPLES=ON \
  -DINTERPOLATECPP_BUILD_TESTS=OFF
cmake --build build/cpp-examples --parallel
```

The generated executables correspond to the files under `cpp/examples/`:

- cubic, smoothing, and acceleration-constrained splines;
- base, interpolation, approximation, and smoothing B-splines;
- trapezoidal, polynomial, Double-S, and parabolic-blend motion;
- quaternion interpolation;
- paths and C++20 concepts.

Their exact output directory depends on the CMake generator and build
configuration. For example, a single-configuration build commonly produces
`build/cpp-examples/examples/cubic_spline_example`.

## Writing a new example

- make the file directly executable with an `if __name__ == "__main__"` block;
- import supported algorithms from `interpolatepy` unless the example is
  explicitly about implementation internals;
- keep data deterministic or seed random generators;
- separate numerical construction from visualization when practical;
- ensure the program completes with `MPLBACKEND=Agg`;
- add it to this page and to the relevant tutorial.
