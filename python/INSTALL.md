# Installing from GitHub

No wheels are published yet, so installation builds the C++ core from source.
That needs a compiler and CMake, and takes a few minutes.

> **Why not `pip install ohdsi-cyclops`?** The package is not on PyPI yet, and
> there is no source distribution either: the Python project lives in `python/`
> but the C++ core it compiles lives in `src/`, and a Python sdist cannot contain
> files outside its own directory. Everything below therefore installs from a
> checkout — which `pip` can do for you directly from a URL.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python ≥ 3.9 | With `pip` ≥ 21.3 for the `subdirectory=` URL form |
| A C++17 compiler | GCC ≥ 9, Clang ≥ 10, or MSVC 2019+ |
| CMake ≥ 3.19 | Installed automatically by the build if absent |
| Git | `pip` shells out to it for the URL form |

Eigen is found automatically if installed; otherwise CMake downloads a pinned
copy (3.4.0) during the build, so the first build needs network access.

Installing Eigen up front makes the build faster and lets it work offline:

```bash
sudo apt-get install libeigen3-dev      # Debian / Ubuntu
```

```bash
brew install eigen                      # macOS
```

On Windows, let CMake fetch it.

---

## One-line install from GitHub

The Python project is in the `python/` subdirectory, so the URL needs
`subdirectory=python`:

```bash
pip install "git+https://github.com/ablack3/Cyclops.git@feature/python-api#subdirectory=python"
```

Verify it:

```bash
python -c "import cyclops; print(cyclops.__version__)"
```

### Pinning a revision

Anything after `@` is a git ref — a branch, tag, or commit SHA. Pin a SHA for
reproducible environments:

```bash
pip install "git+https://github.com/ablack3/Cyclops.git@de913205#subdirectory=python"
```

### In a requirements file

```
ohdsi-cyclops @ git+https://github.com/ablack3/Cyclops.git@feature/python-api#subdirectory=python
```

### With uv

```bash
uv pip install "git+https://github.com/ablack3/Cyclops.git@feature/python-api#subdirectory=python"
```

Or as a dependency in `pyproject.toml`:

```toml
[project]
dependencies = ["ohdsi-cyclops"]

[tool.uv.sources]
ohdsi-cyclops = { git = "https://github.com/ablack3/Cyclops.git", branch = "feature/python-api", subdirectory = "python" }
```

### From the OHDSI upstream repository

Once the bindings are merged upstream, the same command works against
`OHDSI/Cyclops`:

```bash
pip install "git+https://github.com/OHDSI/Cyclops.git@develop#subdirectory=python"
```

### Private clones over SSH

```bash
pip install "git+ssh://git@github.com/ablack3/Cyclops.git@feature/python-api#subdirectory=python"
```

---

## From a local checkout

Preferable if you plan to build more than once — `pip` reuses the CMake build
directory, so subsequent builds are incremental.

```bash
git clone https://github.com/ablack3/Cyclops.git
cd Cyclops
pip install ./python
```

### Editable install for development

```bash
pip install -e "./python[test]"
```

`scikit-build-core` rebuilds the extension automatically when the C++ sources
change, so editing `src/cyclops/` or `python/src/bindings.cpp` and re-importing
picks up the change.

Run the tests:

```bash
pytest python/tests -m "not parity"
```

The `parity` tests compare every fit against the R package and are skipped
automatically unless R is available with `Cyclops` and `jsonlite` installed:

```bash
pytest python/tests -m parity
```

---

## Build options

Pass CMake settings through `pip` with `--config-settings`.

Build against an out-of-tree copy of the C++ core:

```bash
pip install ./python --config-settings=cmake.define.CYCLOPS_ROOT=/path/to/Cyclops
```

Use a specific Eigen:

```bash
pip install ./python --config-settings=cmake.define.Eigen3_DIR=/usr/share/eigen3/cmake
```

Build with debug symbols:

```bash
pip install ./python --config-settings=cmake.build-type=RelWithDebInfo
```

See the whole compile:

```bash
pip install ./python -v --config-settings=cmake.verbose=true
```

---

## Building a wheel

To build once and install on several machines with the same platform and Python
version:

```bash
pip wheel ./python --no-deps -w dist/
pip install dist/ohdsi_cyclops-*.whl
```

For a full set of redistributable wheels, `cibuildwheel` is already configured
in `python/pyproject.toml`. Run it from the **repository root**, not from
`python/`, so the C++ core is inside the build context:

```bash
cibuildwheel --package-dir python
```

---

## Troubleshooting

**`Cannot find the Cyclops C++ facade under .../src`**

The build was pointed at a directory that is not a full checkout. Using the
`git+https://` URL without `#subdirectory=python`, or copying `python/` out of
the repository, both cause this. Install from a complete checkout, or set
`CYCLOPS_ROOT` explicitly.

**`Could NOT find Eigen3` followed by a download failure**

The build fell back to fetching Eigen and had no network. Install Eigen from
your package manager and rebuild.

**`error: Microsoft Visual C++ 14.0 or greater is required`** (Windows)

Install the "Desktop development with C++" workload from the Visual Studio
Build Tools.

**A stale CMake cache after moving the checkout**

`CYCLOPS_ROOT` is cached, so an absolute path from a previous location is
reused. Delete the build directory and reinstall:

```bash
rm -rf python/build
```

**The build succeeds but `import cyclops` finds the wrong package**

`cyclops` is also the import name of an unrelated clinical-ML package on PyPI.
Check which one you have:

```bash
python -c "import cyclops; print(cyclops.__file__)"
```

If it is not the Cyclops bindings, uninstall the other package or use a fresh
virtual environment.

---

## Uninstalling

```bash
pip uninstall ohdsi-cyclops
```
