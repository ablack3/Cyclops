# Roadmap

Status as of the initial Python-bindings branch. Priorities: **P0** blocks a
usable release, **P1** blocks a public release, **P2** is valuable, **P3** is
speculative.

---

## Done

### Phase 1 — Understand Cyclops
- [x] Build system, CMake layout, `Makevars` object lists, `configure` chain
- [x] Rcpp interface: ~60 exports across `RcppCyclopsInterface.cpp` / `RcppModelData.cpp`
- [x] Major classes: `AbstractModelData`/`ModelData<T>`, `CompressedDataMatrix`,
      `CyclicCoordinateDescent`, `AbstractModelSpecifics`, `CcdInterface`,
      the prior hierarchy, the CV drivers
- [x] Dependency graph — established the core needs only C++14, Eigen (headers)
      and TinyThread-on-Windows; no Boost, no live RcppParallel
- [x] Traced data entry, the optimization pipeline, and result read-out
- [x] `docs/ARCHITECTURE.md`

### Phase 2 — Smallest possible binding
- [x] Built the core outside R
- [x] One pybind11 extension, one model, one 6-row logistic fit
- [x] Verified against R: identical log likelihood, coefficients, iteration
      count, convergence flag, predictions

### Phase 3 — Stable C++ facade
- [x] `src/cyclops/api/CyclopsApi.{h,cpp}` — `ModelData`, `Model`, `FitOptions`,
      `PriorOptions`, `FitResult`; STL types only, implementation behind pImpl
- [x] `ApiCcdInterface`, the fourth `CcdInterface` implementation
- [x] `BufferedLogger` / `ThrowingErrorHandler` — host-agnostic logging and errors
- [x] Removed the last live Rcpp reference from the core (`MIGRATION.md` §1)

### Phase 4 — Packaging
- [x] `pyproject.toml` (scikit-build-core, PEP 621), `python/CMakeLists.txt`
- [x] `src/CMakeLists.txt` — reusable `cyclops::core` target, Eigen via
      `find_package` with a pinned `FetchContent` fallback
- [x] Editable install and wheel build on macOS/arm64, Python 3.14
- [x] `cibuildwheel` configuration for CPython 3.9–3.13 on Linux/macOS/Windows
- [x] GitHub Actions: test job + wheel job

### Phase 5 — Python API
- [x] `CyclopsData` (incremental + `from_arrays`), `CyclopsModel`, `Prior`,
      `Control`, `FitResult`
- [x] Estimators: `LogisticRegression`, `PoissonRegression`, `LinearRegression`,
      `ConditionalLogisticRegression` (+ exact/Efron variants),
      `ConditionalPoissonRegression`, `SelfControlledCaseSeries`,
      `CoxRegression`, `TimeVaryingCoxRegression`, `FineGrayRegression`
- [x] NumPy and SciPy-sparse input with no densification anywhere
- [x] Row-ordering handled internally, permutation exposed as `row_order`
- [x] Inference: standard errors, Fisher information, Hessian diagonal,
      likelihood-profile intervals and curves
- [x] Type stubs (`_cyclops.pyi`) and a `ruff` configuration enforced in CI

### Phase 6 — Testing
- [x] 31 R-parity tests across 10 model families, tolerances at ~1e-9/1e-10
- [x] 131 unit tests covering data layout, estimator protocol, prediction links,
      regularization, weights, offset-column accounting, validation and error paths
- [x] Verified the core changes leave R unaffected: `testthat` gives
      248 passed / 0 failed both before and after

### Phase 7 — Upstream-friendly changes
- [x] Fixed an out-of-bounds read in `binaryReductionByStratum` reachable from R
      (`MIGRATION.md` §2)
- [x] Six further upstream defects documented but deliberately not changed
      (`MIGRATION.md`, "Observations reported but not changed")

---

## Remaining

### P0 — before this is usable by anyone else

- [ ] **Build and test on Linux and Windows.** Only macOS/arm64 + CPython 3.14
      has actually been exercised. The CI workflow exists but has never run.
      Windows is the real risk: `Thread.h` switches to TinyThread there, MSVC
      needs `/bigobj` for the templated kernels (configured, unverified), and
      `src/Makevars.win.in`'s wildcard behaviour differs from the Unix list.
- [ ] **Run the parity suite in CI** against an R Cyclops built from the same
      tree. Today it is manual, and it is the only thing standing between an
      upstream merge and silent numerical drift.
- [ ] **Test on the minimum supported Python (3.9).** `pyproject.toml` claims
      `>=3.9`; the code uses `X | None` annotations under
      `from __future__ import annotations`, which is fine, but this is unverified.
- [ ] **No publishable sdist.** An sdist rooted at `python/` cannot contain the
      C++ core, because sdist paths may not escape the project directory — so
      `pip install ohdsi-cyclops --no-binary` would fail. Wheels are fine
      (cibuildwheel copies the whole working directory, so `src/` comes along),
      and `pip install ./Cyclops/python` from a checkout works. Fixing it means
      either a pre-build step that copies `src/` into `python/`, or moving
      `pyproject.toml` to the repository root — which would put Python metadata in
      the R package root. Decide before publishing; until then, publish wheels
      only and document the source-install route.

### P1 — before a public release

- [ ] **Confirm the package name with OHDSI.** `cyclops` collides on PyPI with an
      unrelated clinical-ML package (`DESIGN_DECISIONS.md` §11). Decide between
      `cyclops` and `ohdsi_cyclops` as the import name before anything is
      published; renaming afterwards is a breaking change.
- [ ] **Add a `LICENSE` file.** The repository has none — R packages declare the
      licence in `DESCRIPTION` — so `pyproject.toml` carries the SPDX expression
      with no `license-files`. Wheels should ship the Apache-2.0 text.
- [ ] **Ctrl-C during a long fit.** `BufferedLogger::yield()` has the hook but
      nothing installs it, so a multi-minute cross-validated fit cannot be
      interrupted. Needs `PyErr_CheckSignals` under a `gil_scoped_acquire`, main
      thread only, plus an audit that throwing out of `findMode` does not leak.
- [ ] **Bootstrap.** `CcdInterface::runBoostrap` and R's `runBootstrap` are not
      exposed. Straightforward facade addition.
- [ ] **Hierarchical and fused priors.** `makePrior` in the facade handles the
      exchangeable, per-covariate and exclusion cases;
      `HierarchicalJointPrior` (normal-normal) and `FusedLaplacePrior`
      (neighbourhood fusion) are not wired up. Needed for the OHDSI
      "sparse-with-hierarchy" designs.
- [ ] **Time-varying Cox time-effect map.** `loadStratTimeEffects` is unexposed,
      so `TimeVaryingCoxRegression` currently accepts only fixed covariates.
- [ ] **Fine-Gray censoring weights.** `FineGrayRegression` requires
      `censor_weights=` from the caller; port `getFineGrayWeights()` so it can be
      derived from `(time, event)` as R does.
- [ ] **Schoenfeld residuals / proportionality test.**
      `getSchoenfeldResiduals` and `cyclopsTestProportionality` are unexposed —
      the standard Cox diagnostics.
- [ ] **Documented examples with real data.** `Cyclops::oxford` and the other
      bundled datasets are in `data/`; the tests use synthetic data only.

### P2 — valuable

- [ ] **`survfit`-style baseline hazard** (`R/Survfit.R` has no counterpart here).
- [ ] **`normalize=` on the estimators.** `CyclopsData.normalize()` exists but the
      estimators never call it, and coefficient rescaling (R's
      `coef(fit, rescale = TRUE)`) is not implemented.
- [ ] **Parameterized priors.** `PriorFunction` supports a caller-supplied
      variance function; `RcppPriorFunction` shows the pattern. Note that three
      of the five prior types throw "not yet implemented" upstream
      (`MIGRATION.md` §1).
- [ ] **FP32 precision end to end.** `Precision::Fp32` is plumbed through the
      facade and `ModelData<float>` is instantiated, but nothing tests it.
- [ ] **Reduce the copies on load.** Input is copied twice: NumPy → `std::vector`
      by pybind11, then `std::vector` → `CompressedDataColumn` by the loader.
      Zero-copy would need a core loader taking iterators or spans.
- [ ] **`__sklearn_tags__` / `check_estimator` conformance** for the three
      unstratified estimators, which *can* satisfy the scikit-learn contract.
- [ ] **Multitype models.** `numTypes` in `ModelData` and R's `Multitype.R` are
      unexposed.
- [ ] **`float32` / `int32` input without an upcast.** Everything is forced to
      `float64`/`int64` at the boundary; wide designs pay for that.

### P3 — speculative

- [ ] **GPU builds.** `list_gpu_devices()` returns empty by construction. The
      facade already routes `compute_device` through to
      `AbstractModelSpecifics::factory`, so a CUDA/OpenCL wheel is a build
      problem, not an API problem.
- [ ] **Arrow / Polars input.** OHDSI increasingly stores covariates as Parquet;
      an Arrow-native loader would avoid a SciPy round-trip.
- [ ] **Out-of-core loading** from Parquet or DuckDB, mirroring what Andromeda
      does for R.
- [ ] **Stable ABI wheels** (`pybind11_add_module(... STABLE_ABI)`) to collapse
      the build matrix to one wheel per platform.
- [ ] **Julia / Rust bindings** over the same facade — the argument for putting it
      in `src/cyclops/api/` in the first place.

---

## Technical debt

1. **`_cyclops.pyi` is maintained by hand.** pybind11 emits no stubs, so the
   stub file can silently drift from `bindings.cpp`. Either generate it
   (`pybind11-stubgen`) in CI or add a test that walks the module and compares
   against the stub.
2. **The R bridge in `tests/r_bridge.py` embeds an R script as a string literal.**
   Fine at its current size, awkward past it. Worth extracting to
   `tests/r_bridge.R` once it grows.
3. **The parity suite writes CSV and shells out to `Rscript` per test.** ~0.7 s
   per test. A single R session driven over a pipe would be faster, at the cost of
   a much more complex harness. Acceptable while the suite is 31 tests.
4. **`Control` duplicates `FitOptions` field-for-field**, and `Prior` duplicates
   `PriorOptions`. The dataclasses buy validation, defaults and docstrings, but
   a new C++ option must be added in three places (facade struct, bindings,
   dataclass). A generated binding would remove the duplication; not worth it yet.
5. **The facade's `make_prior` is a partial reimplementation of
   `RcppCcdInterface::makePrior`.** Same structure, minus the hierarchy and
   neighbourhood branches. When those are added, the two will have drifted enough
   that a shared `bsccs::priors::makeJointPrior(...)` in the core — usable by both
   bindings — becomes the right refactor and a good upstream contribution.
6. **`test_data.py::test_row_index_out_of_range_raises` reaches through
   `_handle`** to hit a validation path the Python API cannot express. It tests
   the facade's guard, but through a private attribute.
7. **No benchmarks.** There is no measurement of the binding's overhead relative
   to R, nor of the double copy on load. Needed before any optimization work
   (P2, "reduce the copies") can be justified.

---

## Upstream pull requests to open

In recommended order (see `MIGRATION.md` for the full write-up of each):

1. **Remove the Rcpp dependency from `NewCovariatePrior.h`.** Three lines, makes
   a core header self-contained, no behaviour change.
2. **Fix the out-of-bounds read in `binaryReductionByStratum`.** Memory-safety
   fix with an R-only reproducer; also gives `algorithm = "mm"` the correct
   bound on unstratified data.
3. **Report the six observed defects** as issues: the `pr` segfault without
   `time`, the undocumented `stratumId`-on-covariates requirement, the
   `OBJECTS.threads` typo, the `removeIntercept` disagreement, the ignored
   `takeLog` parameter, and the MM monotonicity assertion.
4. **Propose `src/CMakeLists.txt`** as a standalone library target.
5. **Propose `src/cyclops/api/`** as a reusable non-R interface, framed around
   the JNI and CLI duplication it removes.
