# Design decisions

Why the Python bindings are built the way they are. Each section states the
decision, the alternatives that were live, and the reason for the choice.

---

## 1. Bind the existing C++ core; do not reimplement

Cyclops' value is ~4 500 lines of templated likelihood kernels in
`src/cyclops/engine/`, tuned over a decade for sparse observational-health
designs, plus the four-format column store that makes 10⁵-covariate problems
tractable. Reimplementing that in Python — even in Numba or JAX — would mean
maintaining two numerical code paths that must agree to machine precision
forever.

**Alternatives.**

- *Pure Python/SciPy reimplementation.* Loses the indicator-column
  representation and the risk-set recursions; would not match R; would be an
  order of magnitude slower.
- *Call R from Python via `rpy2`.* Zero C++ work, but requires an R installation
  at runtime, marshals through R's memory model, and inherits R's single-threaded
  interpreter lock on top of the GIL. Unacceptable for a library dependency.
- *Shell out to the `standalone/` CLI.* Requires serialising the design matrix to
  disk on every fit.

**Consequence:** the Python package is a binding. Every numerical result it
produces is by construction the same one R produces, which is what makes the
parity test-suite in `tests/test_parity_with_r.py` a meaningful invariant rather
than a coincidence.

---

## 2. pybind11 rather than Cython, nanobind, or a C ABI

**Chosen: pybind11.**

- The core is already C++14 with heavy template use; pybind11 is header-only C++
  and needs no separate translation step or generated sources in the tree.
- `pybind11` already ships in the R ecosystem's neighbourhood of familiarity —
  the OHDSI C++ code is Rcpp-shaped, and pybind11's `py::class_`/`def` idiom maps
  onto Rcpp's `[[Rcpp::export]]` closely enough that a reviewer who knows
  `RcppCyclopsInterface.cpp` can read `bindings.cpp` immediately.
- Automatic `std::vector` ↔ buffer conversion means the facade's plain-STL
  signatures need no per-function marshalling code. `bindings.cpp` is ~300 lines
  of declarations with no logic.
- `py::call_guard<py::gil_scoped_release>` on `fit()` is one line, so
  long-running fits do not block other Python threads.

**Alternatives.**

- *nanobind* is faster to compile and produces smaller binaries, and would be a
  reasonable choice. It requires C++17 and Python ≥ 3.8 (fine) but is younger,
  has a smaller pool of contributors familiar with it, and its stable-ABI story
  matters less here because the extension is small. pybind11's maturity wins for
  code that OHDSI maintainers may need to modify years from now. Switching later
  is a contained change: only `bindings.cpp` and `python/CMakeLists.txt`.
- *Cython* would mean writing `.pyx` declarations mirroring the C++ headers — a
  second interface definition to keep in sync, and awkward with templates.
- *A hand-written C ABI plus `ctypes`/`cffi`.* Maximum portability and the
  smallest wheel, but every object becomes an opaque handle with manual lifetime
  management, and exceptions must be converted to error codes by hand. The facade
  would be forced into C, losing `std::vector` and RAII.

---

## 3. The facade lives in `src/cyclops/api/`, not under `python/`

This is the decision most likely to be questioned, so the reasoning in full.

The brief asks for nearly all Python-specific code under `python/`. The facade is
**not Python-specific**: it has no `Python.h`, no pybind11, no `PyObject`. It is
the interface any host language needs. Three consumers exist or are plausible
today — the JNI layer in `src/cyclops/jni/`, the CLI in `standalone/`, and these
bindings — and all three currently (or would) re-derive the same call sequence:
build `ModelData`, call `AbstractModelSpecifics::factory`, construct
`CyclicCoordinateDescent`, install a prior, set `CCDArguments`, dispatch
cross-validation, read results off the optimizer.

Placing it under `python/` would label a language-neutral component
language-specific and guarantee that the next binding copies it.

**The usual objection — merge friction — does not apply here.** The facade is
three *new* files in a *new* directory. Git cannot produce a textual conflict on
a file upstream does not have. Concretely, `git merge upstream/main` touches
`src/cyclops/api/` only if upstream independently creates that path.

**And the R build is provably unaffected:**

- `src/Makevars.in` lists its object files explicitly — `cyclops/api/*.o` is not
  among them.
- `src/Makevars.win.in` globs `cyclops/*.cpp`, one directory level, so
  `cyclops/api/CyclopsApi.cpp` does not match.
- Verified empirically: `R CMD INSTALL` succeeds with no new warnings, and the
  `testthat` suite gives 248 passed / 0 failed both before and after.

**Alternative considered: `python/src/facade/`.** Zero footprint outside
`python/`, and defensible if the facade were expected to stay Python-only. But
`MIGRATION.md` §3 recommends upstreaming it precisely because OHDSI benefits from
having it — and a component you intend to upstream should already live where it
belongs.

**Consequence:** `python/` contains only `bindings.cpp`, the Python package, the
build files, tests, and docs. The C++ that is not Python-specific is not in
`python/`.

---

## 4. Wrapper classes, not exposed internals

`bsccs::CyclicCoordinateDescent` has ~60 public methods, some GPU-specific
(`turnOnSyncCV`, `ccdUpdateBetaVec`), some deprecated in comments
(`setHyperprior(double)` is marked "TODO depricate"), and `CCDArguments` carries
fields no longer read (`inFileName`, `useGPU`, `hyperprior`). Exposing that
surface to Python would:

- publish upstream's internal churn as our public API — a refactor in
  `CyclicCoordinateDescent` would become a breaking release here;
- leak `Eigen::MatrixXd` and `bsccs::shared_ptr` across the boundary, forcing
  pybind11 to know about Eigen and about the smart-pointer aliases;
- make invalid states reachable, since the correct call *order* (initialize →
  prior → control → weights → start values → fit) is nowhere encoded.

The facade instead exposes what a caller actually needs, with the ordering
enforced (`Model::create` requires finalized data; `ModelData` methods reject
loading after `finalize()`), and hides the rest behind pImpl. `FitResult` is a
plain struct carrying exactly the fields R's `cyclopsFit` object reports, so
"what does a fit return" has one answer in both languages.

The facade also *encodes* the parts of `fitCyclopsModel()` that are orchestration
rather than numerics, and that are easy to get silently wrong:

- the intercept warm start (`log(ȳ/(1-ȳ))` for logistic, `log(ȳ)` for Poisson,
  `ȳ` for least squares) — CCD is only locally convergent, so omitting this
  changes the iterate path;
- automatic exclusion of the intercept from regularization;
- the `"auto"` cross-validation selector resolution, which differs from
  `CcdInterface::getDefaultSelectorTypeOrOverride` for logistic and Poisson;
- the `POOR_BLR_STEP` → Lange-criterion retry.

Every one of those is covered by a parity test. They are the reason a Python fit
matches R to ~1e-10 rather than to 1e-4.

---

## 5. scikit-build-core rather than setuptools or Meson

**Chosen: scikit-build-core + CMake.**

- The project already needs CMake to compile the C++ core; scikit-build-core
  makes CMake the build system rather than bolting it onto `setuptools`.
- It is PEP 517-native and PEP 621-native: one `pyproject.toml`, no `setup.py`,
  no `MANIFEST.in`.
- It handles the wheel-tagging, `SKBUILD_PROJECT_VERSION` plumbing and
  editable-install machinery that hand-rolled `setuptools` extensions get wrong.
- It composes with `cibuildwheel` without extra configuration.

**Alternatives.**

- *`setuptools` + a custom `build_ext`.* The traditional route, and the one most
  Python developers can debug unaided — but it means reimplementing compiler
  detection, Eigen discovery and cross-compilation flags that CMake already
  handles, and reviewers would be reading two build systems.
- *meson-python.* Excellent (it is what SciPy uses), and Meson's dependency
  handling is arguably cleaner. Rejected because it would introduce a *second*
  build system to a repository that already ships CMake, and because
  `src/CMakeLists.txt` is intended to be upstreamed for the JNI and CLI builds
  too — those are CMake, so CMake it is.
- *`scikit-build` (classic).* Deprecated in favour of scikit-build-core.

---

## 6. The estimators are scikit-learn-shaped but not scikit-learn estimators

They implement `fit`, `predict`, `predict_proba`, `decision_function`,
`get_params`, `set_params`, and the trailing-underscore attribute convention, so
`sklearn.base.clone`, `GridSearchCV` and `Pipeline` work. They do **not** subclass
`sklearn.base.BaseEstimator`.

**Why not.** Subclassing would make scikit-learn a hard runtime dependency of a
package whose value proposition (conditional logistic regression, Cox at OHDSI
scale, cross-validated regularization) has no scikit-learn equivalent, and whose
users often work in environments where scikit-learn is not installed. It would
also invite `check_estimator` conformance, which the stratified and survival
models cannot satisfy: they need `strata=`/`time=` in `fit`, and scikit-learn's
estimator contract has no place for them.

**Where Cyclops semantics beat scikit-learn's conventions, Cyclops wins.**

- `prior_variance`, not `C` or `alpha`. Cyclops parameterizes by prior variance
  throughout, its documentation and the R API say variance, and the mapping is not
  a simple reciprocal (a Laplace prior with variance *v* has rate `sqrt(2/v)`).
  Renaming would create a silent mismatch with every OHDSI protocol.
- `prior_variance="cv"` triggers Cyclops' own cross-validated search rather than
  requiring `GridSearchCV`. The built-in auto-search is a Brent search over the
  variance that refits at the optimum, is threaded, and is what OHDSI studies
  actually use.
- `predict` follows the model family, matching `predict.cyclopsFit`:
  probabilities for logistic, expected counts for Poisson, the linear predictor
  for the conditional and survival models — because a conditional likelihood has
  no baseline rate and a Cox model no baseline hazard to attach one to.

**One deliberate improvement over R.** `predict.cyclopsFit` can only score the
rows it was fitted on unless given a full new Cyclops data object. Here
`predict(X)` evaluates any matrix with the same column layout, computing
`X @ coef_ + intercept_` in NumPy and applying the link in Python. A parity test
(`test_logistic_predictions_match`) checks that this reproduces the C++
`predictEstimate` on the training rows to 1e-12, so the Python-side link is
verified, not assumed.

---

## 7. Data enter through `loadX`/`loadY`, in CSC layout

`AbstractModelData` offers two loading paths. The `RcppModelData` constructors
take R S4 `Matrix` slots directly and are Rcpp-coupled. The
`loadY`/`loadX`/`loadMultipleX` family takes `std::vector` and is already
language-neutral — so the facade uses the latter, and no new core entry point was
needed.

`loadMultipleX` requires entries grouped by covariate and ascending in row within
each group, which is exactly compressed-sparse-column layout. A
`scipy.sparse.csc_matrix` therefore transfers with no restructuring: `indptr`,
`indices` and `data` go straight across. Sparse input is **never densified**,
which is the whole point of using Cyclops.

`loadX` (one column per call) is used rather than `loadMultipleX` (one call for
everything) because the latter needs a `covariateId` value per stored entry —
an `int64` array the size of `nnz` that would have to be materialised purely to
satisfy the signature. Per-column calls avoid that allocation and mirror what R's
column-at-a-time streaming API does.

Fully dense columns are loaded as `DENSE` rather than through CSC, so they do not
carry a redundant index vector and structural zeros are preserved.

---

## 8. Row ordering is handled in Python, and the permutation is exposed

The kernels require rows grouped by stratum, and for the survival models
additionally ordered by descending time within a stratum. Nothing in the C++ core
checks this; violating it produces silently wrong risk sets rather than an error.

`CyclopsData.from_arrays` therefore sorts with the same keys
`NewDataConversion.R` uses and stores the permutation in `row_order`. Weights and
starting values are permuted to match; `predict()` on the underlying model returns
values in stored order, and the estimators map back. Two parity tests
(`test_strata_out_of_order`, `test_cox_time_out_of_order`) shuffle the input and
require the same answer.

**Alternative: require callers to pre-sort.** Faster and simpler, and it is what
the C++ layer does. Rejected because a wrong answer with no diagnostic is the
worst possible failure mode for a statistical library.

---

## 9. Errors become exceptions; progress output is buffered

`ProgressLogger` and `ErrorHandler` are the core's two host hooks.
`BufferedLogger`/`ThrowingErrorHandler` (`src/cyclops/api/BufferedLogger.h`) are
the plain-C++ counterparts of `RcppProgressLogger`/`RcppErrorHandler`:

- errors throw `CyclopsError`, matching `Rcpp::stop`'s non-returning contract, and
  surface in Python as a `RuntimeError` subclass. When the core sets the handler
  to concurrent mode the message is deferred and raised on `flush()`, because
  throwing out of a cross-validation worker would terminate the process;
- progress lines are buffered and drained by `take_log()` rather than written to
  `stdout`, so a library caller decides where diagnostics go.

`yield()` — the interrupt hook — is a no-op with an opt-in callback that only
fires on the thread that constructed the logger, since CPython only permits signal
checks on the main thread. Ctrl-C during a long single fit is consequently not yet
responsive; see `TODO.md`.

---

## 10. Merge strategy with upstream

```
upstream  →  OHDSI/Cyclops
origin    →  our fork
```

- Python work lives on feature branches and merges to our `main`.
- `python/` is untouchable by upstream, so it never conflicts.
- The four changes outside `python/` are documented in `MIGRATION.md` with an
  upstream recommendation each. Two are bug fixes worth submitting immediately
  and independently of the Python work; two are additive new files.
- Routine sync is `git fetch upstream && git merge upstream/main`. Expected
  conflicts: none. The real risk is *semantic*: if upstream changes a `bsccs` API
  the facade calls, the facade needs the same edit any other caller would, and the
  parity suite is what detects it.
- **The parity suite is the regression gate.** After every upstream merge, run
  `pytest tests/ -m parity` against an R package built from the merged tree. It
  compares coefficients, log likelihood, log prior, convergence flag, iteration
  count, predictions and standard errors at ~1e-10 across ten model families, so a
  behavioural change in the core shows up as a test failure rather than as a
  quiet numerical drift.

---

## 11. Distribution name `ohdsi-cyclops`, import name `cyclops`

The brief's examples use `from cyclops import LogisticRegression`, and matching
the R package name keeps documentation transferable, so the import name is
`cyclops`.

That name is not unique on PyPI — VectorInstitute publishes a clinical-ML package
that also imports as `cyclops` — so the *distribution* is `ohdsi-cyclops`,
signalling provenance and avoiding a name grab. The two cannot be installed
together; if that turns out to matter in practice, `ohdsi_cyclops` is the fallback
import name, and it is a one-line rename in `pyproject.toml` plus the package
directory. Flagged in `TODO.md` as a decision to confirm with OHDSI before a
public release.
