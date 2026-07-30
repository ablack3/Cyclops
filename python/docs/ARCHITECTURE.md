# Cyclops Architecture (and the proposed Python interface)

This document is the Phase 1 deliverable: a written summary of the *actual* Cyclops
architecture, derived from reading the source at `3.7.1` (commit `89dd48b1`), followed by
the design of the Python interface built on top of it.

---

## 1. Repository layout

```
Cyclops/
├── DESCRIPTION, NAMESPACE          # R package metadata
├── configure(.win)                 # generates src/Makevars from src/Makevars.in
├── tools/configure.R
├── R/                              # R API (~180 KB of R)
├── src/
│   ├── Makevars.in                 # explicit object list (Unix)
│   ├── Makevars.win.in             # wildcard-based object list (Windows)
│   ├── Rcpp*.{h,cpp}               # ← the ONLY Rcpp-coupled translation units
│   ├── cyclops/                    # ← the portable C++ core
│   │   ├── CcdInterface.{h,cpp}    # abstract driver / orchestration
│   │   ├── CyclicCoordinateDescent.{h,cpp}   # the optimizer (60 KB)
│   │   ├── ModelData.{h,cpp}       # design matrix + outcome container
│   │   ├── CompressedDataMatrix.{h,cpp}      # column store (4 formats)
│   │   ├── Iterators.h             # format-specialised column iterators
│   │   ├── Types.h                 # enums, typedefs, smart-pointer aliases
│   │   ├── Thread.h, Timer, Timing
│   │   ├── drivers/                # cross-validation + bootstrap drivers
│   │   ├── priors/                 # prior hierarchy
│   │   ├── io/                      # readers, writers, ProgressLogger ABC
│   │   ├── engine/                 # ModelSpecifics: per-likelihood kernels
│   │   └── jni/                    # Java bindings (parallel to Rcpp ones)
│   ├── utils/                      # HParSearch (Brent search), RZeroIn
│   └── tinythread/                 # Windows-only thread shim
├── standalone/                     # CMake CLI build + vendored Eigen 3.1.3, tclap
└── tests/testthat/                 # 44 test files — the numerical ground truth
```

Two important structural facts:

1. **The core is already language-neutral.** `src/cyclops/` contains no live Rcpp
   dependency — see §6.
2. **A non-R build already exists** (`standalone/`, driven by the root `CMakeLists.txt`),
   which is strong evidence upstream intends the core to be reusable. It is, however,
   bit-rotted (`cmake_minimum_required(VERSION 2.6)`, a hard-coded R 3.5 framework path).

---

## 2. Major C++ classes

### 2.1 Data layer

| Class | Role |
|---|---|
| `bsccs::AbstractModelData` | Pure-virtual, floating-point-agnostic façade over model data. ~45 virtual methods, **all in terms of `std::vector`/`std::string`** — no Rcpp. |
| `bsccs::ModelData<RealType>` | The implementation. Explicitly instantiated for `double` and `float` (`ModelData.cpp:937-938`). Holds `pid` (stratum index per row), `y`, `z`, `offs` (time), row labels, and the design matrix `X`. |
| `bsccs::CompressedDataMatrix<RealType>` | Column-major store of `CompressedDataColumn<RealType>`. |
| `bsccs::CompressedDataColumn<RealType>` | One covariate. `FormatType` ∈ {`DENSE`, `SPARSE`, `INDICATOR`, `INTERCEPT`} plus an `int64_t` numerical label (the OHDSI covariate id). |
| `bsccs::SparseIndexer<RealType>` | id → column-index map used while loading. |
| `Iterators.h` | `DenseIterator`, `SparseIterator`, `IndicatorIterator`, `InterceptIterator`, `GroupByIterator` — compile-time dispatch so the hot loops never branch on format. |
| `bsccs::RcppModelData<RealType>` | **R-only** subclass adding `Rcpp::NumericVector`-based constructors and `standardize()`. Not required to use the core. |

The four column formats are the heart of Cyclops' performance story: an `INDICATOR`
column stores only row indices (no values), which is what makes 100 k-covariate
observational-health designs tractable.

### 2.2 Optimizer

`bsccs::CyclicCoordinateDescent` (`CyclicCoordinateDescent.{h,cpp}`) owns the mode-finding
loop. It holds a reference to `AbstractModelData`, a reference to an
`AbstractModelSpecifics`, a `priors::JointPriorPtr`, and a `ProgressLogger`/`ErrorHandler`
pair. Key surface:

- `update(const ModeFindingArguments&)` — entry point; optionally runs the "KKT swindle"
  (active-set screening) then `findMode(...)`.
- `findMode(maxIterations, convergenceType, epsilon, algorithmType, qQN, doItAll)` — the
  loop: for each index `j`, `ccdUpdateBeta(j)` → `applyBounds` → `updateXBeta(delta, j)`.
- `getLogLikelihood()`, `getLogPrior()`, `getObjectiveFunction(convergenceType)`.
- `getBeta(i)`, `setBeta`, `setStartingBeta`, `setFixedBeta`, `getIsRegularized`.
- `getHessianDiagonal(i)`, `getAsymptoticVariance(i,j)`,
  `computeFisherInformation(indices)` → `Eigen::MatrixXd`.
- `getPredictiveEstimates(double* y, double* weights)`.
- `getUpdateReturnFlag()` → `UpdateReturnFlags` ∈ {`SUCCESS`, `FAIL`, `MAX_ITERATIONS`,
  `ILLCONDITIONED`, `MISSING_COVARIATES`, `POOR_BLR_STEP`}, `getIterationCount()`.
- `getSchoenfeldResiduals(...)` (Cox diagnostics).

Trust-region-ish safeguarding lives in `applyBounds`/`resetBounds` (`initialBound`,
`maxBoundCount`).

### 2.3 Likelihood kernels

`bsccs::AbstractModelSpecifics` is the strategy object supplying everything
likelihood-specific: `computeGradientAndHessian`, `computeNumeratorForGradient`,
`updateXBeta`, `computeRemainingStatistics`, `getLogLikelihood`,
`getPredictiveLogLikelihood`, `getPredictiveEstimates`, `computeFisherInformation`,
`computeSchoenfeldResiduals`, …

Construction is via a static factory — **already language-neutral**:

```cpp
static AbstractModelSpecifics* factory(ModelType modelType,
                                       const AbstractModelData& modelData,
                                       DeviceType deviceType,
                                       const std::string& deviceName);
```

It switches on `modelData.getPrecisionType()` then delegates to a template
`precisionFactory<RealType>` which instantiates
`ModelSpecifics<ModelKind<RealType>, RealType>` (`engine/ModelSpecifics.h[pp]`, ~4 500
lines of templated kernels). `engine/*Gpu*.hpp` + `Cuda*.cu` provide the optional GPU
path, compiled only in the separate `CyclopsGPU`/OpenCL builds.

`ModelType` (`Types.h:129`) enumerates: `NORMAL`, `POISSON`, `LOGISTIC`,
`CONDITIONAL_LOGISTIC`, `TIED_CONDITIONAL_LOGISTIC`, `EFRON_CONDITIONAL_LOGISTIC`,
`CONDITIONAL_POISSON`, `SELF_CONTROLLED_MODEL`, `COX`, `COX_RAW`, `TIME_VARYING_COX`,
`FINE_GRAY`.

Note `ModelSpecifics.hpp:2317`: `const int i = (k < hPidSize) ? hPid[k] : k;` — when no
stratum vector is supplied, `pid` degrades to the identity, which is exactly what
unstratified logistic/Poisson/normal models need.

### 2.4 Priors

```
priors::JointPrior (ABC)
├── FullyExchangeableJointPrior   — one CovariatePrior for all columns
├── MixtureJointPrior             — per-column CovariatePrior (used for exclusions)
└── HierarchicalJointPrior        — two-level normal-normal

priors::CovariatePrior (ABC)      — static makePrior(PriorType, variance)
├── NoPrior, LaplacePrior, NormalPrior, BarUpdatePrior, JeffreysPrior
└── FusedLaplacePrior             — neighbourhood fusion
```

`CovariatePrior::makePrior(PriorType, double variance)` is Rcpp-free. Variances are shared
through `VariancePtr` so cross-validation can retune a hyperparameter in place.

### 2.5 Orchestration

`bsccs::CcdInterface` (`CcdInterface.{h,cpp}`) is the abstract driver. It owns a
`CCDArguments` aggregate (`ModeFindingArguments` + `CrossValidationArguments` +
`ComputeDeviceArguments`) and implements the language-neutral verbs:

`initializeModel`, `fitModel`, `runFitMLEAtMode`, `predictModel`, `profileModel`,
`evaluateProfileModel`, `runCrossValidation`, `runBoostrap`, `logModel`, `diagnoseModel`,
`setZeroBetaAsFixed`.

Four hooks are pure-virtual and supplied by the language binding:
`initializeModelImpl`, `predictModelImpl`, `logModelImpl`, `diagnoseModelImpl`. The two
existing implementations are `RcppCcdInterface` (R) and `CmdLineCcdInterface`
(`standalone/`).

Cross-validation lives in `drivers/`:
`GridSearchCrossValidationDriver` and `AutoSearchCrossValidationDriver` (Brent search via
`utils/HParSearch`), with `CrossValidationSelector` / `ProportionSelector` /
`BootstrapSelector` choosing folds `BY_PID` or `BY_ROW`.

### 2.6 Logging / error handling — the key extension point

```cpp
// src/cyclops/io/ProgressLogger.h
class ProgressLogger { virtual void writeLine(const std::ostringstream&) = 0;
                       virtual void yield() = 0; ... };
class ErrorHandler   { virtual void throwError(const std::ostringstream&) = 0; ... };
```

The R binding injects `RcppProgressLogger` (→ `Rcpp::Rcout`, `R_CheckUserInterrupt`) and
`RcppErrorHandler` (→ `Rcpp::stop`). A Python binding injects its own. This abstraction is
already correct and needs no change.

---

## 3. Optimization pipeline (end to end)

```
1. Build data          ModelData<double>::loadY / loadMultipleX / addIntercept
                       / setOffsetCovariate / setIsFinalized
2. Choose kernel       AbstractModelSpecifics::factory(modelType, data, CPU, "native")
3. Construct optimizer new CyclicCoordinateDescent(data, specifics, prior, logger, error)
4. Configure           setPrior, setNoiseLevel, setWeights, setStartingBeta,
                       setFixedBeta, CCDArguments{modeFinding, crossValidation}
5a. (optional) CV      CcdInterface::runCrossValidation  → tunes prior variance,
                                                           refits at optimum
5b. Fit                CcdInterface::fitModel → ccd.update(modeFinding)
                          └─ [kktSwindle] → findMode
                                └─ per index j:  ccdUpdateBeta(j)
                                                 → computeGradientAndHessian (kernel)
                                                 → + prior gradient/hessian
                                                 → applyBounds
                                                 → updateXBeta(delta, j)
                                                 → performCheckConvergence
6. Read out            ccd.getBeta(i), getLogLikelihood, getLogPrior,
                       getUpdateReturnFlag, getIterationCount, getHyperprior
7. Predict             ccd.getPredictiveEstimates(y, weights)
```

`DiagnosticsOutputWriter` (`io/OutputWriter.h:221`) defines the canonical result payload
that the R `cyclopsFit` object exposes: `log_likelihood`, `log_prior`, `return_flag`,
`iterations`, `prior_info`, `variance`, `covariate_count`, `cross_validation`.

---

## 4. Where data enters

There are two data-entry paths, and only one of them is Rcpp-coupled.

**(a) The "matrix" path — Rcpp-coupled.** `.cyclopsModelData` (`RcppModelData.cpp:707`)
takes R S4 `Matrix` objects (`dgCMatrix` for sparse, `ngCMatrix` for indicator) plus
`pid`/`y`/`z`/`offs`, and constructs `RcppModelData<RealType>` directly from the R slots.
Used by `createCyclopsData(formula, ...)` and the X/y constructor.

**(b) The "SQL"/streaming path — already Rcpp-free.** `AbstractModelData` exposes:

```cpp
void loadY(const std::vector<IdType>& stratumId,   // may be empty
           const std::vector<IdType>& rowId,       // may be empty
           const std::vector<double>& y,
           const std::vector<double>& time);       // may be empty

int loadMultipleX(const std::vector<int64_t>& covariateId,
                  const std::vector<int64_t>& rowId,
                  const std::vector<double>& covariateValue,
                  bool checkCovariateIds, bool checkCovariateBounds,
                  bool append, bool forceSparse);
```

`loadMultipleX` (`ModelData.cpp:147`) requires entries **grouped by `covariateId`, and
ascending in `rowId` within each group** — i.e. exactly compressed-sparse-column layout.
It auto-selects `INDICATOR` vs `SPARSE` per column based on whether the first value is
0/1, and upcasts `INDICATOR → SPARSE` on encountering a non-{0,1} value. Explicit zeros
are dropped.

This is the path the Python binding uses: a `scipy.sparse.csc_matrix` maps onto it with no
transformation beyond `indptr → repeat(ids, counts)`.

Two behavioural details that matter for correctness:

- If `loadY` is given an empty `rowId`, `rowIdMap` stays empty and `loadMultipleX` treats
  `rowId` values as 0-based row indices directly.
- If `loadY` is given an empty `stratumId`, `pid` stays empty, `nPatients = nRows`, and the
  kernels fall back to the row identity (§2.3).

---

## 5. Where results are returned

R-side (`ModelFit.R:66-360`), `fitCyclopsModel` performs this sequence — reproduced
faithfully by the Python facade so results match bit-for-bit:

1. `.checkInterface` → `.cyclopsInitializeModel(dataPtr, modelType, computeDevice,
   computeMLE = TRUE)`.
2. Prior setup. **If the prior is not `"none"`, the data has an intercept, and
   `forceIntercept` is `FALSE`, the intercept is added to `prior$exclude`** (and a warning
   is emitted).
3. `selectorType == "auto"` resolution: `"byRow"` for `lr`/`pr`; otherwise `"byPid"` if
   `rowsPerStratum < nStrata`, else `"byRow"`.
4. `.cyclopsSetControl(...)`.
5. Starting values. **If the data has an intercept and no starting coefficients were
   supplied, the intercept is initialised to `log(ȳ/(1-ȳ))` (`lr`), `log(ȳ)` (`pr`) or `ȳ`
   (`ls`), all other coefficients to 0.** This is not cosmetic — CCD is only locally
   convergent for these likelihoods and the starting point changes the iterate path.
6. Weights / censor weights.
7. `.cyclopsRunCrossValidation` if `prior$useCrossValidation`, else `.cyclopsFitModel`.
8. **If `return_flag == "POOR_BLR_STEP"` and `convergenceType == "gradient"`, refit with
   `convergenceType = "lange"`.**
9. `.cyclopsLogModel` → `(column_label, estimate)` pairs.

Note that all coefficient read-out loops start at index `1` rather than `0` when
`getHasOffsetCovariate()` is true — the offset occupies column 0 with label `-1`.

---

## 6. Dependency graph of the core

```
src/cyclops/**  ──depends on──►  C++14 standard library
                             ►  Eigen (headers only; Dense only)
                             ►  tinythread   (Windows only, via Thread.h)
                             ►  ProgressLogger / ErrorHandler   (injected)
```

- **No Boost.** Every `boost/` include in `src/cyclops/` is commented out; `CcdInterface`
  ships its own `IncrementableIterator` in place of `boost::counting_iterator`. (`BH` is
  not even in `DESCRIPTION`'s `LinkingTo`.)
- **No RcppParallel.** `engine/ParallelLoops.h:11` does `#undef USE_RCPP_PARALLEL`, so all
  the `::RcppParallel::` code is dead. Threading goes through `Thread.h` +
  `engine/ThreadPool.h`.
- **Eigen** is used only for `CyclicCoordinateDescent::Matrix` (Fisher information and the
  asymptotic variance matrix). In the R build it comes from `RcppEigen`; a Python build
  needs a standalone Eigen (fetched by CMake).
- **Live Rcpp coupling inside the core: exactly three lines.**
  `priors/NewCovariatePrior.h:182,185,188` call `Rcpp::stop(...)` in the three
  not-yet-implemented parameterized-prior branches. Everything else is in comments. This is
  the single core change the Python build requires, and it is an unambiguous upstream
  improvement (see `MIGRATION.md`).

---

## 7. The R interface

`src/RcppCyclopsInterface.cpp` (45 KB) and `src/RcppModelData.cpp` (32 KB) expose ~60
`// [[Rcpp::export]]` functions, all `.`-prefixed and internal. The pattern throughout is:

```cpp
XPtr<RcppCcdInterface> interface(inRcppCcdInterface);   // opaque handle from R
interface->doSomething();
return List::create(Named("x") = ..., Named("y") = ...);
```

State lives in two `Rcpp::XPtr`-held objects stashed in an R environment:
`cyclopsDataPtr` (`AbstractModelData`) and `cyclopsInterfacePtr` (`RcppCcdInterface`, which
owns the `CyclicCoordinateDescent` and `AbstractModelSpecifics`). `RcppCcdInterface` also
carries an `Rcpp::List result` member that the `*OutputWriter` classes fill in via
`OutputHelper::RcppOutputHelper`.

This is a thin, faithful translation layer — there is essentially no algorithmic logic in
the Rcpp files, which is why a parallel Python layer duplicates nothing of substance.

---

## 8. Proposed Python interface

### 8.1 Layering

```
   cyclops (Python)                     python/src/cyclops/*.py
     LogisticRegression, PoissonRegression, CoxRegression,
     ConditionalLogisticRegression, ...  (scikit-learn-shaped)
        │
        ▼
   _cyclops (pybind11 extension)        python/src/bindings.cpp
     thin, mechanical; no logic
        │
        ▼
   cyclops::api (C++ facade)            src/cyclops/api/*.{h,cpp}   ← language-neutral
     ModelData, Model, FitOptions, PriorOptions, FitResult
        │
        ▼
   bsccs (existing implementation)      src/cyclops/**              ← untouched
```

The facade lives in `src/cyclops/api/` rather than under `python/` on purpose: it is
Rcpp-free, Python-free, and reusable by the JNI and CLI front-ends. It is *purely
additive* — a new directory that neither `Makevars.in` (explicit object list) nor
`Makevars.win.in` (`cyclops/*.cpp` wildcard, one level only) picks up, so the R build is
untouched. See `DESIGN_DECISIONS.md` §3.

### 8.2 Facade surface

```cpp
namespace cyclops::api {

enum class ModelKind { Normal, Poisson, Logistic, ConditionalLogistic,
                       TiedConditionalLogistic, EfronConditionalLogistic,
                       ConditionalPoisson, SelfControlledCaseSeries,
                       Cox, CoxRaw, TimeVaryingCox, FineGray };
enum class PriorKind { None, Laplace, Normal, BarUpdate, Jeffreys };
enum class ConvergenceKind { Gradient, Lange, Mittal, OneStep, ZhangOles };
enum class SelectorKind  { Auto, ByPid, ByRow };
enum class AlgorithmKind { Ccd, Mm };
enum class NormalizationKind { StandardDeviation, Max, Median, Q95 };
enum class NoiseLevel { Silent, Quiet, Noisy };
enum class Precision { Fp64, Fp32 };

struct PriorOptions { PriorKind kind; double variance; bool use_cross_validation;
                      std::vector<int64_t> exclude; bool force_intercept;
                      std::vector<PriorKind> kinds;      // per-covariate override
                      std::vector<double>    variances; };

struct FitOptions  { /* mode finding */ int max_iterations; double tolerance;
                     ConvergenceKind convergence; AlgorithmKind algorithm;
                     double initial_bound; int max_bound_count;
                     bool use_kkt_swindle; int swindle_multiplier; bool do_it_all;
                     /* cross-validation */ bool auto_search; int fold;
                     int cv_repetitions; double lower_limit, upper_limit;
                     int grid_steps; double starting_variance;
                     SelectorKind selector; int min_cv_data;
                     /* misc */ NoiseLevel noise; int threads; long seed;
                     bool reset_coefficients; };

struct FitResult   { std::vector<int64_t> covariate_ids;
                     std::vector<double>  coefficients;
                     double log_likelihood, log_prior;
                     std::string return_flag; int iterations;
                     std::string prior_info; std::vector<double> variance;
                     std::string cross_validation_info;
                     int covariate_count; double fit_seconds; };

class ModelData {                       // opaque; owns bsccs::AbstractModelData
  static ModelDataPtr create(ModelKind, Precision, bool silent);
  void set_outcome(y, time, stratum_id, row_id);
  void add_covariates_csc(indptr, row_indices, values, covariate_ids, force_sparse);
  void add_dense_covariate(covariate_id, values);
  void add_intercept();
  void set_offset_covariate(int64_t id, bool already_log);   // id == -1 ⇒ use `time`
  std::vector<double> normalize(NormalizationKind);
  void finalize();
  /* introspection: row_count, covariate_count, stratum_count, covariate_ids,
     covariate_types, has_intercept, has_offset, y, time, intercept_label,
     univariable_correlation, sum, sum_by_stratum, ... */
};

class Model {                           // opaque; owns the CCD + kernels
  Model(ModelData&, ComputeDevice);
  void set_prior(const PriorOptions&);
  void set_options(const FitOptions&);
  void set_weights(std::vector<double>);
  void set_censor_weights(std::vector<double>);
  void set_start_values(std::vector<double>);
  void set_fixed(std::vector<bool>);
  FitResult fit();                      // dispatches CV vs plain fit
  std::vector<double> predict();        // getPredictiveEstimates
  std::vector<double> linear_predictor();
  double log_likelihood();
  std::vector<double> gradient();
  std::vector<double> hessian_diagonal(const std::vector<int64_t>& ids);
  std::vector<double> standard_errors(const std::vector<int64_t>& ids);
  ProfileResult profile(...);           // likelihood profile CIs
};

}  // namespace cyclops::api
```

Everything is `std::vector`, `std::string`, POD, or an opaque handle. Nothing from `bsccs`
appears in the facade headers.

### 8.3 Python surface

scikit-learn-shaped estimators, *not* a transliteration of the R API:

```python
from cyclops import LogisticRegression

model = LogisticRegression(prior="laplace", prior_variance=0.1, max_iter=500)
model.fit(X, y)                 # X: ndarray | scipy.sparse; y: ndarray
model.coef_, model.intercept_
model.predict_proba(X)
model.log_likelihood_, model.n_iter_, model.converged_
```

- `X` accepts `numpy.ndarray`, `scipy.sparse.csc_matrix`/`csr_matrix`, or anything
  `scipy.sparse` can convert. Sparse input is passed straight through as CSC — no
  densification anywhere.
- `LogisticRegression(prior="laplace", prior_variance="cv")` triggers Cyclops'
  cross-validated regularisation search, which is the feature that has no scikit-learn
  equivalent.
- Stratified models (`ConditionalLogisticRegression`, `CoxRegression`, …) take
  `strata=`/`time=`/`event=` in `fit`, because there is no sklearn convention to borrow.
- A lower-level `cyclops.data.CyclopsData` + `cyclops.model.fit(...)` pair is available for
  the OHDSI use-case of streaming 10⁵-covariate designs in column chunks.

See `DESIGN_DECISIONS.md` for the rationale behind each of these choices, and `TODO.md`
for what is implemented today versus planned.
