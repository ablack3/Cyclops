# Changes outside `python/`

Every modification to files the R package shares, with rationale, alternatives
considered, and an upstream recommendation. The goal is that a future
`git merge upstream/main` touches none of these except by coincidence.

Summary: **5 changes, 7 files edited, 2 files added.** All are Rcpp-free,
Python-free, and additive or bug-fixing. None changes numerical results for
existing R code paths.

| # | Path | Kind | Upstream PR? |
|---|---|---|---|
| 1 | `src/cyclops/priors/NewCovariatePrior.h` | fix: remove hidden Rcpp dependency | **Yes** — standalone |
| 2 | `src/cyclops/ModelData.h`, `src/cyclops/ModelData.cpp` | fix: out-of-bounds read on unstratified data | **Yes** — standalone |
| 3 | `src/cyclops/CcdInterface.{h,cpp}`, `src/cyclops/Timer.{h,cpp}` | fix: POSIX-only timing breaks the MSVC build | **Yes** — standalone |
| 4 | `src/cyclops/api/` (new) | feature: language-neutral C++ facade | **Yes** — after the fixes land |
| 5 | `src/CMakeLists.txt` (new) | build: standalone CMake target | **Yes** — with or after 4 |

Nothing in `R/`, `src/Rcpp*`, `src/Makevars*`, `DESCRIPTION`, `NAMESPACE`,
`configure`, or `tests/testthat/` was touched.

---

## 1. `NewCovariatePrior.h` — remove the hidden Rcpp dependency

**Change.** Three `Rcpp::stop(...)` calls in `priors::makePrior` become
`throw std::runtime_error(...)`, plus `#include <stdexcept>`.

```diff
   case NORMAL :
-      Rcpp::stop("Parameterized normal priors are not yet implemented");
+      throw std::runtime_error("Parameterized normal priors are not yet implemented");
```

**Why it was needed.** This was the *only* live Rcpp reference in
`src/cyclops/` (everything else is inside comments), so the core could not be
compiled without Rcpp on the include path. The header also never included
`Rcpp.h` — it compiled solely because `RcppCyclopsInterface.h` happens to
include `Rcpp.h` first, making it unusable from any other translation unit.

**Behaviour for R users: unchanged.** `RcppExports.cpp` wraps every entry point
in `BEGIN_RCPP`/`END_RCPP`, which catches `std::exception&` and converts it to an
R error. The message text is identical. All three branches are
not-yet-implemented paths that no released code reaches.

**Alternatives considered.**

- *Route through the injected `ErrorHandler`.* Architecturally the better fit —
  it is exactly what `ProgressLogger.h` exists for — but `makePrior` is a free
  function with no handler in scope. Threading one in changes the signature and
  its two call sites, a larger diff for no behavioural gain.
- *Guard with `#ifdef R_BUILD`.* `R_BUILD` is defined in both `Makevars` files
  and used nowhere in the codebase. Reviving it to gate an Rcpp call would
  entrench the coupling this change removes.
- *Leave it and add Rcpp to the Python build.* Would drag R itself into the
  Python wheel. Non-starter.

**Upstream recommendation: submit as a standalone PR.** Small, self-contained,
strictly reduces coupling, and makes the header self-sufficient. Good first PR
because it needs no discussion of the Python work.

---

## 2. `ModelData` — out-of-bounds read when data are unstratified

**Change.** Added `ModelData<RealType>::getStratumIndexRef()`, made
`binaryReductionByStratum` use it instead of the raw `pid` member, and
reduced `getPidVectorSTL()` to a copy of it.

```diff
   template <typename T, typename F>
   void binaryReductionByStratum(T& out, const size_t reductionIndex, F func) const {
-      binaryReductionByGroup(out, reductionIndex, pid, func);
+      binaryReductionByGroup(out, reductionIndex, getStratumIndexRef(), func);
   }
```

`getStratumIndexRef()` returns `pid` when it is fully populated, and otherwise a
lazily built identity vector (`0, 1, …, K-1`) held in a new `mutable` member.

**Why it was needed.** `loadY()` only fills `pid` when it is given stratum IDs
*and* row IDs, so unstratified data (`lr`, `pr`, `ls` loaded through
`convertToCyclopsData` or the equivalent streaming API) leave it empty.
`binaryReductionByGroup` then evaluates `out[groups[it.index()]]` with
`groups.size() == 0`, reading past the end of the vector.

This is reached from `ModelSpecifics::initializeMM` under the default
`MmBoundType::METHOD_2` — that is, from any `algorithm = "mm"` fit on
unstratified data. Observed consequences:

- **In R (3.7.1, released):** no crash. `loadY` calls `pid.reserve(n)` when row
  IDs are present, so the read lands in allocated-but-uninitialized capacity,
  which the allocator has zero-filled. Every row is silently treated as stratum
  0, giving a *looser but still valid* MM bound. The fit converges to the right
  mode, so the bug is invisible — it just does more iterations than it should,
  and depends on allocator behaviour for that.
- **Through the C++ facade:** hard segmentation fault. No `reserve` happens
  because no row IDs are supplied, so `pid.data()` is null.

After the fix, MM on unstratified data agrees exactly with R's `createCyclopsData`
formula path, which sets `pid <- 1:length(y)` and therefore always had the
correct bound. Verified: identical coefficients to 10 significant digits and an
identical iteration count (27) on the same data.

**Behaviour for R users: strictly improved.** The formula path is unaffected
(`pid` is already populated, so the same reference is returned). The
`convertToCyclopsData` path stops reading uninitialized memory and gets the
tighter, intended per-row bound — same optimum, fewer iterations. No non-MM code
path reads `binaryReductionByStratum`.

**Alternatives considered.**

- *Populate `pid` with the identity inside `loadY`.* The cleanest conceptual fix
  — it removes the "empty means identity" convention entirely — but it adds 4
  bytes per row for every R user and changes an invariant that
  `getPidVectorSTL()`, `AbstractModelSpecifics`' constructor, and
  `ModelSpecifics.hpp:2317` (`k < hPidSize ? hPid[k] : k`) all currently rely on.
  Too broad to land without discussion.
- *Work around it in the Python layer* by always passing `strata = arange(n)`
  for unstratified models. Hides a genuine memory-safety bug that R users are
  also exposed to, and costs one row-label string per row (≈24 MB per million
  rows) because `loadY` only fills `pid` when row IDs are present.
- *Guard inside `binaryReductionByGroup`.* Wrong layer: that function correctly
  takes whatever group vector it is handed. The defect is in the caller's choice
  of vector.

**Upstream recommendation: submit as a standalone PR**, ideally before the facade
PR. It is a memory-safety fix with a reproducer that needs no Python:

```r
outcomes   <- data.frame(rowId = 1:100, y = rbinom(100, 1, 0.5))
covariates <- data.frame(rowId = rep(1:100, 2), covariateId = rep(1:2, each = 100),
                         covariateValue = rnorm(200))
data <- convertToCyclopsData(outcomes, covariates, modelType = "lr")
fitCyclopsModel(data, control = createControl(algorithm = "mm"))
# runs, but initializeMM() reads past the end of `pid`
```

Worth pairing with a `testthat` case that compares the MM fit from
`convertToCyclopsData` against the one from `createCyclopsData`.

---

## 3. Timing — replace `gettimeofday` with `std::chrono`

**Change.** `CcdInterface` and `Timer` measured elapsed time with
`gettimeofday()` and `struct timeval`. Both now use `bsccs::chrono`, the
codebase's own `<chrono>` wrapper in `src/cyclops/Timing.h`, which
`RcppCyclopsInterface.cpp` and `engine/ModelSpecifics.hpp` already use.

```diff
-	struct timeval time1, time2;
-	gettimeofday(&time1, NULL);
+	const auto time1 = now();
 	...
-	gettimeofday(&time2, NULL);
+	const auto time2 = now();
 	return calculateSeconds(time1, time2);
```

`calculateSeconds` keeps its name and role; its parameters become
`CcdInterface::TimePoint`. The 63-line MSVC block in `CcdInterface.cpp` is
deleted.

**Why it was needed.** The core does not compile with MSVC. Building the
extension on `windows-latest` fails with:

```
src\cyclops\CcdInterface.cpp(119,4): error C2027: use of undefined type 'bsccs::timeval'
```

Three separate defects combine:

1. `CcdInterface.h` declares `calculateSeconds(const struct timeval&, ...)`
   without including any header that defines `timeval`. In C++ that
   elaborated-type-specifier *declares* `bsccs::timeval` as a new incomplete
   type, so the error surfaces at the call site rather than the declaration.
   The header did once include `<sys/time.h>` / `<winsock.h>`, but the whole
   block is commented out.
2. `Timer.h` includes `<sys/time.h>` unconditionally, which MSVC does not ship.
3. The `#ifdef _MSC_VER` shim in `CcdInterface.cpp` that supplies
   `gettimeofday` references `FILETIME` and `GetSystemTimeAsFileTime` without
   including `<windows.h>`, and never declares `timeval` — so it could not have
   compiled either. It is dead code.

None of this affects R, because R on Windows builds with MinGW (Rtools), where
`_MSC_VER` is undefined and `<sys/time.h>` and `gettimeofday` both exist. The
MSVC path has therefore never been exercised — which is why an incomplete shim
survived in the tree.

**Behaviour for R users: unchanged apart from the clock.** The reported
durations (`timeLoad`, `timeFit`, `timeUpdate`) now come from a monotonic
`steady_clock` instead of wall-clock `gettimeofday`, which is what measuring an
interval wants: immune to NTP steps and clock adjustments. No coefficient, log
likelihood, or convergence result depends on these values. Verified: `testthat`
gives 248 passed / 0 failed before and after.

**Alternatives considered.**

- *Include `<winsock2.h>` under `_MSC_VER` and repair the shim.* The smallest
  diff, and it restores what the commented-out block intended. Rejected because
  `<winsock2.h>` is order-sensitive with `<windows.h>`, the shim would still need
  `<windows.h>` added, and the result could only be validated by pushing to CI
  and waiting — whereas the `<chrono>` version compiles everywhere and is
  verifiable locally.
- *Build the Windows wheels with MinGW*, matching R. Mixing a MinGW-built
  extension with an MSVC-built CPython is a known source of ABI trouble; MSVC is
  the convention for Windows wheels.
- *Leave the core alone and stub the timings out of the facade.* The facade does
  not call `gettimeofday`; `CcdInterface.cpp` does, and it is compiled either
  way.

**Upstream recommendation: submit as a standalone PR.** It deletes 63 lines of
unreachable platform code, makes two headers self-contained, and lets the core
build with MSVC for the first time. Independent of the Python work.

---

## 4. `src/cyclops/api/` — the language-neutral C++ facade (new files)

**Change.** Two new files, no existing file modified:

- `src/cyclops/api/CyclopsApi.h` — the stable API: `ModelData`, `Model`,
  `FitOptions`, `PriorOptions`, `FitResult`, and the enums. Standard-library
  types only; no `bsccs` type appears, and the implementation is hidden behind
  pImpl.
- `src/cyclops/api/CyclopsApi.cpp` — the translation, including
  `ApiCcdInterface`, a fourth `CcdInterface` implementation alongside
  `RcppCcdInterface` and `CmdLineCcdInterface`.
- `src/cyclops/api/BufferedLogger.h` — host-agnostic `ProgressLogger` and
  `ErrorHandler` implementations, the plain-C++ counterparts of
  `src/RcppProgressLogger.h`.

**Why in `src/` rather than `python/`.** The facade contains nothing
Python-specific — it is the reusable interface the JNI layer in
`src/cyclops/jni/` and the CLI in `standalone/` could both consume instead of
each re-deriving the call sequence. Putting it under `python/` would make a
language-neutral component look language-specific and guarantee it gets
duplicated. See `DESIGN_DECISIONS.md` §3 for the full argument.

**Why it does not affect the R build.** `src/Makevars.in` names its objects
explicitly, and `src/Makevars.win.in` globs `cyclops/*.cpp` — one directory
level, so `cyclops/api/*.cpp` matches neither. The new sources are invisible to
`R CMD INSTALL`. Verified: the R package builds clean and its `testthat` suite
passes unchanged.

**Merge risk: essentially zero.** New files in a new directory cannot conflict
textually. The only coupling is to the `bsccs` APIs the facade calls; if upstream
changes one of those, the facade needs the same edit any other caller would.

**Upstream recommendation: propose after changes 1-3 land.** Frame it as
"reusable non-R interface" rather than "Python support" — the value to OHDSI is
that the JNI layer and CLI stop duplicating orchestration logic, and that a
Python/Julia/Rust binding becomes a mechanical exercise. Expect discussion about
API scope; the header is deliberately small so that conversation is tractable.

---

## 5. `src/CMakeLists.txt` — standalone library target (new file)

**Change.** A new `src/CMakeLists.txt` defining `cyclops::core`: a static library
built from the source list in `src/Makevars.in` plus `src/cyclops/api`, with
Eigen and `Threads::Threads` as dependencies and `DOUBLE_PRECISION` /
`CYCLOPS_VERSION` as compile definitions. `CYCLOPS_VERSION` is parsed from
`DESCRIPTION`, so there is no second place to bump the version.

**Why it was needed.** Non-R front-ends currently have to re-list ~20
translation units by hand. `python/CMakeLists.txt` is nine lines because this
file exists.

**Why not extend the existing root `CMakeLists.txt`.** It is bit-rotted:
`cmake_minimum_required(VERSION 2.6)` (rejected outright by CMake ≥ 4),
`project(SCCS)`, a hard-coded `/Library/Frameworks/R.framework/Versions/3.5`
path, `-O0` unconditionally, and it only descends into
`standalone/codebase/CCD-DP`. Repairing it is a separate, larger piece of work
that would also have to modernise `standalone/`; adding `src/CMakeLists.txt` is
orthogonal and can be consumed via `add_subdirectory` today.

**Alternatives considered.**

- *Keep the target definition in `python/CMakeLists.txt`.* Then the source list
  lives under `python/`, and the JNI or CLI builds cannot reuse it — the exact
  duplication this file removes.
- *Vendor Eigen.* `standalone/codebase/Eigen` holds Eigen 3.1.3 (circa 2012).
  `find_package(Eigen3)` with a pinned `FetchContent` fallback keeps wheel builds
  reproducible without adding another copy of Eigen to the tree.

**Upstream recommendation: submit with or just after change 4.** Purely additive;
`R CMD INSTALL` never invokes CMake. A natural follow-up is to have
`standalone/` consume the same target, which would delete most of
`standalone/codebase/CCD-DP/CMakeLists.txt`.

---

## Observations reported but *not* changed

Found while reading the code. Each is an upstream bug or wart; none blocks the
Python bindings, so all were left alone to keep the diff minimal.

1. **`src/Makevars.in` never compiles TinyThread.** `OBJECTS` references
   `$(OBJECTS.threads)` while the variable is defined as `OBJECTS.thread`
   (singular), so `tinythread/tinythread.o` is silently dropped on Unix. Benign
   today — `Thread.h` only routes through TinyThread when `WIN_BUILD` is defined,
   and Windows uses the wildcard `Makevars.win.in` — but the typo will bite
   whoever enables TinyThread elsewhere.

2. **`convertToCyclopsData(modelType = "pr")` segfaults without a `time`
   column.** `NewDataConversion.R:267` unconditionally calls
   `finalizeSqlCyclopsData(useOffsetCovariate = -1)` for `pr` and `cpr`, which
   promotes the `time` vector to the offset column. When no `time` was supplied,
   `moveTimeToCovariate` pushes an empty column and the fit crashes. Reproducer:

   ```r
   outcomes   <- data.frame(rowId = 1:20, y = rpois(20, 2))
   covariates <- data.frame(rowId = rep(1:20, 2), covariateId = rep(1:2, each = 20),
                            covariateValue = rnorm(40))
   fitCyclopsModel(convertToCyclopsData(outcomes, covariates, modelType = "pr"))
   #  *** caught segfault ***
   ```

   The fix is a validation error in `convertToCyclopsData`, or defaulting `time`
   to 1. There is already a `// TODO SEGV in Poisson model` comment at
   `engine/ModelSpecifics.hpp:594`, so the fragility is known.

3. **`convertToCyclopsData` needs `stratumId` on the *covariates* frame** for
   `clr`/`cpr`, but only documents it for `outcomes`. The Cox branch merges it in
   from `outcomes` (`NewDataConversion.R:222`); the conditional branch does not,
   and instead fails with `Index out of bounds: [index='stratumId']`.

4. **`Models::removeIntercept` disagrees with
   `.cyclopsGetRemoveInterceptNames()`.** The C++ helper (`Types.h:189`) omits
   the four survival models that the R-level list includes. The facade follows
   the R list, since that is the user-facing contract; the two should be
   reconciled and one made the single source of truth.

5. **`ModelData::moveTimeToCovariate(bool takeLog)` ignores `takeLog`.** Callers
   compensate by calling `logTransformCovariate(0)` afterwards
   (`RcppModelData.cpp:549`). The dead parameter invites exactly the
   double-logging bug it looks like it prevents.

6. **Cox models cannot produce asymptotic standard errors.**
   `CyclicCoordinateDescent::computeFisherInformation` returns a singular matrix
   for the Cox likelihood, so `getSEs()` fails in R with a LAPACK error:

   ```r
   test <- data.frame(length = c(4, 3.5, 3, 2.5, 2, 1.5, 1),
                      event  = c(1, 1, 0, 1, 1, 0, 1),
                      x1 = c(0, 2, 0, 0, 1, 1, 1), x2 = c(0, 0, 1, 1, 1, 0, 0))
   fit <- fitCyclopsModel(createCyclopsData(Surv(length, event) ~ x1 + x2,
                                            data = test, modelType = "cox"))
   Cyclops:::getSEs(fit, c(1, 2))
   #  Error: Lapack routine dgesv: system is exactly singular
   ```

   The coefficients are correct — they match `coxph` to 1e-10 — so only the
   curvature is affected. The R suite never compares Cox standard errors against
   `coxph`, which is consistent with the limitation being known. The facade
   raises a `CyclopsError` naming the cause rather than returning the `inf`/`NaN`
   that inverting a singular matrix produces.

7. **The Jeffreys-prior preconditions are enforced only in R.**
   `fitCyclopsModel` rejects a Jeffreys prior with more than one covariate, or
   with a non-binary covariate, before reaching the optimizer
   (`R/ModelFit.R:169-183`). Nothing in C++ checks either, so any other binding
   silently produces an undefined result. The facade reproduces both checks; they
   would be better placed in the core so every front-end inherits them.

8. **MM raises `"Non-increasing!"` at tight tolerances.** The monotonicity
   assertion in `CyclicCoordinateDescent.cpp:1241` fires on floating-point noise
   near the optimum, so `algorithm = "mm"` with `tolerance` much below `1e-6`
   aborts a fit that has effectively converged. A relative epsilon would be more
   robust than `change < 0.0`.
