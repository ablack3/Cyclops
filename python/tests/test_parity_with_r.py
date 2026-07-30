"""Parity with the R Cyclops package.

The R package is the reference implementation: same C++ core, years of use, and
its own test-suite validated against ``glm``, ``survival::coxph``, ``gnm`` and
friends. These tests fit identical data through both front-ends and compare.

Because both paths reach the same optimizer, agreement should be far tighter
than statistical tolerance — coefficients to ~1e-10 and log likelihoods to
~1e-12. The tolerances below are deliberately near machine precision so that a
regression in the binding layer (a mis-ordered row, a dropped option, a
different warm start) fails loudly instead of hiding inside a loose 1e-4.

Run with:  pytest tests/ -m parity
"""

from __future__ import annotations

import numpy as np
import pytest

from cyclops import (
    ConditionalLogisticRegression,
    ConditionalPoissonRegression,
    CoxRegression,
    LinearRegression,
    LogisticRegression,
    PoissonRegression,
    SelfControlledCaseSeries,
)
from cyclops.data import CyclopsData
from cyclops.model import Control, CyclopsModel, Prior
from r_bridge import fit_in_r

pytestmark = pytest.mark.parity

#: Both front-ends run the same coordinate descent, so any disagreement beyond
#: accumulated floating-point noise is a binding bug.
COEF_TOL = 1e-9
LL_TOL = 1e-10


def _summary(python_result):
    """Read the diagnostics off either a fitted estimator or a `FitResult`.

    Estimators expose them with scikit-learn's trailing underscore; `FitResult`
    without.
    """

    def read(name):
        if hasattr(python_result, name):
            return getattr(python_result, name)
        return getattr(python_result, f"{name}_")

    return (
        read("return_flag"),
        read("log_likelihood"),
        read("log_prior"),
        read("iterations") if hasattr(python_result, "iterations") else python_result.n_iter_,
    )


def assert_matches(
    python_result,
    r_result,
    *,
    coefficients,
    coef_tol: float = COEF_TOL,
    ll_tol: float = LL_TOL,
    check_iterations: bool = True,
):
    """Compare a Python fit against an :class:`~r_bridge.RFit`."""
    return_flag, log_likelihood, log_prior, iterations = _summary(python_result)

    assert return_flag == r_result.return_flag, (
        f"convergence flag differs: python={return_flag} r={r_result.return_flag}"
    )
    assert coefficients.shape == r_result.coefficients.shape, (
        f"coefficient count differs: python={coefficients.shape[0]} "
        f"r={r_result.coefficients.shape[0]}"
    )
    np.testing.assert_allclose(
        coefficients, r_result.coefficients, rtol=coef_tol, atol=coef_tol
    )
    np.testing.assert_allclose(
        log_likelihood, r_result.log_likelihood, rtol=ll_tol, atol=ll_tol
    )
    np.testing.assert_allclose(
        log_prior, r_result.log_prior, rtol=ll_tol, atol=ll_tol
    )
    if check_iterations:
        assert iterations == r_result.iterations, (
            f"iteration count differs: python={iterations} r={r_result.iterations}"
        )


# ---------------------------------------------------------------------------
# Unconditional models
# ---------------------------------------------------------------------------


def test_logistic_unregularized(small):
    """The base case: intercept warm start, gradient convergence, no penalty."""
    model = LogisticRegression().fit(small.X, small.y)
    reference = fit_in_r(small.X, small.y, "lr", add_intercept=True)

    coefficients = np.concatenate(([model.intercept_], model.coef_))
    assert_matches(model, reference, coefficients=coefficients)
    # R labels the intercept 0 and the covariates 1..p.
    np.testing.assert_array_equal(
        np.concatenate(([0], model.covariate_ids_)), reference.covariate_ids
    )


def test_logistic_no_intercept(small):
    model = LogisticRegression(fit_intercept=False).fit(small.X, small.y)
    reference = fit_in_r(small.X, small.y, "lr", add_intercept=False)
    assert_matches(model, reference, coefficients=model.coef_)


@pytest.mark.parametrize("variance", [0.01, 0.1, 1.0, 10.0])
def test_logistic_laplace(small, variance):
    """L1 at several strengths; also checks the automatic intercept exclusion."""
    model = LogisticRegression(prior="laplace", prior_variance=variance).fit(
        small.X, small.y
    )
    reference = fit_in_r(
        small.X,
        small.y,
        "lr",
        add_intercept=True,
        prior="laplace",
        prior_variance=variance,
    )
    coefficients = np.concatenate(([model.intercept_], model.coef_))
    assert_matches(model, reference, coefficients=coefficients)
    assert model.prior_info_ == reference.prior_info


@pytest.mark.parametrize("variance", [0.05, 2.0])
def test_logistic_normal(small, variance):
    model = LogisticRegression(prior="normal", prior_variance=variance).fit(
        small.X, small.y
    )
    reference = fit_in_r(
        small.X,
        small.y,
        "lr",
        add_intercept=True,
        prior="normal",
        prior_variance=variance,
    )
    assert_matches(
        model,
        reference,
        coefficients=np.concatenate(([model.intercept_], model.coef_)),
    )


def test_logistic_excluded_covariates(small):
    """Unpenalized columns must land on the same coefficients."""
    excluded_columns = [0, 2]
    model = LogisticRegression(
        prior="laplace", prior_variance=0.1, exclude=excluded_columns
    ).fit(small.X, small.y)
    reference = fit_in_r(
        small.X,
        small.y,
        "lr",
        add_intercept=True,
        prior="laplace",
        prior_variance=0.1,
        exclude=[column + 1 for column in excluded_columns],  # R ids are 1-based
    )
    assert_matches(
        model,
        reference,
        coefficients=np.concatenate(([model.intercept_], model.coef_)),
    )


def test_logistic_sample_weights(small):
    rng = np.random.default_rng(5)
    weights = rng.integers(0, 3, size=small.n_samples).astype(np.float64)
    model = LogisticRegression().fit(small.X, small.y, sample_weight=weights)
    reference = fit_in_r(small.X, small.y, "lr", add_intercept=True, weights=weights)
    assert_matches(
        model,
        reference,
        coefficients=np.concatenate(([model.intercept_], model.coef_)),
    )


def test_logistic_predictions_match(small):
    """Python's link evaluation must agree with the C++ `predictEstimate`."""
    model = LogisticRegression().fit(small.X, small.y)
    reference = fit_in_r(
        small.X, small.y, "lr", add_intercept=True, want_predictions=True
    )
    np.testing.assert_allclose(
        model.predict_proba(small.X)[:, 1], reference.predictions, rtol=1e-9, atol=1e-12
    )


def test_logistic_standard_errors(small):
    model = LogisticRegression().fit(small.X, small.y)
    reference = fit_in_r(
        small.X, small.y, "lr", add_intercept=True, want_standard_errors=True
    )
    assert reference.standard_errors is not None
    np.testing.assert_allclose(
        model.standard_errors(), reference.standard_errors, rtol=1e-8, atol=1e-10
    )


def test_poisson_with_offset(small):
    model = PoissonRegression().fit(small.X, small.counts, offset=small.offset)
    reference = fit_in_r(
        small.X, small.counts, "pr", add_intercept=True, time=small.offset
    )
    assert_matches(
        model,
        reference,
        coefficients=np.concatenate(([model.intercept_], model.coef_)),
    )


def test_poisson_without_offset(small):
    """An offset of 1 is the only way to express "no offset" through R's
    `convertToCyclopsData`, which always promotes one for `pr`."""
    ones = np.ones(small.n_samples)
    model = PoissonRegression().fit(small.X, small.counts, offset=ones)
    reference = fit_in_r(small.X, small.counts, "pr", add_intercept=True, time=ones)
    assert_matches(
        model,
        reference,
        coefficients=np.concatenate(([model.intercept_], model.coef_)),
    )

    # ...and it must agree with omitting the offset entirely.
    plain = PoissonRegression().fit(small.X, small.counts)
    np.testing.assert_allclose(plain.coef_, model.coef_, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(
        plain.intercept_, model.intercept_, rtol=1e-9, atol=1e-9
    )


def test_least_squares(small):
    model = LinearRegression().fit(small.X, small.continuous)
    reference = fit_in_r(small.X, small.continuous, "ls", add_intercept=True)
    assert_matches(
        model,
        reference,
        coefficients=np.concatenate(([model.intercept_], model.coef_)),
    )


# ---------------------------------------------------------------------------
# Stratified models
# ---------------------------------------------------------------------------


def test_conditional_logistic(small):
    model = ConditionalLogisticRegression().fit(
        small.X, small.y, strata=small.strata
    )
    reference = fit_in_r(small.X, small.y, "clr", strata=small.strata)
    assert_matches(model, reference, coefficients=model.coef_)
    assert model.data_.n_strata == reference.n_strata


def test_conditional_logistic_regularized(small):
    model = ConditionalLogisticRegression(
        prior="laplace", prior_variance=0.5
    ).fit(small.X, small.y, strata=small.strata)
    reference = fit_in_r(
        small.X,
        small.y,
        "clr",
        strata=small.strata,
        prior="laplace",
        prior_variance=0.5,
    )
    assert_matches(model, reference, coefficients=model.coef_)


def test_conditional_poisson(small):
    model = ConditionalPoissonRegression().fit(
        small.X, small.counts, strata=small.strata, offset=small.offset
    )
    reference = fit_in_r(
        small.X, small.counts, "cpr", strata=small.strata, time=small.offset
    )
    assert_matches(model, reference, coefficients=model.coef_)


def test_self_controlled_case_series(small):
    """SCCS consumes the offset as its `time` vector, not as a design column."""
    model = SelfControlledCaseSeries().fit(
        small.X, small.counts, strata=small.strata, offset=small.offset
    )
    reference = fit_in_r(
        small.X,
        small.counts,
        "sccs",
        strata=small.strata,
        time=small.offset,
    )
    assert_matches(model, reference, coefficients=model.coef_)


def test_strata_out_of_order(small):
    """Unsorted strata must be permuted internally, not silently mis-grouped."""
    rng = np.random.default_rng(3)
    shuffle = rng.permutation(small.n_samples)
    X, y, strata = small.X[shuffle], small.y[shuffle], small.strata[shuffle]

    model = ConditionalLogisticRegression().fit(X, y, strata=strata)
    reference = fit_in_r(X, y, "clr", strata=strata)  # R sorts too
    assert_matches(model, reference, coefficients=model.coef_)

    sorted_model = ConditionalLogisticRegression().fit(
        small.X, small.y, strata=small.strata
    )
    np.testing.assert_allclose(model.coef_, sorted_model.coef_, rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# Survival models
# ---------------------------------------------------------------------------


def test_cox(small):
    event = small.y
    model = CoxRegression().fit(small.X, event, time=small.time)
    reference = fit_in_r(small.X, event, "cox", time=small.time)
    assert_matches(model, reference, coefficients=model.coef_)


def test_cox_regularized(small):
    event = small.y
    model = CoxRegression(prior="laplace", prior_variance=0.2).fit(
        small.X, event, time=small.time
    )
    reference = fit_in_r(
        small.X, event, "cox", time=small.time, prior="laplace", prior_variance=0.2
    )
    assert_matches(model, reference, coefficients=model.coef_)


def test_cox_stratified(small):
    event = small.y
    model = CoxRegression().fit(
        small.X, event, time=small.time, strata=small.strata
    )
    reference = fit_in_r(
        small.X, event, "cox", time=small.time, strata=small.strata
    )
    assert_matches(model, reference, coefficients=model.coef_)


def test_cox_time_out_of_order(small):
    """Cox needs rows in descending-time order; the binding must impose it."""
    rng = np.random.default_rng(11)
    shuffle = rng.permutation(small.n_samples)
    model = CoxRegression().fit(
        small.X[shuffle], small.y[shuffle], time=small.time[shuffle]
    )
    ordered = CoxRegression().fit(small.X, small.y, time=small.time)
    np.testing.assert_allclose(model.coef_, ordered.coef_, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        model.log_likelihood_, ordered.log_likelihood_, rtol=1e-12, atol=1e-12
    )


# ---------------------------------------------------------------------------
# Sparse designs and cross-validation
# ---------------------------------------------------------------------------


def test_sparse_indicator_design(sparse_binary):
    X, y, _ = sparse_binary
    model = LogisticRegression(prior="laplace", prior_variance=0.1).fit(X, y)
    reference = fit_in_r(
        X, y, "lr", add_intercept=True, prior="laplace", prior_variance=0.1
    )
    assert_matches(
        model,
        reference,
        coefficients=np.concatenate(([model.intercept_], model.coef_)),
        # 300 penalized coefficients accumulate more floating-point noise than
        # the 4-covariate cases.
        coef_tol=1e-7,
        ll_tol=1e-9,
    )


def test_sparse_and_dense_agree(small):
    """The same numbers stored sparsely must give the same fit."""
    import scipy.sparse as sp

    dense = LogisticRegression(prior="laplace", prior_variance=0.3).fit(
        small.X, small.y
    )
    sparse = LogisticRegression(prior="laplace", prior_variance=0.3).fit(
        sp.csc_matrix(small.X), small.y
    )
    np.testing.assert_allclose(dense.coef_, sparse.coef_, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        dense.log_likelihood_, sparse.log_likelihood_, rtol=1e-12, atol=1e-12
    )


@pytest.mark.slow
def test_cross_validated_variance(medium):
    """A cross-validated search must land on the same hyperparameter.

    Fold assignment is seeded, so a shared seed makes this deterministic across
    both front-ends.
    """
    seed = 4242
    data = CyclopsData.from_arrays(
        medium.X, medium.y, "lr", add_intercept=True
    )
    model = CyclopsModel(
        data,
        prior=Prior(kind="laplace", variance="cv"),
        control=Control(seed=seed, fold=10, cv_repetitions=1, selector="byrow"),
    )
    result = model.fit()

    reference = fit_in_r(
        medium.X,
        medium.y,
        "lr",
        add_intercept=True,
        prior="laplace",
        use_cross_validation=True,
        seed=seed,
        fold=10,
        cv_repetitions=1,
        selector="byRow",
    )

    np.testing.assert_allclose(
        result.variance, reference.variance, rtol=1e-6, atol=1e-8
    )
    assert_matches(
        result, reference, coefficients=result.coefficients, coef_tol=1e-7, ll_tol=1e-9
    )


# ---------------------------------------------------------------------------
# Non-convergent and degenerate cases
# ---------------------------------------------------------------------------


def test_separable_data_reports_same_flag():
    """A separable design has no finite MLE; both sides must say so identically."""
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

    model = LogisticRegression(max_iter=100).fit(X, y)
    reference = fit_in_r(X, y, "lr", add_intercept=True, max_iterations=100)

    assert model.return_flag_ == reference.return_flag
    assert not model.converged_
    np.testing.assert_allclose(
        np.concatenate(([model.intercept_], model.coef_)),
        reference.coefficients,
        rtol=1e-7,
        atol=1e-7,
    )


def test_max_iterations_respected(small):
    model = LogisticRegression(max_iter=2).fit(small.X, small.y)
    reference = fit_in_r(small.X, small.y, "lr", add_intercept=True, max_iterations=2)
    assert model.return_flag_ == reference.return_flag == "MAX_ITERATIONS"
    assert_matches(
        model,
        reference,
        coefficients=np.concatenate(([model.intercept_], model.coef_)),
    )


@pytest.mark.parametrize("convergence", ["lange", "zhang"])
def test_alternative_convergence_criteria(small, convergence):
    model = LogisticRegression(control=Control(convergence=convergence)).fit(
        small.X, small.y
    )
    reference = fit_in_r(
        small.X,
        small.y,
        "lr",
        add_intercept=True,
        convergence=convergence,
    )
    assert_matches(
        model,
        reference,
        coefficients=np.concatenate(([model.intercept_], model.coef_)),
    )
