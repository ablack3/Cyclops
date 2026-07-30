"""Tests for :mod:`cyclops.estimators` — the scikit-learn-shaped surface."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from cyclops import (
    ConditionalLogisticRegression,
    ConditionalPoissonRegression,
    CoxRegression,
    FineGrayRegression,
    LinearRegression,
    LogisticRegression,
    NotFittedError,
    PoissonRegression,
    SelfControlledCaseSeries,
)


# ---------------------------------------------------------------------------
# Shape and protocol
# ---------------------------------------------------------------------------


def test_fit_returns_self_and_sets_attributes(small):
    model = LogisticRegression()
    assert model.fit(small.X, small.y) is model

    assert model.coef_.shape == (small.n_features,)
    assert isinstance(model.intercept_, float)
    assert model.n_features_in_ == small.n_features
    assert model.n_iter_ > 0
    assert model.converged_
    assert model.return_flag_ == "SUCCESS"
    assert np.isfinite(model.log_likelihood_)
    np.testing.assert_array_equal(model.classes_, [0, 1])


def test_attributes_before_fit_raise(small):
    model = LogisticRegression()
    with pytest.raises(NotFittedError):
        model.decision_function(small.X)
    with pytest.raises(NotFittedError):
        model.standard_errors()


def test_get_and_set_params_round_trip():
    model = LogisticRegression(prior="laplace", prior_variance=0.25, max_iter=42)
    params = model.get_params()
    assert params["prior"] == "laplace"
    assert params["prior_variance"] == 0.25
    assert params["max_iter"] == 42

    clone = LogisticRegression(**params)
    assert clone.get_params() == params

    model.set_params(max_iter=7)
    assert model.max_iter == 7
    with pytest.raises(ValueError, match="Invalid parameter"):
        model.set_params(nonsense=1)


def test_clone_via_sklearn(small):
    sklearn = pytest.importorskip("sklearn")
    from sklearn.base import clone

    model = LogisticRegression(prior="normal", prior_variance=2.0)
    copy = clone(model)
    assert copy.get_params() == model.get_params()
    assert not hasattr(copy, "coef_")


def test_repr_lists_parameters():
    text = repr(LogisticRegression(prior="laplace", prior_variance=0.1))
    assert "LogisticRegression(" in text and "prior='laplace'" in text


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def test_predict_proba_sums_to_one(small):
    model = LogisticRegression().fit(small.X, small.y)
    probabilities = model.predict_proba(small.X)
    assert probabilities.shape == (small.n_samples, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_predict_matches_probability_threshold(small):
    model = LogisticRegression().fit(small.X, small.y)
    expected = (model.predict_proba(small.X)[:, 1] > 0.5).astype(np.int64)
    np.testing.assert_array_equal(model.predict(small.X), expected)


def test_predict_proba_agrees_with_engine_on_training_rows(small):
    """The Python link must reproduce the C++ `predictEstimate` exactly."""
    model = LogisticRegression().fit(small.X, small.y)
    engine = model.model_.predict()
    ours = model.predict_proba(small.X)[:, 1]
    np.testing.assert_allclose(ours, engine, rtol=1e-12, atol=1e-14)


def test_poisson_predict_applies_offset(small):
    model = PoissonRegression().fit(small.X, small.counts, offset=small.offset)
    without = model.predict(small.X)
    with_offset = model.predict(small.X, offset=small.offset)
    np.testing.assert_allclose(with_offset, without * small.offset, rtol=1e-12)


def test_poisson_predict_agrees_with_engine(small):
    model = PoissonRegression().fit(small.X, small.counts, offset=small.offset)
    np.testing.assert_allclose(
        model.predict(small.X, offset=small.offset),
        model.model_.predict(),
        rtol=1e-10,
        atol=1e-12,
    )


def test_cox_predict_is_the_linear_predictor(small):
    model = CoxRegression().fit(small.X, small.y, time=small.time)
    np.testing.assert_allclose(
        model.predict(small.X), small.X @ model.coef_, rtol=1e-12
    )
    np.testing.assert_allclose(model.hazard_ratios(), np.exp(model.coef_))


def test_predict_rejects_wrong_feature_count(small):
    model = LogisticRegression().fit(small.X, small.y)
    with pytest.raises(ValueError, match="features"):
        model.predict(small.X[:, :-1])


def test_predict_accepts_sparse_input(small):
    model = LogisticRegression().fit(small.X, small.y)
    dense = model.predict_proba(small.X)
    sparse = model.predict_proba(sp.csr_matrix(small.X))
    np.testing.assert_allclose(dense, sparse, rtol=1e-12)


# ---------------------------------------------------------------------------
# Regularization
# ---------------------------------------------------------------------------


def test_laplace_shrinks_more_than_no_prior(small):
    plain = LogisticRegression().fit(small.X, small.y)
    penalized = LogisticRegression(prior="laplace", prior_variance=0.01).fit(
        small.X, small.y
    )
    assert np.abs(penalized.coef_).sum() < np.abs(plain.coef_).sum()


def test_laplace_produces_exact_zeros(sparse_binary):
    X, y, _ = sparse_binary
    model = LogisticRegression(prior="laplace", prior_variance=0.005).fit(X, y)
    assert np.count_nonzero(model.coef_ == 0.0) > 0


def test_l1_and_l2_aliases(small):
    laplace = LogisticRegression(prior="laplace", prior_variance=0.1).fit(
        small.X, small.y
    )
    l1 = LogisticRegression(prior="l1", prior_variance=0.1).fit(small.X, small.y)
    np.testing.assert_allclose(laplace.coef_, l1.coef_)

    normal = LogisticRegression(prior="normal", prior_variance=0.1).fit(
        small.X, small.y
    )
    l2 = LogisticRegression(prior="l2", prior_variance=0.1).fit(small.X, small.y)
    np.testing.assert_allclose(normal.coef_, l2.coef_)


def test_intercept_is_unpenalized_by_default(small):
    model = LogisticRegression(prior="laplace", prior_variance=1e-4).fit(
        small.X, small.y
    )
    # With shrinkage this aggressive every penalized coefficient collapses; a
    # surviving intercept is the observable consequence of excluding it.
    assert model.intercept_ != 0.0
    assert not model.model_.is_regularized()[0]


def test_force_intercept_penalizes_it(small):
    model = LogisticRegression(
        prior="laplace", prior_variance=1e-4, force_intercept=True
    ).fit(small.X, small.y)
    assert model.model_.is_regularized()[0]


def test_excluded_columns_are_unpenalized(small):
    model = LogisticRegression(
        prior="laplace", prior_variance=1e-4, exclude=[1]
    ).fit(small.X, small.y)
    regularized = model.model_.is_regularized()
    # is_regularized() is in stored column order: intercept first, then features.
    assert not regularized[0]        # intercept
    assert not regularized[1 + 1]    # excluded feature
    assert regularized[1 + 0]        # a penalized neighbour


def test_exclude_out_of_range_raises(small):
    with pytest.raises(ValueError, match="outside"):
        LogisticRegression(prior="laplace", exclude=[99]).fit(small.X, small.y)


def test_unknown_prior_raises(small):
    with pytest.raises(ValueError, match="Unknown prior"):
        LogisticRegression(prior="elasticnet").fit(small.X, small.y)


@pytest.mark.slow
def test_cross_validated_variance_is_selected(medium):
    model = LogisticRegression(
        prior="laplace", prior_variance="cv", random_state=1
    ).fit(medium.X, medium.y)
    assert model.prior_variance_ is not None
    assert model.prior_variance_ > 0
    assert model.cross_validation_info_ != ""


def test_cross_validation_without_prior_raises(small):
    with pytest.raises(Exception, match="requires a regularising prior"):
        LogisticRegression(prior="none", prior_variance="cv").fit(small.X, small.y)


# ---------------------------------------------------------------------------
# Intercept handling per model family
# ---------------------------------------------------------------------------


def test_fit_intercept_false(small):
    model = LogisticRegression(fit_intercept=False).fit(small.X, small.y)
    assert model.intercept_ == 0.0
    assert model.coef_.shape == (small.n_features,)
    assert not model.data_.has_intercept


@pytest.mark.parametrize(
    "estimator,kwargs",
    [
        (ConditionalLogisticRegression, {"strata": "strata"}),
        (CoxRegression, {"time": "time"}),
    ],
)
def test_models_without_intercept_reject_one(small, estimator, kwargs):
    resolved = {key: getattr(small, value) for key, value in kwargs.items()}
    with pytest.raises(ValueError, match="cannot include an intercept"):
        estimator(fit_intercept=True).fit(small.X, small.y, **resolved)


def test_conditional_models_report_zero_intercept(small):
    model = ConditionalLogisticRegression().fit(
        small.X, small.y, strata=small.strata
    )
    assert model.intercept_ == 0.0
    assert model.coef_.shape == (small.n_features,)


# ---------------------------------------------------------------------------
# Required arguments
# ---------------------------------------------------------------------------


def test_conditional_logistic_requires_strata(small):
    with pytest.raises(ValueError, match="requires `strata`"):
        ConditionalLogisticRegression().fit(small.X, small.y)


def test_conditional_poisson_requires_strata(small):
    with pytest.raises(ValueError, match="requires `strata`"):
        ConditionalPoissonRegression().fit(small.X, small.counts)


def test_sccs_requires_strata_and_offset(small):
    with pytest.raises(ValueError, match="requires `strata`"):
        SelfControlledCaseSeries().fit(small.X, small.counts, offset=small.offset)
    with pytest.raises(ValueError, match="requires `offset`"):
        SelfControlledCaseSeries().fit(small.X, small.counts, strata=small.strata)


def test_cox_requires_time(small):
    with pytest.raises(ValueError, match="requires `time`"):
        CoxRegression().fit(small.X, small.y)


def test_fine_gray_requires_censor_weights(small):
    with pytest.raises(ValueError, match="requires `censor_weights`"):
        FineGrayRegression().fit(small.X, small.y, time=small.time)


def test_fine_gray_fits_with_weights(small):
    weights = np.clip(np.linspace(0.2, 1.0, small.n_samples), 0.0, 1.0)
    model = FineGrayRegression().fit(
        small.X, small.y, time=small.time, censor_weights=weights
    )
    assert model.coef_.shape == (small.n_features,)
    assert np.all(np.isfinite(model.coef_))


# ---------------------------------------------------------------------------
# Outcome validation
# ---------------------------------------------------------------------------


def test_logistic_rejects_non_binary_outcome(small):
    with pytest.raises(ValueError, match="only 0 and 1"):
        LogisticRegression().fit(small.X, small.counts)


def test_poisson_rejects_negative_outcome(small):
    with pytest.raises(ValueError, match="non-negative"):
        PoissonRegression().fit(small.X, -small.counts)


def test_cox_rejects_non_binary_event(small):
    with pytest.raises(ValueError, match="0/1 event indicator"):
        CoxRegression().fit(small.X, small.counts, time=small.time)


def test_two_dimensional_y_raises(small):
    with pytest.raises(ValueError, match="one-dimensional"):
        LogisticRegression().fit(small.X, small.y.reshape(-1, 1))


# ---------------------------------------------------------------------------
# Weights, warm starts, convergence
# ---------------------------------------------------------------------------


def test_zero_weight_rows_are_ignored(small):
    """Zero-weighting rows must equal deleting them."""
    keep = np.ones(small.n_samples, dtype=bool)
    keep[::7] = False
    weights = keep.astype(np.float64)

    weighted = LogisticRegression().fit(small.X, small.y, sample_weight=weights)
    subset = LogisticRegression().fit(small.X[keep], small.y[keep])
    np.testing.assert_allclose(weighted.coef_, subset.coef_, rtol=1e-6, atol=1e-8)


def test_duplicated_rows_equal_integer_weights(small):
    weights = np.ones(small.n_samples)
    weights[:20] = 2.0
    weighted = LogisticRegression().fit(small.X, small.y, sample_weight=weights)

    X = np.vstack([small.X, small.X[:20]])
    y = np.concatenate([small.y, small.y[:20]])
    duplicated = LogisticRegression().fit(X, y)
    np.testing.assert_allclose(weighted.coef_, duplicated.coef_, rtol=1e-6, atol=1e-8)


def test_weights_length_is_checked(small):
    with pytest.raises(ValueError, match="one entry per sample"):
        LogisticRegression().fit(
            small.X, small.y, sample_weight=np.ones(small.n_samples - 1)
        )


def test_start_values_reach_the_same_mode(small):
    """A different warm start must not change a well-conditioned optimum."""
    default = LogisticRegression(tol=1e-10).fit(small.X, small.y)
    warm = LogisticRegression(tol=1e-10).fit(
        small.X, small.y, start_values=default.coef_ * 0.5
    )
    np.testing.assert_allclose(warm.coef_, default.coef_, rtol=1e-5, atol=1e-7)


def test_start_values_length_is_checked(small):
    with pytest.raises(ValueError, match="start_values must have"):
        LogisticRegression().fit(small.X, small.y, start_values=np.zeros(2))


def test_max_iter_one_reports_non_convergence(small):
    model = LogisticRegression(max_iter=1).fit(small.X, small.y)
    assert model.return_flag_ == "MAX_ITERATIONS"
    assert not model.converged_
    assert model.n_iter_ == 1


def test_tighter_tolerance_uses_more_iterations(small):
    loose = LogisticRegression(tol=1e-2).fit(small.X, small.y)
    tight = LogisticRegression(tol=1e-12).fit(small.X, small.y)
    assert tight.n_iter_ >= loose.n_iter_


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def test_standard_errors_shape_and_sign(small):
    model = LogisticRegression().fit(small.X, small.y)
    errors = model.standard_errors()
    assert errors.shape == (small.n_features + 1,)  # intercept included
    assert np.all(errors > 0)


def test_confidence_intervals_bracket_the_estimate(small):
    model = LogisticRegression().fit(small.X, small.y)
    intervals = model.confidence_intervals([0, 2])
    assert len(intervals) == 2
    for interval, column in zip(intervals, [0, 2]):
        assert interval.lower < model.coef_[column] < interval.upper
        assert interval.evaluations > 0


def test_gradient_is_near_zero_at_the_mode(small):
    model = LogisticRegression(tol=1e-12).fit(small.X, small.y)
    assert np.max(np.abs(model.model_.gradient())) < 1e-4


def test_profile_likelihood_peaks_at_the_estimate(small):
    model = LogisticRegression().fit(small.X, small.y)
    estimate = model.coef_[0]
    points = estimate + np.array([-0.5, -0.1, 0.0, 0.1, 0.5])
    curve = model.model_.profile_likelihood(
        int(model.covariate_ids_[0]), points
    )
    assert curve.values.shape == points.shape
    assert np.argmax(curve.values) == 2


# ---------------------------------------------------------------------------
# Sparse designs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["csc", "csr", "coo", "lil"])
def test_sparse_formats_agree(small, fmt):
    dense = LogisticRegression(prior="laplace", prior_variance=0.2).fit(
        small.X, small.y
    )
    sparse = LogisticRegression(prior="laplace", prior_variance=0.2).fit(
        sp.csc_matrix(small.X).asformat(fmt), small.y
    )
    np.testing.assert_allclose(dense.coef_, sparse.coef_, rtol=1e-9, atol=1e-11)


def test_wide_sparse_design_recovers_signal(sparse_binary):
    X, y, signal = sparse_binary
    model = LogisticRegression(prior="laplace", prior_variance=0.5).fit(X, y)
    assert model.coef_.shape == (X.shape[1],)
    # The 10 true signals should dominate the fitted coefficients.
    top = np.argsort(np.abs(model.coef_))[::-1][:20]
    assert len(set(top) & set(np.flatnonzero(signal))) >= 4


def test_single_feature_and_reshaped_input(small):
    column = small.X[:, 0]
    two_dimensional = LogisticRegression().fit(column.reshape(-1, 1), small.y)
    one_dimensional = LogisticRegression().fit(column, small.y)
    np.testing.assert_allclose(one_dimensional.coef_, two_dimensional.coef_)


def test_linear_regression_recovers_least_squares(small):
    model = LinearRegression().fit(small.X, small.continuous)
    design = np.column_stack([np.ones(small.n_samples), small.X])
    expected, *_ = np.linalg.lstsq(design, small.continuous, rcond=None)
    np.testing.assert_allclose(
        np.concatenate(([model.intercept_], model.coef_)), expected, rtol=1e-4, atol=1e-5
    )
