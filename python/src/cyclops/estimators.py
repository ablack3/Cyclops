"""scikit-learn-shaped estimators.

Naming follows scikit-learn where a convention exists (``fit``/``predict``,
``coef_``, ``intercept_``, ``n_iter_``) and Cyclops where none does (``strata``,
``prior_variance="cv"``, ``log_likelihood_``). These are *not* registered
scikit-learn estimators — they do not subclass ``BaseEstimator`` — but they
implement enough of the protocol for ``clone``, ``GridSearchCV`` and pipelines to
work; see ``docs/DESIGN_DECISIONS.md`` §6.
"""

from __future__ import annotations

import inspect
from typing import ClassVar, Sequence

import numpy as np
import scipy.sparse as sp

from cyclops.data import INTERCEPT_ID, CyclopsData
from cyclops.model import Control, CyclopsModel, Prior

__all__ = [
    "LogisticRegression",
    "PoissonRegression",
    "LinearRegression",
    "ConditionalLogisticRegression",
    "ConditionalPoissonRegression",
    "SelfControlledCaseSeries",
    "CoxRegression",
    "FineGrayRegression",
]


class NotFittedError(RuntimeError):
    """Raised when a fitted attribute is read before :meth:`fit`."""


class BaseCyclopsEstimator:
    """Shared plumbing: parameter handling, data assembly, linear predictor.

    Args:
        prior: Penalty family — ``"none"``, ``"laplace"``/``"l1"``,
            ``"normal"``/``"l2"``, ``"barupdate"``, ``"jeffreys"``.
        prior_variance: Prior variance, or ``"cv"`` to cross-validate it.
            Cyclops parameterizes by variance: larger means weaker shrinkage.
        exclude: 0-based column indices of ``X`` to leave unpenalized.
        force_intercept: Penalize the intercept too (off by default, as in R).
        fit_intercept: Include an intercept. ``None`` selects the only value the
            model supports.
        max_iter: Maximum coordinate-descent iterations.
        tol: Convergence tolerance.
        n_threads: Worker threads for cross-validation (``-1`` for all cores).
        random_state: Seed for cross-validation fold assignment.
        verbose: Emit Cyclops progress messages (readable via
            :attr:`fit_log_`).
        control: A :class:`~cyclops.model.Control` whose fields override every
            control-related argument above. Escape hatch for the knobs the
            estimator does not surface.
    """

    #: Cyclops model-type code.
    _model_type: ClassVar[str]
    #: False for the models that absorb the intercept into strata or a baseline hazard.
    _allows_intercept: ClassVar[bool] = True

    def __init__(
        self,
        *,
        prior: str = "none",
        prior_variance: float | str = 1.0,
        exclude: Sequence[int] | None = None,
        force_intercept: bool = False,
        fit_intercept: bool | None = None,
        max_iter: int = 1000,
        tol: float = 1e-6,
        n_threads: int = 1,
        random_state: int | None = None,
        verbose: bool = False,
        control: Control | None = None,
    ):
        self.prior = prior
        self.prior_variance = prior_variance
        self.exclude = exclude
        self.force_intercept = force_intercept
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.n_threads = n_threads
        self.random_state = random_state
        self.verbose = verbose
        self.control = control

    # -- scikit-learn parameter protocol ------------------------------------

    @classmethod
    def _param_names(cls) -> list[str]:
        signature = inspect.signature(cls.__init__)
        return sorted(
            name
            for name, parameter in signature.parameters.items()
            if name != "self" and parameter.kind != parameter.VAR_KEYWORD
        )

    def get_params(self, deep: bool = True) -> dict:
        return {name: getattr(self, name) for name in self._param_names()}

    def set_params(self, **params) -> "BaseCyclopsEstimator":
        valid = self._param_names()
        for name, value in params.items():
            if name not in valid:
                raise ValueError(
                    f"Invalid parameter {name!r} for {type(self).__name__}; "
                    f"valid parameters are: {', '.join(valid)}"
                )
            setattr(self, name, value)
        return self

    # -- fitting ------------------------------------------------------------

    def fit(
        self,
        X,
        y,
        *,
        sample_weight=None,
        strata=None,
        time=None,
        offset=None,
        covariate_ids: Sequence[int] | None = None,
        start_values=None,
        censor_weights=None,
    ) -> "BaseCyclopsEstimator":
        """Fit the model.

        Args:
            X: ``(n_samples, n_features)`` dense array or SciPy sparse matrix.
                Sparse input is used as-is; no densification happens anywhere.
            y: Outcome, ``(n_samples,)``. See each subclass for its scale.
            sample_weight: Non-negative per-row weights; zero drops a row.
            strata: Stratum labels, required by the conditional models and
                optional for the survival ones.
            time: Survival or exposure time.
            offset: Poisson/SCCS exposure offset on the natural scale.
            covariate_ids: Stable int64 identifier per column. Defaults to
                ``1..n_features``. Use OHDSI covariate ids here to keep
                ``coef_`` aligned across cohorts with different column subsets.
            start_values: Starting coefficients, one per column of ``X``.
            censor_weights: Fine-Gray subject-specific censoring weights.
        """
        y = np.ascontiguousarray(y, dtype=np.float64)
        self._validate_outcome(y)

        fit_intercept = self._resolve_fit_intercept()

        # Resolve the feature ids here rather than reading them back off the
        # stored columns: those also carry the intercept (id 0) and any promoted
        # offset (id -1), so `exclude` and `standard_errors` would silently
        # address the wrong column.
        n_features = 1 if np.ndim(X) == 1 else X.shape[1]
        if covariate_ids is None:
            feature_ids = np.arange(1, n_features + 1, dtype=np.int64)
        else:
            feature_ids = np.ascontiguousarray(covariate_ids, dtype=np.int64)
            if feature_ids.shape != (n_features,):
                raise ValueError(
                    f"covariate_ids must have {n_features} elements, "
                    f"got {feature_ids.shape[0]}"
                )

        data = CyclopsData.from_arrays(
            X,
            y,
            self._model_type,
            time=time,
            strata=strata,
            offset=offset,
            covariate_ids=feature_ids,
            add_intercept=fit_intercept,
            silent=not self.verbose,
        )

        self.n_features_in_ = n_features
        self.fit_intercept_ = fit_intercept

        model = CyclopsModel(
            data,
            prior=self._build_prior(feature_ids),
            control=self._build_control(),
        )

        order = data.row_order
        if sample_weight is not None:
            weights = np.ascontiguousarray(sample_weight, dtype=np.float64)
            if weights.shape != y.shape:
                raise ValueError("sample_weight must have one entry per sample")
            model.set_weights(weights if order is None else weights[order])

        if censor_weights is not None:
            weights = np.ascontiguousarray(censor_weights, dtype=np.float64)
            if weights.shape != y.shape:
                raise ValueError("censor_weights must have one entry per sample")
            model.set_censor_weights(weights if order is None else weights[order])

        if start_values is not None:
            start = np.ascontiguousarray(start_values, dtype=np.float64)
            if start.shape != (n_features,):
                raise ValueError(
                    f"start_values must have {n_features} entries, got {start.shape[0]}"
                )
            if fit_intercept:
                start = np.concatenate(([0.0], start))
            model.set_start_values(start)

        result = model.fit()

        self._data = data
        self._model = model
        self._result = result

        ids = np.asarray(result.covariate_ids)
        coefficients = np.asarray(result.coefficients, dtype=np.float64)
        if fit_intercept:
            # add_intercept() puts the intercept first (after any offset).
            intercept_position = int(np.flatnonzero(ids == INTERCEPT_ID)[0])
            self.intercept_ = float(coefficients[intercept_position])
            keep = np.ones(coefficients.shape[0], dtype=bool)
            keep[intercept_position] = False
            self.coef_ = coefficients[keep]
            self.covariate_ids_ = ids[keep]
        else:
            self.intercept_ = 0.0
            self.coef_ = coefficients
            self.covariate_ids_ = ids

        self.n_iter_ = result.iterations
        self.log_likelihood_ = result.log_likelihood
        self.log_prior_ = result.log_prior
        self.return_flag_ = result.return_flag
        self.converged_ = result.converged
        self.prior_variance_ = result.variance[0] if result.variance else None
        self.cross_validation_info_ = result.cross_validation_info
        self.prior_info_ = result.prior_info
        self.fit_seconds_ = result.fit_seconds
        self.fit_log_ = model.take_log()
        return self

    # -- inference ----------------------------------------------------------

    def decision_function(self, X, *, offset=None) -> np.ndarray:
        """Linear predictor ``X @ coef_ + intercept_`` (plus ``log(offset)``).

        Unlike ``predict.cyclopsFit``, which can only score the rows it was
        fitted on, this evaluates any design matrix with the same column layout.
        """
        self._check_fitted()
        matrix = self._check_features(X)
        eta = matrix @ self.coef_ + self.intercept_
        eta = np.asarray(eta, dtype=np.float64).ravel()
        if offset is not None:
            offset = np.ascontiguousarray(offset, dtype=np.float64)
            if offset.shape != eta.shape:
                raise ValueError("offset must have one entry per sample")
            eta = eta + np.log(offset)
        return eta

    def standard_errors(self) -> np.ndarray:
        """Asymptotic standard errors, aligned with ``[intercept_, *coef_]``.

        Valid only for an unregularized fit; a penalized fit's information matrix
        does not describe the sampling distribution of the estimator.
        """
        self._check_fitted()
        # Name the columns explicitly: passing no ids would include the offset
        # column, whose coefficient is fixed and has no standard error.
        ids = list(self.covariate_ids_)
        if self.fit_intercept_:
            ids.insert(0, INTERCEPT_ID)
        return self._model.standard_errors(ids)

    def confidence_intervals(self, columns: Sequence[int] | None = None, **kwargs):
        """Likelihood-profile confidence intervals for the given columns of ``X``."""
        self._check_fitted()
        ids = (
            None
            if columns is None
            else [int(self.covariate_ids_[c]) for c in columns]
        )
        return self._model.confidence_intervals(ids, **kwargs)

    @property
    def model_(self) -> CyclopsModel:
        """The underlying :class:`~cyclops.model.CyclopsModel`."""
        self._check_fitted()
        return self._model

    @property
    def data_(self) -> CyclopsData:
        """The underlying :class:`~cyclops.data.CyclopsData`."""
        self._check_fitted()
        return self._data

    # -- internals ----------------------------------------------------------

    def _validate_outcome(self, y: np.ndarray) -> None:
        if y.ndim != 1:
            raise ValueError(f"y must be one-dimensional, got shape {y.shape}")

    def _resolve_fit_intercept(self) -> bool:
        if self.fit_intercept is None:
            return self._allows_intercept
        if self.fit_intercept and not self._allows_intercept:
            raise ValueError(
                f"{type(self).__name__} cannot include an intercept: it is absorbed "
                "by the strata or the baseline hazard"
            )
        return bool(self.fit_intercept)

    def _build_prior(self, feature_ids: np.ndarray) -> Prior:
        exclude_ids: list[int] = []
        if self.exclude is not None:
            n_features = feature_ids.shape[0]
            for column in self.exclude:
                index = int(column)
                if not 0 <= index < n_features:
                    raise ValueError(
                        f"exclude contains column {index}, outside "
                        f"[0, {n_features})"
                    )
                exclude_ids.append(int(feature_ids[index]))

        return Prior(
            kind=self.prior,
            variance=self.prior_variance,
            exclude=exclude_ids or None,
            force_intercept=self.force_intercept,
        )

    def _build_control(self) -> Control:
        if self.control is not None:
            return self.control
        return Control(
            max_iterations=self.max_iter,
            tolerance=self.tol,
            threads=self.n_threads,
            seed=self.random_state,
            noise="noisy" if self.verbose else "silent",
        )

    def _check_fitted(self) -> None:
        if not hasattr(self, "coef_"):
            raise NotFittedError(
                f"This {type(self).__name__} is not fitted yet; call fit() first"
            )

    def _check_features(self, X):
        matrix = X.tocsr(copy=False) if sp.issparse(X) else np.asarray(X)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {matrix.shape[1]} features, but this "
                f"{type(self).__name__} was fitted with {self.n_features_in_}"
            )
        return matrix

    def __repr__(self) -> str:
        params = ", ".join(
            f"{name}={value!r}"
            for name, value in sorted(self.get_params().items())
            if value is not None
        )
        return f"{type(self).__name__}({params})"


# ---------------------------------------------------------------------------
# Binary-outcome models
# ---------------------------------------------------------------------------


class _BinaryOutcomeMixin:
    """Shared 0/1 outcome validation and label prediction."""

    def _validate_outcome(self, y: np.ndarray) -> None:
        super()._validate_outcome(y)
        unique = np.unique(y)
        if not np.all(np.isin(unique, (0.0, 1.0))):
            raise ValueError(
                f"y must contain only 0 and 1 for {type(self).__name__}; "
                f"found {unique[:5]}"
            )
        self.classes_ = np.array([0, 1])


class LogisticRegression(_BinaryOutcomeMixin, BaseCyclopsEstimator):
    """Unconditional logistic regression (Cyclops ``"lr"``).

    ``y`` must be 0/1.

    Example:
        >>> model = LogisticRegression(prior="laplace", prior_variance="cv")
        >>> model.fit(X, y)                         # doctest: +SKIP
        >>> model.predict_proba(X)[:, 1]            # doctest: +SKIP
    """

    _model_type = "lr"

    def predict_proba(self, X) -> np.ndarray:
        """``(n_samples, 2)`` array of class probabilities."""
        positive = 1.0 / (1.0 + np.exp(-self.decision_function(X)))
        return np.column_stack((1.0 - positive, positive))

    def predict_log_proba(self, X) -> np.ndarray:
        return np.log(self.predict_proba(X))

    def predict(self, X) -> np.ndarray:
        """Predicted class labels, thresholded at 0.5."""
        return (self.decision_function(X) > 0.0).astype(np.int64)


class ConditionalLogisticRegression(_BinaryOutcomeMixin, BaseCyclopsEstimator):
    """Conditional (stratified) logistic regression (Cyclops ``"clr"``).

    Requires ``strata=``; has no intercept, since it cancels in the conditional
    likelihood. ``decision_function`` gives the linear predictor, from which only
    within-stratum comparisons are meaningful.
    """

    _model_type = "clr"
    _allows_intercept = False

    def fit(self, X, y, *, strata=None, **kwargs):
        if strata is None:
            raise ValueError(
                f"{type(self).__name__} requires `strata`; use LogisticRegression "
                "for unstratified data"
            )
        return super().fit(X, y, strata=strata, **kwargs)


class ExactConditionalLogisticRegression(ConditionalLogisticRegression):
    """Conditional logistic regression with exact tie handling (``"clr_exact"``)."""

    _model_type = "clr_exact"


class EfronConditionalLogisticRegression(ConditionalLogisticRegression):
    """Conditional logistic regression with Efron tie handling (``"clr_efron"``)."""

    _model_type = "clr_efron"


# ---------------------------------------------------------------------------
# Count / continuous models
# ---------------------------------------------------------------------------


class PoissonRegression(BaseCyclopsEstimator):
    """Poisson regression (Cyclops ``"pr"``).

    ``y`` holds non-negative counts. Pass person-time as ``offset=`` on the
    natural scale; Cyclops logs it internally.
    """

    _model_type = "pr"

    def _validate_outcome(self, y: np.ndarray) -> None:
        super()._validate_outcome(y)
        if np.any(y < 0):
            raise ValueError("y must be non-negative for Poisson regression")

    def predict(self, X, *, offset=None) -> np.ndarray:
        """Expected count ``exp(eta)``."""
        return np.exp(self.decision_function(X, offset=offset))


class ConditionalPoissonRegression(PoissonRegression):
    """Conditional Poisson regression (Cyclops ``"cpr"``). Requires ``strata=``."""

    _model_type = "cpr"
    _allows_intercept = False

    def fit(self, X, y, *, strata=None, **kwargs):
        if strata is None:
            raise ValueError(f"{type(self).__name__} requires `strata`")
        return super().fit(X, y, strata=strata, **kwargs)

    def predict(self, X, *, offset=None) -> np.ndarray:
        """Linear predictor: the conditional likelihood has no baseline rate."""
        return self.decision_function(X, offset=offset)


class SelfControlledCaseSeries(BaseCyclopsEstimator):
    """Self-controlled case series (Cyclops ``"sccs"``).

    Requires both ``strata=`` (one per person) and ``offset=`` (person-time per
    interval).
    """

    _model_type = "sccs"
    _allows_intercept = False

    def fit(self, X, y, *, strata=None, offset=None, **kwargs):
        if strata is None:
            raise ValueError(f"{type(self).__name__} requires `strata`")
        if offset is None:
            raise ValueError(f"{type(self).__name__} requires `offset` (person-time)")
        return super().fit(X, y, strata=strata, offset=offset, **kwargs)

    def predict(self, X, *, offset=None) -> np.ndarray:
        return self.decision_function(X, offset=offset)


class LinearRegression(BaseCyclopsEstimator):
    """Least-squares regression (Cyclops ``"ls"``)."""

    _model_type = "ls"

    def predict(self, X) -> np.ndarray:
        return self.decision_function(X)


# ---------------------------------------------------------------------------
# Survival models
# ---------------------------------------------------------------------------


class CoxRegression(BaseCyclopsEstimator):
    """Cox proportional hazards (Cyclops ``"cox"``).

    Call ``fit(X, event, time=...)`` — the outcome is the 0/1 event indicator and
    ``time`` the follow-up duration. ``strata=`` is optional. Rows are sorted
    internally by ``(stratum, -time, event)``, which is what the risk-set
    recursions require.

    ``predict`` returns the log relative hazard, so it orders subjects by risk
    but is not an absolute survival probability.
    """

    _model_type = "cox"
    _allows_intercept = False

    def _validate_outcome(self, y: np.ndarray) -> None:
        super()._validate_outcome(y)
        unique = np.unique(y)
        if not np.all(np.isin(unique, (0.0, 1.0))):
            raise ValueError(
                "the outcome must be a 0/1 event indicator; pass durations as `time`"
            )

    def fit(self, X, y, *, time=None, **kwargs):
        if time is None:
            raise ValueError(f"{type(self).__name__} requires `time`")
        return super().fit(X, y, time=time, **kwargs)

    def predict(self, X) -> np.ndarray:
        """Log relative hazard (the linear predictor)."""
        return self.decision_function(X)

    def hazard_ratios(self) -> np.ndarray:
        """``exp(coef_)``."""
        self._check_fitted()
        return np.exp(self.coef_)


class TimeVaryingCoxRegression(CoxRegression):
    """Cox model with time-varying covariates (Cyclops ``"cox_time"``)."""

    _model_type = "cox_time"


class FineGrayRegression(CoxRegression):
    """Fine-Gray competing-risks regression (Cyclops ``"fgr"``).

    Requires ``censor_weights=`` — subject-specific inverse-probability-of-censoring
    weights in ``[0, 1]``, the analogue of ``getFineGrayWeights()`` in R.
    """

    _model_type = "fgr"

    def fit(self, X, y, *, time=None, censor_weights=None, **kwargs):
        if censor_weights is None:
            raise ValueError(
                f"{type(self).__name__} requires `censor_weights`; see "
                "getFineGrayWeights() in the R package for how they are derived"
            )
        return super().fit(X, y, time=time, censor_weights=censor_weights, **kwargs)
