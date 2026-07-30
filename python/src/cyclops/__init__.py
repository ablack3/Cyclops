"""Cyclops: cyclic coordinate descent for large-scale regularized regression.

Python bindings for the OHDSI Cyclops C++ engine. Fits logistic, Poisson,
least-squares, conditional (stratified) and survival regressions with L1/L2
regularization on sparse designs with hundreds of thousands of covariates.

Two levels of API:

* estimators — :class:`LogisticRegression`, :class:`CoxRegression` and friends,
  shaped like scikit-learn::

      from cyclops import LogisticRegression
      model = LogisticRegression(prior="laplace", prior_variance=0.1)
      model.fit(X, y)
      model.coef_, model.predict_proba(X)

* the data/model pair — :class:`~cyclops.data.CyclopsData` and
  :class:`~cyclops.model.CyclopsModel`, for streaming very wide designs and for
  the controls the estimators do not surface.
"""

from __future__ import annotations

from cyclops import _cyclops
from cyclops.data import INTERCEPT_ID, CyclopsData
from cyclops.estimators import (
    ConditionalLogisticRegression,
    ConditionalPoissonRegression,
    CoxRegression,
    EfronConditionalLogisticRegression,
    ExactConditionalLogisticRegression,
    FineGrayRegression,
    LinearRegression,
    LogisticRegression,
    NotFittedError,
    PoissonRegression,
    SelfControlledCaseSeries,
    TimeVaryingCoxRegression,
)
from cyclops.model import Control, CyclopsError, CyclopsModel, FitResult, Prior

__version__ = _cyclops.version()

#: Cyclops model-type codes accepted by :meth:`CyclopsData.create`.
MODEL_TYPES = (
    "ls", "pr", "lr", "clr", "clr_exact", "clr_efron", "cpr",
    "sccs", "cox", "cox_raw", "cox_time", "fgr",
)


def list_gpu_devices() -> list[str]:
    """GPU devices visible to this build; empty for the CPU-only wheels."""
    return _cyclops.list_gpu_devices()


__all__ = [
    "__version__",
    "MODEL_TYPES",
    # estimators
    "LogisticRegression",
    "PoissonRegression",
    "LinearRegression",
    "ConditionalLogisticRegression",
    "ExactConditionalLogisticRegression",
    "EfronConditionalLogisticRegression",
    "ConditionalPoissonRegression",
    "SelfControlledCaseSeries",
    "CoxRegression",
    "TimeVaryingCoxRegression",
    "FineGrayRegression",
    # lower level
    "CyclopsData",
    "CyclopsModel",
    "Prior",
    "Control",
    "FitResult",
    "INTERCEPT_ID",
    # errors
    "CyclopsError",
    "NotFittedError",
    # misc
    "list_gpu_devices",
]
