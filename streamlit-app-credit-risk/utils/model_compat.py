"""Compatibility helpers for loading older pickled scikit-learn models."""

from __future__ import annotations

import sys


def ensure_sklearn_loss_compat() -> None:
    """Alias scikit-learn's internal loss module to the legacy pickle name.

    Some saved estimators reference the private module name ``_loss`` during
    unpickling. Newer scikit-learn versions expose the implementation under
    ``sklearn._loss`` instead, so we register a compatibility alias before
    loading the pickle.
    """
    import sklearn._loss as sklearn_loss
    from sklearn._loss import link as link_module
    from sklearn._loss import loss as loss_module

    for source_module in (loss_module, link_module):
        for name in dir(source_module):
            if name.startswith("_"):
                continue
            setattr(sklearn_loss, name, getattr(source_module, name))

    sys.modules.setdefault("_loss", sklearn_loss)
    sys.modules.setdefault("_loss.loss", loss_module)
    sys.modules.setdefault("_loss.link", link_module)
