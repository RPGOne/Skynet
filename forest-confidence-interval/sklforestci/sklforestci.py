import numpy as np
from sklearn.ensemble.forest import _generate_sample_indices
from .due import _due, _BibTeX

__all__ = ["calc_inbag", "random_forest_error", "_bias_correction",
           "_core_computation"]

_due.cite(_BibTeX("""
@ARTICLE{Wager2014-wn,
  title       = "Confidence Intervals for Random Forests: The Jackknife and the Infinitesimal Jackknife",
  author      = "Wager, Stefan and Hastie, Trevor and Efron, Bradley",
  journal     = "J. Mach. Learn. Res.",
  volume      =  15,
  number      =  1,
  pages       = "1625--1651",
  month       =  jan,
  year        =  2014,}"""),
          description=("Confidence Intervals for Random Forests:",
                       "The Jackknife and the Infinitesimal Jackknife"),
          path='sklforestci')


def calc_inbag(n_samples, forest):
    """
    Derive samples used to create trees in scikit-learn RandomForest objects.

    Recovers the samples in each tree from the random state of that tree using
    :func:`forest._generate_sample_indices`.

    Parameters
    ----------
    n_samples : int
        The number of samples used to fit the scikit-learn RandomForest object.

    forest : RandomForest
        Regressor or Classifier object that is already fit by scikit-learn.

    Returns
    -------
    Array that records how many times a data point was placed in a tree.
    Columns are individual trees. Rows are the number of times a sample was
    used in a tree.
    """
    n_trees = forest.n_estimators
    inbag = np.zeros((n_samples, n_trees))
    sample_idx = []
    for t_idx in range(n_trees):
        sample_idx.append(
            _generate_sample_indices(forest.estimators_[t_idx].random_state,
                                     n_samples))
        inbag[:, t_idx] = np.bincount(sample_idx[-1], minlength=n_samples)
    return inbag


def _core_computation(X_train, X_test, inbag, pred_centered, n_trees):
    cov_hat = np.zeros((X_train.shape[0], X_test.shape[0]))

    for t_idx in range(n_trees):
        inbag_r = (inbag[:, t_idx] - 1).reshape(-1, 1)
        pred_c_r = pred_centered.T[t_idx].reshape(1, -1)
        cov_hat += np.dot(inbag_r, pred_c_r) / n_trees
    V_IJ = np.sum(cov_hat ** 2, 0)
    return V_IJ


def _bias_correction(V_IJ, inbag, pred_centered, n_trees):
    n_train_samples = inbag.shape[0]
    n_var = np.mean(np.square(inbag[0:n_trees]).mean(axis=1).T.view() -
                    np.square(inbag[0:n_trees].mean(axis=1)).T.view())
    boot_var = np.square(pred_centered).sum(axis=1) / n_trees
    bias_correction = n_train_samples * n_var * boot_var / n_trees
    V_IJ_unbiased = V_IJ - bias_correction
    return V_IJ_unbiased


def random_forest_error(forest, inbag, X_train, X_test):
    """
    Calculates error bars from scikit-learn RandomForest estimators.

    RandomForest is a regressor or classifier object
    this variance can be used to plot error bars for RandomForest objects

    Parameters
    ----------
    forest : RandomForest
        Regressor or Classifier object.

    inbag : ndarray
        The inbag matrix that fit the data.

    X : ndarray
        An array with shape (n_sample, n_features).

    Returns
    -------
    An array with the unbiased sampling variance (V_IJ_unbiased)
    for a RandomForest object.

    See Also
    ----------
    :func:`calc_inbag`

    Notes
    -----
    The calculation of error is based on the infinitesimal jackknife variance,
    as described in [Wager2014]_ and is a Python implementation of the R code
    provided at: https://github.com/swager/randomForestCI

    .. [Wager2014] S. Wager, T. Hastie, B. Efron. "Confidence Intervals for
       Random Forests: The Jackknife and the Infinitesimal Jackknife", Journal
       of Machine Learning Research vol. 15, pp. 1625-1651, 2014.
    """
    pred = np.array([tree.predict(X_test) for tree in forest]).T
    pred_mean = np.mean(pred, 0)
    pred_centered = pred - pred_mean
    n_trees = forest.n_estimators
    V_IJ = _core_computation(X_train, X_test, inbag, pred_centered, n_trees)
    V_IJ_unbiased = _bias_correction(V_IJ, inbag, pred_centered, n_trees)
    return V_IJ_unbiased
