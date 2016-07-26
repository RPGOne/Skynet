import numpy as np
import scipy.sparse as sp
from scipy import linalg, optimize

from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_warns
from sklearn.utils.testing import raises
from sklearn.utils.testing import ignore_warnings
from sklearn.utils import ConvergenceWarning

from sklearn.linear_model.logistic import (
    LogisticRegression,
    logistic_regression_path, LogisticRegressionCV,
    _logistic_loss_and_grad, _logistic_loss_grad_hess
    )
from sklearn.cross_validation import StratifiedKFold
from sklearn.datasets import load_iris, make_classification

X = [[-1, 0], [0, 1], [1, 1]]
X_sp = sp.csr_matrix(X)
Y1 = [0, 1, 1]
Y2 = [2, 1, 0]
iris = load_iris()


def check_predictions(clf, X, y):
    """Check that the model is able to fit the classification data"""
    n_samples = len(y)
    classes = np.unique(y)
    n_classes = classes.shape[0]

    predicted = clf.fit(X, y).predict(X)
    assert_array_equal(clf.classes_, classes)

    assert_equal(predicted.shape, (n_samples,))
    assert_array_equal(predicted, y)

    probabilities = clf.predict_proba(X)
    assert_equal(probabilities.shape, (n_samples, n_classes))
    assert_array_almost_equal(probabilities.sum(axis=1), np.ones(n_samples))
    assert_array_equal(probabilities.argmax(axis=1), y)


def test_predict_2_classes():
    """Simple sanity check on a 2 classes dataset

    Make sure it predicts the correct result on simple datasets.
    """
    check_predictions(LogisticRegression(random_state=0), X, Y1)
    check_predictions(LogisticRegression(random_state=0), X_sp, Y1)

    check_predictions(LogisticRegression(C=100, random_state=0), X, Y1)
    check_predictions(LogisticRegression(C=100, random_state=0), X_sp, Y1)

    check_predictions(LogisticRegression(fit_intercept=False,
                                         random_state=0), X, Y1)
    check_predictions(LogisticRegression(fit_intercept=False,
                                         random_state=0), X_sp, Y1)


def test_error():
    """Test for appropriate exception on errors"""
    assert_raises(ValueError, LogisticRegression(C=-1).fit, X, Y1)


def test_predict_3_classes():
    check_predictions(LogisticRegression(C=10), X, Y2)
    check_predictions(LogisticRegression(C=10), X_sp, Y2)


def test_predict_iris():
    """Test logistic regression with the iris dataset"""
    n_samples, n_features = iris.data.shape

    target = iris.target_names[iris.target]
    clf = LogisticRegression(C=len(iris.data)).fit(iris.data, target)
    assert_array_equal(np.unique(target), clf.classes_)

    pred = clf.predict(iris.data)
    assert_greater(np.mean(pred == target), .95)

    probabilities = clf.predict_proba(iris.data)
    assert_array_almost_equal(probabilities.sum(axis=1), np.ones(n_samples))

    pred = iris.target_names[probabilities.argmax(axis=1)]
    assert_greater(np.mean(pred == target), .95)


def test_sparsify():
    """Test sparsify and densify members."""
    n_samples, n_features = iris.data.shape
    target = iris.target_names[iris.target]
    clf = LogisticRegression(random_state=0).fit(iris.data, target)

    pred_d_d = clf.decision_function(iris.data)

    clf.sparsify()
    assert_true(sp.issparse(clf.coef_))
    pred_s_d = clf.decision_function(iris.data)

    sp_data = sp.coo_matrix(iris.data)
    pred_s_s = clf.decision_function(sp_data)

    clf.densify()
    pred_d_s = clf.decision_function(sp_data)

    assert_array_almost_equal(pred_d_d, pred_s_d)
    assert_array_almost_equal(pred_d_d, pred_s_s)
    assert_array_almost_equal(pred_d_d, pred_d_s)


def test_inconsistent_input():
    """Test that an exception is raised on inconsistent input"""
    rng = np.random.RandomState(0)
    X_ = rng.random_sample((5, 10))
    y_ = np.ones(X_.shape[0])
    y_[0] = 0

    clf = LogisticRegression(random_state=0)

    # Wrong dimensions for training data
    y_wrong = y_[:-1]
    assert_raises(ValueError, clf.fit, X, y_wrong)

    # Wrong dimensions for test data
    assert_raises(ValueError, clf.fit(X_, y_).predict,
                  rng.random_sample((3, 12)))


def test_write_parameters():
    """Test that we can write to coef_ and intercept_"""
    clf = LogisticRegression(random_state=0)
    clf.fit(X, Y1)
    clf.coef_[:] = 0
    clf.intercept_[:] = 0
    assert_array_almost_equal(clf.decision_function(X), 0)


@raises(ValueError)
def test_nan():
    """Test proper NaN handling.

    Regression test for Issue #252: fit used to go into an infinite loop.
    """
    Xnan = np.array(X, dtype=np.float64)
    Xnan[0, 1] = np.nan
    LogisticRegression(random_state=0).fit(Xnan, Y1)


def test_consistency_path():
    """Test that the path algorithm is consistent"""
    rng = np.random.RandomState(0)
    X = np.concatenate((rng.randn(100, 2) + [1, 1], rng.randn(100, 2)))
    y = [1] * 100 + [-1] * 100
    Cs = np.logspace(0, 4, 10)

    f = ignore_warnings
    # can't test with fit_intercept=True since LIBLINEAR
    # penalizes the intercept
    for method in ('lbfgs', 'newton-cg', 'liblinear'):
        coefs, Cs = f(logistic_regression_path)(
            X, y, Cs=Cs, fit_intercept=False, tol=1e-16, solver=method)
        for i, C in enumerate(Cs):
            lr = LogisticRegression(C=C, fit_intercept=False, tol=1e-16)
            lr.fit(X, y)
            lr_coef = lr.coef_.ravel()
            assert_array_almost_equal(lr_coef, coefs[i], decimal=4)

    # test for fit_intercept=True
    for method in ('lbfgs', 'newton-cg', 'liblinear'):
        Cs = [1e3]
        coefs, Cs = f(logistic_regression_path)(
            X, y, Cs=Cs, fit_intercept=True, tol=1e-4, solver=method)
        lr = LogisticRegression(C=Cs[0], fit_intercept=True, tol=1e-4,
                                intercept_scaling=10000)
        lr.fit(X, y)
        lr_coef = np.concatenate([lr.coef_.ravel(), lr.intercept_])
        assert_array_almost_equal(lr_coef, coefs[0], decimal=4)


def test_liblinear_random_state():
    X, y = make_classification(n_samples=20)
    lr1 = LogisticRegression(random_state=0)
    lr1.fit(X, y)
    lr2 = LogisticRegression(random_state=0)
    lr2.fit(X, y)
    assert_array_almost_equal(lr1.coef_, lr2.coef_)


def test_logistic_loss_and_grad():
    X_ref, y = make_classification(n_samples=20)
    n_features = X_ref.shape[1]

    X_sp = X_ref.copy()
    X_sp[X_sp < .1] = 0
    X_sp = sp.csr_matrix(X_sp)
    for X in (X_ref, X_sp):
        w = np.zeros(n_features)

        # First check that our derivation of the grad is correct
        loss, grad = _logistic_loss_and_grad(w, X, y, alpha=1.)
        approx_grad = optimize.approx_fprime(
            w, lambda w: _logistic_loss_and_grad(w, X, y, alpha=1.)[0], 1e-3
            )
        assert_array_almost_equal(grad, approx_grad, decimal=2)

        # Second check that our intercept implementation is good
        w = np.zeros(n_features + 1)
        loss_interp, grad_interp = _logistic_loss_and_grad(
            w, X, y, alpha=1.
            )
        assert_array_almost_equal(loss, loss_interp)

        approx_grad = optimize.approx_fprime(
            w, lambda w: _logistic_loss_and_grad(w, X, y, alpha=1.)[0], 1e-3
            )
        assert_array_almost_equal(grad_interp, approx_grad, decimal=2)


def test_logistic_loss_grad_hess():
    rng = np.random.RandomState(0)
    n_samples, n_features = 50, 5
    X_ref = rng.randn(n_samples, n_features)
    y = np.sign(X_ref.dot(5 * rng.randn(n_features)))
    X_ref -= X_ref.mean()
    X_ref /= X_ref.std()
    X_sp = X_ref.copy()
    X_sp[X_sp < .1] = 0
    X_sp = sp.csr_matrix(X_sp)
    for X in (X_ref, X_sp):
        w = .1 * np.ones(n_features)

        # First check that _logistic_loss_grad_hess is consistent
        # with _logistic_loss_and_grad
        loss, grad = _logistic_loss_and_grad(w, X, y, alpha=1.)
        loss_2, grad_2, hess = _logistic_loss_grad_hess(w, X, y, alpha=1.)
        assert_array_almost_equal(grad, grad_2)

        # Now check our hessian along the second direction of the grad
        vector = np.zeros_like(grad)
        vector[1] = 1
        hess_col = hess(vector)

        # Computation of the Hessian is particularly fragile to numerical
        # errors when doing simple finite differences. Here we compute the
        # grad along a path in the direction of the vector and then use a
        # least-square regression to estimate the slope
        e = 1e-3
        d_x = np.linspace(-e, e, 30)
        d_grad = np.array([
            _logistic_loss_and_grad(w + t * vector, X, y, alpha=1.)[1]
            for t in d_x
            ])

        d_grad -= d_grad.mean(axis=0)
        approx_hess_col = linalg.lstsq(d_x[:, np.newaxis], d_grad)[0].ravel()

        assert_array_almost_equal(approx_hess_col, hess_col, decimal=3)

        # Second check that our intercept implementation is good
        w = np.zeros(n_features + 1)
        loss_interp, grad_interp = _logistic_loss_and_grad(
            w, X, y, alpha=1.
            )
        loss_interp_2, grad_interp_2, hess = \
            _logistic_loss_grad_hess(w, X, y, alpha=1.)
        assert_array_almost_equal(loss_interp, loss_interp_2)
        assert_array_almost_equal(grad_interp, grad_interp_2)


def test_logistic_cv():
    """test for LogisticRegressionCV object"""
    n_samples, n_features = 50, 5
    rng = np.random.RandomState(0)
    X_ref = rng.randn(n_samples, n_features)
    y = np.sign(X_ref.dot(5 * rng.randn(n_features)))
    X_ref -= X_ref.mean()
    X_ref /= X_ref.std()
    lr_cv = LogisticRegressionCV(Cs=[1.], fit_intercept=False,
                                 solver='liblinear')
    lr_cv.fit(X_ref, y)
    lr = LogisticRegression(C=1., fit_intercept=False)
    lr.fit(X_ref, y)
    assert_array_almost_equal(lr.coef_, lr_cv.coef_)

    assert_array_equal(lr_cv.coef_.shape, (1, n_features))
    assert_array_equal(lr_cv.classes_, [-1, 1])
    assert_equal(len(lr_cv.classes_), 2)

    coefs_paths = np.asarray(list(lr_cv.coefs_paths_.values()))
    assert_array_equal(coefs_paths.shape, (1, 3, 1, n_features))
    assert_array_equal(lr_cv.Cs_.shape, (1, ))
    scores = np.asarray(list(lr_cv.scores_.values()))
    assert_array_equal(scores.shape, (1, 3, 1))


def test_logistic_cv_sparse():
    X, y = make_classification(n_samples=50, n_features=5,
                               random_state=0)
    X[X < 1.0] = 0.0
    csr = sp.csr_matrix(X)

    clf = LogisticRegressionCV(fit_intercept=True)
    clf.fit(X, y)
    clfs = LogisticRegressionCV(fit_intercept=True)
    clfs.fit(csr, y)
    assert_array_almost_equal(clfs.coef_, clf.coef_)
    assert_array_almost_equal(clfs.intercept_, clf.intercept_)
    assert_equal(clfs.C_, clf.C_)


def test_intercept_logistic_helper():
    n_samples, n_features = 10, 5
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               random_state=0)

    # Fit intercept case.
    alpha = 1.
    w = np.ones(n_features + 1)
    loss_interp, grad_interp, hess_interp = _logistic_loss_grad_hess(
        w, X, y, alpha)

    # Do not fit intercept. This can be considered equivalent to adding
    # a feature vector of ones, i.e column of one vectors.
    X_ = np.hstack((X, np.ones(10)[:, np.newaxis]))
    loss, grad, hess = _logistic_loss_grad_hess(w, X_, y, alpha)

    # In the fit_intercept=False case, the feature vector of ones is
    # penalized. This should be taken care of.
    assert_almost_equal(loss_interp + 0.5 * (w[-1] ** 2), loss)

    # Check gradient.
    assert_array_almost_equal(grad_interp[:n_features], grad[:n_features])
    assert_almost_equal(grad_interp[-1] + alpha * w[-1], grad[-1])

    rng = np.random.RandomState(0)
    grad = rng.rand(n_features + 1)
    hess_interp = hess_interp(grad)
    hess = hess(grad)
    assert_array_almost_equal(hess_interp[:n_features], hess[:n_features])
    assert_almost_equal(hess_interp[-1] + alpha * grad[-1], hess[-1])


def test_ova_iris():
    """Test that our OvA implementation is correct using the iris dataset."""
    train, target = iris.data, iris.target
    n_samples, n_features = train.shape

    # Use pre-defined fold as folds generated for different y
    cv = StratifiedKFold(target, 3)
    clf = LogisticRegressionCV(cv=cv)
    clf.fit(train, target)

    clf1 = LogisticRegressionCV(cv=cv)
    target[target == 0] = 1
    clf1.fit(train, target)

    assert_array_almost_equal(clf.scores_[2], clf1.scores_[2])
    assert_array_almost_equal(clf.intercept_[2:], clf1.intercept_)
    assert_array_almost_equal(clf.coef_[2][np.newaxis, :], clf1.coef_)

    # Test the shape of various attributes.
    assert_equal(clf.coef_.shape, (3, n_features))
    assert_array_equal(clf.classes_, [0, 1, 2])
    assert_equal(len(clf.classes_), 3)

    coefs_paths = np.asarray(list(clf.coefs_paths_.values()))
    assert_array_almost_equal(coefs_paths.shape, (3, 3, 10, n_features + 1))
    assert_equal(clf.Cs_.shape, (10, ))
    scores = np.asarray(list(clf.scores_.values()))
    assert_equal(scores.shape, (3, 3, 10))


def test_logistic_regression_solvers():
    X, y = make_classification(n_features=10, n_informative=5, random_state=0)
    clf_n = LogisticRegression(solver='newton-cg', fit_intercept=False)
    clf_n.fit(X, y)
    clf_lbf = LogisticRegression(solver='lbfgs', fit_intercept=False)
    clf_lbf.fit(X, y)
    clf_lib = LogisticRegression(fit_intercept=False)
    clf_lib.fit(X, y)
    assert_array_almost_equal(clf_n.coef_, clf_lib.coef_, decimal=3)
    assert_array_almost_equal(clf_lib.coef_, clf_lbf.coef_, decimal=3)
    assert_array_almost_equal(clf_n.coef_, clf_lbf.coef_, decimal=3)


def test_logistic_regression_solvers_multiclass():
    X, y = make_classification(n_samples=20, n_features=20, n_informative=10,
                               n_classes=3, random_state=0)
    clf_n = LogisticRegression(solver='newton-cg', fit_intercept=False)
    clf_n.fit(X, y)
    clf_lbf = LogisticRegression(solver='lbfgs', fit_intercept=False)
    clf_lbf.fit(X, y)
    clf_lib = LogisticRegression(fit_intercept=False)
    clf_lib.fit(X, y)
    assert_array_almost_equal(clf_n.coef_, clf_lib.coef_, decimal=4)
    assert_array_almost_equal(clf_lib.coef_, clf_lbf.coef_, decimal=4)
    assert_array_almost_equal(clf_n.coef_, clf_lbf.coef_, decimal=4)


def test_logistic_regressioncv_class_weights():
    X, y = make_classification(n_samples=20, n_features=20, n_informative=10,
                               n_classes=3, random_state=0)

    # Test the liblinear fails when class_weight of type dict is
    # provided, when it is multiclass
    clf_lib = LogisticRegressionCV(class_weight={0: 0.1, 1: 0.2},
                                   solver='liblinear')
    assert_raises(ValueError, clf_lib.fit, X, y)

    # Test for class_weight=auto
    X, y = make_classification(n_samples=20, n_features=20, n_informative=10,
                               random_state=0)
    clf_lbf = LogisticRegressionCV(solver='lbfgs', fit_intercept=False,
                                   class_weight='auto')
    clf_lbf.fit(X, y)
    clf_lib = LogisticRegressionCV(solver='liblinear', fit_intercept=False,
                                   class_weight='auto')
    clf_lib.fit(X, y)
    assert_array_almost_equal(clf_lib.coef_, clf_lbf.coef_, decimal=4)


def test_logistic_regression_convergence_warnings():
    """Test that warnings are raised if model does not converge"""

    X, y = make_classification(n_samples=20, n_features=20)
    clf_lib = LogisticRegression(solver='liblinear', max_iter=2)
    assert_warns(ConvergenceWarning, clf_lib.fit, X, y)
    assert_equal(clf_lib.n_iter_, 2)
