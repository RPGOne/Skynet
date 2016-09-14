"""
python speedup_kmeans.py --profile
python speedup_kmeans.py

git worktree add workdir_master master
rob sedr "\<sklearn\>" sklearn_master True
git mv sklearn sklearn_master
python setup develop
python -c "import sklearn_master; print(sklearn_master.__file__)"
python -c "import sklearn; print(sklearn.__file__)"
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import sklearn  # NOQA
from sklearn.datasets.samples_generator import make_blobs
from sklearn.utils.extmath import row_norms, squared_norm  # NOQA
import sklearn.cluster
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances  # NOQA

import sklearn_master.cluster
(print, rrr, profile) = ut.inject2(__name__, '[tester]')


def test_kmeans_plus_plus_speed(n_clusters=2000, n_features=128, per_cluster=10, asint=False, fix=True):
    """
    from speedup_kmeans import *
    from sklearn.cluster.k_means_ import *
    """
    rng = np.random.RandomState(42)
    # Make random cluster centers on a ball
    centers = rng.rand(n_clusters, n_features)
    centers /= np.linalg.norm(centers, axis=0)[None, :]
    centers = (centers * 512).astype(np.uint8) / 512
    centers /= np.linalg.norm(centers, axis=0)[None, :]

    n_samples = int(n_clusters * per_cluster)
    n_clusters, n_features = centers.shape
    X, true_labels = make_blobs(n_samples=n_samples, centers=centers,
                                cluster_std=1., random_state=42)

    if asint:
        X = (X * 512).astype(np.int32)

    x_squared_norms = row_norms(X, squared=True)

    if fix:
        _k_init = sklearn.cluster.k_means_._k_init
    else:
        _k_init = sklearn_master.cluster.k_means_._k_init
    random_state = np.random.RandomState(42)
    n_local_trials = None  # NOQA

    with ut.Timer('testing kmeans init') as t:
        centers = _k_init(X, n_clusters, random_state=random_state, x_squared_norms=x_squared_norms)
    return centers, t.ellapsed


def main():
    if True:
        import pandas as pd
        pd.options.display.max_rows = 1000
        pd.options.display.width = 1000

        basis = {
            #'n_clusters': [10, 100, 1000, 2000][::-1],
            #'n_features': [4, 32, 128, 512][::-1],
            #'per_cluster': [1, 10, 100, 200][::-1],
            'n_clusters': [10, 100, 500][::-1],
            'n_features': [32, 128][::-1],
            'per_cluster': [1, 10, 20][::-1],
            'asint': [True, False],
        }
        vals = []
        for kw in ut.ProgIter(ut.all_dict_combinations(basis), lbl='gridsearch',
                              bs=False, adjust=False, freq=1):
            print('kw = ' + ut.repr2(kw))
            exec(ut.execstr_dict(kw))
            centers1, new_speed = test_kmeans_plus_plus_speed(fix=True, **kw)
            centers2, old_speed = test_kmeans_plus_plus_speed(fix=False, **kw)
            import utool
            with utool.embed_on_exception_context:
                assert np.all(centers1 == centers2), 'new code disagrees'

            kw['new_speed'] = new_speed
            kw['old_speed'] = old_speed
            vals.append(kw)
            print('---------')

        df = pd.DataFrame.from_dict(vals)
        df['percent_change'] = 100 * (df['old_speed'] - df['new_speed']) / df['old_speed']
        df = df.reindex_axis(list(basis.keys()) + ['new_speed', 'old_speed', 'percent_change'], axis=1)
        df['absolute_change'] = (df['old_speed'] - df['new_speed'])
        print(df.sort('absolute_change', ascending=False))
        #print(df)

        print(df['percent_change'][df['absolute_change'] > .1].mean())

    #print(df.loc[df['percent_change'].argsort()[::-1]])
    else:
        new_speed = test_kmeans_plus_plus_speed()
        try:
            profile.dump_stats('out.lprof')
            profile.print_stats(stripzeros=True)
        except Exception:
            pass
        print('new_speed = %r' % (new_speed,))

if __name__ == '__main__':
    main()
