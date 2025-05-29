"""Model definitions.

Every model is a full `Pipeline` -- preprocessing included -- so that
cross-validation and tuning re-fit the transforms inside each fold. A bare
estimator handed to `cross_val_score` alongside pre-transformed data is the
most common way leakage re-enters a project after it has been designed out.

Scaling is on for models with a linear decision surface and off for trees, which
read only rank order.
"""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

from .features import build_preprocessor, collapse_codes

RANDOM_STATE = 42


def _wrap(estimator, scale: bool, drop_sex: bool = False) -> Pipeline:
    return Pipeline([
        ("collapse", FunctionTransformer(collapse_codes)),
        ("prep", build_preprocessor(scale=scale, drop_sex=drop_sex)),
        ("clf", estimator),
    ])


def logistic(C: float = 1.0, penalty: str = "l2", class_weight=None, drop_sex=False) -> Pipeline:
    return _wrap(
        LogisticRegression(
            C=C, penalty=penalty, class_weight=class_weight,
            solver="liblinear" if penalty == "l1" else "lbfgs",
            max_iter=4000, random_state=RANDOM_STATE,
        ),
        scale=True, drop_sex=drop_sex,
    )


def decision_tree(max_depth=6, min_samples_leaf=50, class_weight=None,
                  drop_sex=False) -> Pipeline:
    """A single tree, depth-limited so it stays readable.

    Included because it is the interpretable benchmark: a depth-6 tree can be
    drawn on one page and argued with, which neither boosting nor a forest can.
    Left unconstrained it would memorise the training split, so the depth and
    leaf-size limits are the model, not tidying.
    """
    return _wrap(
        DecisionTreeClassifier(
            max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            class_weight=class_weight, random_state=RANDOM_STATE,
        ),
        scale=False, drop_sex=drop_sex,
    )


def random_forest(n_estimators=400, max_depth=None, min_samples_leaf=1,
                  class_weight=None, drop_sex=False) -> Pipeline:
    return _wrap(
        RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, class_weight=class_weight,
            n_jobs=1, random_state=RANDOM_STATE,
        ),
        scale=False, drop_sex=drop_sex,
    )


def hist_gradient_boosting(learning_rate=0.1, max_leaf_nodes=31, max_iter=200,
                           min_samples_leaf=20, l2_regularization=0.0,
                           class_weight=None, drop_sex=False) -> Pipeline:
    """sklearn's own boosted trees.

    Preferred over LightGBM here: comparable accuracy on tabular data this size
    and one fewer dependency. LightGBM stays in the comparison to confirm that
    choice costs nothing.
    """
    return _wrap(
        HistGradientBoostingClassifier(
            learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
            max_iter=max_iter, min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization, class_weight=class_weight,
            random_state=RANDOM_STATE,
        ),
        scale=False, drop_sex=drop_sex,
    )


def lightgbm(n_estimators=300, learning_rate=0.05, num_leaves=31,
             min_child_samples=20, class_weight=None, drop_sex=False) -> Pipeline:
    import lightgbm as lgb

    return _wrap(
        lgb.LGBMClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate,
            num_leaves=num_leaves, min_child_samples=min_child_samples,
            class_weight=class_weight, random_state=RANDOM_STATE,
            # n_jobs=1 deliberately: cross_validate already parallelises across
            # folds, and nesting LightGBM's own thread pool inside that
            # oversubscribes the CPU badly enough to stall the run.
            n_jobs=1, verbose=-1,
        ),
        scale=False, drop_sex=drop_sex,
    )


BUILDERS = {
    "logistic_l2": lambda **kw: logistic(penalty="l2", **kw),
    "decision_tree": decision_tree,
    "random_forest": random_forest,
    "hist_gradient_boosting": hist_gradient_boosting,
    "lightgbm": lightgbm,
}
