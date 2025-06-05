"""Model comparison, the class-imbalance test, the encoding ablation, and tuning.

Answers four of the problem statement's objectives in one pass:

* **objective 5** -- compare logistic regression, a decision tree, and ensembles
* **objective 2** -- handle class imbalance (tested here as class weights)
* **objective 4** -- financially meaningful feature engineering (the ablation)
* **objective 6** -- choose metrics (PR-AUC leads, with its no-skill floor)

Everything it prints is also written to `experiments/`, so the README and the
notebooks read numbers from disk instead of repeating them.

Run:  ./.venv/bin/python run_compare.py
"""
import json
import pathlib
import sys
import warnings

sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import experiments, models
from src.data import TARGET
from src.evaluate import comparison_table, evaluate, majority_baseline
from src.split import load_split, make_splits

OUT = pathlib.Path('experiments'); OUT.mkdir(exist_ok=True)
MODELS = pathlib.Path('models'); MODELS.mkdir(exist_ok=True)

make_splits()  # builds the train/val/test parquet files on a fresh clone; no-op afterwards
tr, va = load_split('train'), load_split('val')
Xtr, ytr = tr.drop(columns=[TARGET]), tr[TARGET]
Xva, yva = va.drop(columns=[TARGET]), va[TARGET]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"train {Xtr.shape}   validation {Xva.shape}   base rate {yva.mean():.4f}")
experiments.clear()

# ---------------------------------------------------------------------------
# 1. Model comparison, each candidate with and without class weighting.
# ---------------------------------------------------------------------------
# `class_weight="balanced"` is the cheapest of the three imbalance remedies the
# PS names (the others being SMOTE and downsampling). It re-weights the loss so
# the minority class counts ~3.5x more, which moves the decision boundary. The
# question is whether that buys anything PR-AUC can see, or only shifts the
# probabilities -- which is what the Brier column is there to catch.
BASE_PARAMS = {
    'logistic_l2': dict(),
    'decision_tree': dict(max_depth=6, min_samples_leaf=50),
    'random_forest': dict(min_samples_leaf=5),
    'hist_gradient_boosting': dict(),
    'lightgbm': dict(),
}

print(f"\n{'='*78}\n1. MODEL COMPARISON -- 5-fold stratified CV on train, scored on validation\n{'='*78}")
rows = []
for name, base in BASE_PARAMS.items():
    for weighting in (None, 'balanced'):
        params = {**base, **({'class_weight': weighting} if weighting else {})}
        pipe = models.BUILDERS[name](**params)
        cvres = cross_validate(pipe, Xtr, ytr, cv=cv,
                               scoring=['roc_auc', 'average_precision'], n_jobs=-1)
        pipe.fit(Xtr, ytr)
        m = evaluate(yva, pipe.predict_proba(Xva)[:, 1], 0.5)
        tag = name + ('+balanced' if weighting else '')
        rows.append(dict(
            model=tag, class_weight=weighting or 'none',
            cv_roc_auc=cvres['test_roc_auc'].mean(), cv_roc_sd=cvres['test_roc_auc'].std(),
            cv_pr_auc=cvres['test_average_precision'].mean(),
            cv_pr_sd=cvres['test_average_precision'].std(),
            val_roc_auc=m['roc_auc'], val_pr_auc=m['pr_auc'], brier=m['brier'],
            precision=m['precision'], recall=m['recall'], f1=m['f1']))
        experiments.record('1_model_comparison', tag, params, m,
                           notes=f"cv_roc_auc={cvres['test_roc_auc'].mean():.4f}")
        print(f"  {tag:32} cv_pr {rows[-1]['cv_pr_auc']:.4f}+/-{rows[-1]['cv_pr_sd']:.4f}  "
              f"val_roc {m['roc_auc']:.4f}  val_pr {m['pr_auc']:.4f}  brier {m['brier']:.4f}")

comp = pd.DataFrame(rows).sort_values('val_pr_auc', ascending=False)
comp.to_csv(OUT / 'model_comparison.csv', index=False)
print('\n' + comp.round(4).to_string(index=False))
print(f"\nno-skill PR-AUC on validation (the base rate): {yva.mean():.4f}")
print('majority-class baseline accuracy: '
      f"{majority_baseline(yva)['accuracy']:.4f}  (recall 0 -- it catches no defaulters)")

# ---------------------------------------------------------------------------
# 2. Encoding ablation: the one feature-engineering decision that paid.
# ---------------------------------------------------------------------------
# Arm A hands the raw frame to a scaler, so every categorical column goes in as
# an integer. For PAY_* that asserts the codes -2..8 are ordered by risk. Arm B
# collapses and one-hot encodes them first. Same model, split and seed. PAY_* is
# what the change targets, but EDUCATION, MARRIAGE and SEX are one-hot in arm B
# too, so the gain is for the categorical encoding as a whole.
print(f"\n{'='*78}\n2. ENCODING ABLATION -- are the PAY_* codes ordinal?\n{'='*78}")
naive = Pipeline([('scale', StandardScaler()),
                  ('clf', LogisticRegression(max_iter=2000, random_state=42))]).fit(Xtr, ytr)
collapsed = models.logistic().fit(Xtr, ytr)

abl = []
for label, fitted in [('A: PAY_* as integers (ordinal assumption)', naive),
                      ('B: PAY_* as collapsed categories', collapsed)]:
    m = evaluate(yva, fitted.predict_proba(Xva)[:, 1], 0.5, label)
    abl.append(m)
    experiments.record('2_encoding_ablation', label, {'model': 'logistic_l2'}, m)
ablation = comparison_table(abl)
delta = abl[1]['roc_auc'] - abl[0]['roc_auc']
ablation['roc_auc_gain_vs_A'] = [np.nan, delta]
ablation.to_csv(OUT / 'encoding_ablation.csv', index=False)
print(ablation.round(4).to_string(index=False))
print(f"\nre-encoding alone moves ROC-AUC by {delta:+.4f} -- no new features, no tuning")

# ---------------------------------------------------------------------------
# 3. Tuning the leading family.
# ---------------------------------------------------------------------------
# Scored on average_precision rather than accuracy or ROC-AUC: with a 22% base
# rate the useful question is how well the top of the ranking holds up, which is
# what the precision-recall curve measures.
print(f"\n{'='*78}\n3. TUNING -- RandomizedSearchCV on HistGradientBoosting, 30 draws\n{'='*78}")
search = RandomizedSearchCV(
    models.hist_gradient_boosting(),
    dict(clf__learning_rate=loguniform(0.01, 0.3),
         clf__max_leaf_nodes=randint(8, 64),
         clf__max_iter=randint(100, 500),
         clf__min_samples_leaf=randint(10, 120),
         clf__l2_regularization=loguniform(1e-3, 10.0)),
    n_iter=30, cv=cv, scoring='average_precision',
    random_state=42, n_jobs=-1, refit=True)
search.fit(Xtr, ytr)

m_tuned = evaluate(yva, search.predict_proba(Xva)[:, 1], 0.5)
default_cv = comp.loc[comp.model == 'hist_gradient_boosting', 'cv_pr_auc'].iloc[0]
default_val = comp.loc[comp.model == 'hist_gradient_boosting', 'val_pr_auc'].iloc[0]
print(f"best CV PR-AUC {search.best_score_:.4f}  (untuned {default_cv:.4f}, "
      f"{search.best_score_ - default_cv:+.4f})")
for k, v in search.best_params_.items():
    print(f"    {k:28} {v}")
print(f"  validation PR-AUC {m_tuned['pr_auc']:.4f}  (untuned {default_val:.4f}, "
      f"{m_tuned['pr_auc'] - default_val:+.4f})")
experiments.record('3_tuning', 'hist_gradient_boosting_tuned',
                   {k: str(v) for k, v in search.best_params_.items()}, m_tuned,
                   notes=f'cv_pr_auc={search.best_score_:.4f}, 30 draws')

params = {k.replace('clf__', ''): (float(v) if isinstance(v, float) else int(v))
          for k, v in search.best_params_.items()}
joblib.dump(search.best_estimator_, MODELS / 'hgb_tuned.joblib')
(MODELS / 'hgb_params.json').write_text(json.dumps(
    {'params': params,
     'cv_pr_auc': float(search.best_score_),
     'cv_pr_auc_untuned': float(default_cv),
     'val_pr_auc': float(m_tuned['pr_auc']),
     'val_pr_auc_untuned': float(default_val)}, indent=2))
print('\nwrote experiments/model_comparison.csv, experiments/encoding_ablation.csv,')
print('      models/hgb_tuned.joblib, models/hgb_params.json')

print(f"\n{'='*78}\nLEADERBOARD (validation, by PR-AUC)\n{'='*78}")
print(experiments.leaderboard('pr_auc').round(4).to_string(index=False))
