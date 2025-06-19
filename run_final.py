"""Final model, cutoff, SHAP, test evaluation, and the prediction file.

Order matters here. Everything that involves a *choice* -- hyperparameters,
dropping SEX, where to put the cutoff -- is decided on train and validation. The
test split is read once, at the end, to report what the chosen model scores on
data no decision was conditioned on.

Run:  ./.venv/bin/python run_compare.py && ./.venv/bin/python run_final.py
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

from src import experiments, models
from src.data import FINCLUB_VALIDATION, TARGET, load_finclub_validation
from src.evaluate import comparison_table, evaluate, majority_baseline, threshold_metrics
from src.features import collapse_codes
from src.split import load_split
from src.threshold import capacity_threshold, threshold_sweep

OUT = pathlib.Path('experiments'); OUT.mkdir(exist_ok=True)
MODELS = pathlib.Path('models'); MODELS.mkdir(exist_ok=True)
PREDS = pathlib.Path('outputs'); PREDS.mkdir(exist_ok=True)

CAPACITY = 0.20          # share of the book the review team can work
BREAK_EVEN = 0.0909      # cost_fp / (cost_fn + cost_fp) at a 10:1 ratio

tr, va = load_split('train'), load_split('val')
Xtr, ytr = tr.drop(columns=[TARGET]), tr[TARGET]
Xva, yva = va.drop(columns=[TARGET]), va[TARGET]

HP = json.loads((MODELS / 'hgb_params.json').read_text())['params']

# ---------------------------------------------------------------------------
# 1. Fit the final model.
# ---------------------------------------------------------------------------
# SEX is excluded. Sex cannot lawfully enter a credit decision in most
# jurisdictions, which settles it before any accuracy argument starts.
print('=' * 78)
print('1. FINAL MODEL -- HistGradientBoosting, tuned, without SEX')
print('=' * 78)
for k, v in HP.items():
    print(f'    {k:22} {v}')
final = models.hist_gradient_boosting(drop_sex=True, **HP).fit(Xtr, ytr)
p_va = final.predict_proba(Xva)[:, 1]
m_va = evaluate(yva, p_va)
print('validation: ' + '  '.join(f'{k} {m_va[k]:.4f}' for k in ('roc_auc', 'pr_auc', 'brier')))

# ---------------------------------------------------------------------------
# 2. Where to cut.
# ---------------------------------------------------------------------------
print(f"\n{'='*78}\n2. CUTOFF -- review the riskiest {CAPACITY:.0%}\n{'='*78}")
THRESHOLD = capacity_threshold(p_va, CAPACITY)

policies = [
    threshold_metrics(yva, p_va, 0.5) | {
        'policy': 'library default (0.5)'},
    threshold_metrics(yva, p_va, BREAK_EVEN) | {
        'policy': f'cost break-even at 10:1 ({BREAK_EVEN})'},
    threshold_metrics(yva, p_va, THRESHOLD) | {
        'policy': f'review capacity ({CAPACITY:.0%} of the book)'},
]
pol = pd.DataFrame(policies)[['policy', 'threshold', 'flagged_rate', 'precision',
                              'recall', 'f1', 'lift', 'tp', 'fp', 'fn', 'tn']]
pol.to_csv(OUT / 'threshold_policies.csv', index=False)
print(pol.round(4).to_string(index=False))

sweep = threshold_sweep(yva, p_va)
sweep.to_csv(OUT / 'threshold_sweep.csv', index=False)
print(f'\nchosen threshold {THRESHOLD:.4f}; full sweep in experiments/threshold_sweep.csv')

# ---------------------------------------------------------------------------
# 3. SHAP -- what the model is actually using.
# ---------------------------------------------------------------------------
print(f"\n{'='*78}\n3. SHAP -- global importance and two individual cases\n{'='*78}")
import shap

prep = final.named_steps['prep']
names = list(prep.get_feature_names_out())
Xva_t = prep.transform(collapse_codes(Xva))
explainer = shap.TreeExplainer(final.named_steps['clf'])

N_SHAP = 2000
sv = explainer.shap_values(Xva_t[:N_SHAP])
if isinstance(sv, list):
    sv = sv[1]
glob = (pd.DataFrame({'feature': names, 'mean_abs_shap': np.abs(sv).mean(axis=0)})
        .sort_values('mean_abs_shap', ascending=False).reset_index(drop=True))
glob.to_csv(OUT / 'shap_global.csv', index=False)
print(f'mean |SHAP| over {N_SHAP} validation rows, in log-odds -- top 12:')
print(glob.head(12).round(4).to_string(index=False))

base_logodds = float(np.ravel(explainer.expected_value)[0])
sub = p_va[:N_SHAP]
local = []
for tag, i in [('highest risk', int(np.argmax(sub))), ('lowest risk', int(np.argmin(sub)))]:
    c = pd.DataFrame({'feature': names, 'shap': sv[i]})
    top = c.reindex(c.shap.abs().sort_values(ascending=False).index).head(6)
    local.append({'case': tag, 'row': i, 'p_default': float(sub[i]),
                  'base_rate_from_model': float(1 / (1 + np.exp(-base_logodds))),
                  'drivers': top.to_dict('records')})
    print(f'\n--- {tag}: p(default) = {sub[i]:.4f} ---')
    print(top.round(4).to_string(index=False))
(OUT / 'shap_local.json').write_text(json.dumps(local, indent=2, default=str))

# ---------------------------------------------------------------------------
# 4. The test split, read once.
# ---------------------------------------------------------------------------
print(f"\n{'='*78}\n4. TEST SET\n{'='*78}")
te = load_split('test')
Xte, yte = te.drop(columns=[TARGET]), te[TARGET]
p_te = final.predict_proba(Xte)[:, 1]

results = comparison_table([
    majority_baseline(yte),
    evaluate(yte, p_te, THRESHOLD, f'HGB (no SEX) @ {CAPACITY:.0%} capacity cutoff'),
    evaluate(yte, p_te, 0.5, 'HGB (no SEX) @ 0.5 for reference'),
])
results.to_csv(OUT / 'final_test_results.csv', index=False)
print(results.round(4).to_string(index=False))

m_te = evaluate(yte, p_te, THRESHOLD)
gap = pd.DataFrame([
    {'split': 'validation', **{k: m_va[k] for k in ('roc_auc', 'pr_auc', 'brier')},
     **{k: v for k, v in threshold_metrics(yva, p_va, THRESHOLD).items()
        if k in ('precision', 'recall', 'f1')}},
    {'split': 'test', **{k: m_te[k] for k in ('roc_auc', 'pr_auc', 'brier',
                                              'precision', 'recall', 'f1')}},
])
gap.loc[len(gap)] = ['delta'] + list(gap.iloc[1, 1:] - gap.iloc[0, 1:])
gap.to_csv(OUT / 'generalisation_gap.csv', index=False)
print('\nvalidation vs test:')
print(gap.round(4).to_string(index=False))
experiments.record('4_final', 'hist_gradient_boosting (no SEX)',
                   {**HP, 'drop_sex': True}, m_te, split='test',
                   notes=f'capacity cutoff {THRESHOLD:.4f}')

# ---------------------------------------------------------------------------
# 5. The deliverable: predictions for the Finclub validation customers.
# ---------------------------------------------------------------------------
print(f"\n{'='*78}\n5. PREDICTIONS for the unlabelled validation set\n{'='*78}")
if not FINCLUB_VALIDATION.exists():
    # The Finance Club file is not public, so a fresh clone will not have it.
    print(f'{FINCLUB_VALIDATION} not found -- skipping the prediction file. '
          'Everything above still ran.')
else:
    fc = load_finclub_validation()
    p_fc = final.predict_proba(fc.drop(columns=['Customer_ID']))[:, 1]
    preds = pd.DataFrame({'Customer': fc['Customer_ID'],
                          'next_month_default': (p_fc >= THRESHOLD).astype(int)})
    preds.to_csv(PREDS / 'final_predictions.csv', index=False)
    pd.DataFrame({'Customer': fc['Customer_ID'],
                  'default_probability': p_fc.round(6)}).to_csv(
        PREDS / 'final_prediction_probabilities.csv', index=False)
    print(f'{len(preds)} rows, {preds.next_month_default.mean():.2%} flagged '
          f'at threshold {THRESHOLD:.4f}')

# ---------------------------------------------------------------------------
# 6. Persist the model and its operating configuration.
# ---------------------------------------------------------------------------
joblib.dump(final, MODELS / 'final_model.joblib')
(MODELS / 'config.json').write_text(json.dumps({
    'model': 'HistGradientBoostingClassifier (sklearn), tuned, uncalibrated',
    'artifact': 'final_model.joblib',
    'params': HP,
    'drop_sex': True,
    'threshold': THRESHOLD,
    'threshold_rule': f'flag the riskiest {CAPACITY:.0%} of the scored population',
    'capacity': CAPACITY,
    'trained_on': 'UCI default of credit card clients, train split',
}, indent=2))
print('\nwrote models/final_model.joblib, models/config.json,')
print('      outputs/final_predictions.csv, experiments/*.csv')
