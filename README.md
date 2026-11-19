# Credit Card Default Prediction

This is my solution to the Finance Club (IIT Roorkee) summer project problem statement: predict which credit-card customers will default next month, and decide which of them a bank should actually act on.

The final model is scikit-learn's `HistGradientBoostingClassifier`. On a held-out test set it reaches a **ROC-AUC of 0.78**. When it flags the riskiest 20% of customers, **57% of the flagged customers really default**, against 22% if you picked the same number at random.

Every number below comes from a file in `experiments/` or `outputs/`, written by `run_compare.py`, `run_final.py` and the notebooks.

---

## The data

I used the UCI *Default of Credit Card Clients* dataset (Yeh & Lien, 2009). It has 30,000 credit-card accounts from Taiwan, with six months of history (April–September 2005) and whether the customer defaulted in October 2005. **22.1% defaulted.**

The Finance Club handed out a modified copy of this dataset. In that copy, negative bill amounts were clipped to zero, amounts were slightly perturbed, and some ages were blanked. A negative bill means the customer overpaid, which is a useful low-risk signal, so I trained on the original instead. I still produced predictions for the club's unlabelled validation file, since that was the deliverable. The club's files aren't in this repo; they weren't mine to publish.

One thing that tripped me up early: column index 1 is the latest month. `PAY_0`, `BILL_AMT1` and `PAY_AMT1` are September, and **there is no `PAY_1` column**.

56 rows are exact copies of another row, so I removed them before splitting. Otherwise a row and its copy could end up on opposite sides of the split, and the test score would partly reward memorisation. Then I made a stratified 60/20/20 split:

| split | rows | default rate |
|---|---|---|
| train | 17,966 | 22.11% |
| validation | 5,989 | 22.12% |
| test | 5,989 | 22.11% |

Train is for fitting. Validation is for picking the model, its settings and the cutoff. Test was scored once, at the end.

## What the data showed

**Repayment status matters most, but its codes aren't a scale.** `PAY_0` is documented as −1 = paid on time, 1 = one month late, 2 = two months late, and so on. Codes −2 and 0 aren't documented at all, yet they cover 58.3% of customers.

| PAY_0 code | customers | default rate |
|---|---|---|
| −2 | 2,759 | 13.2% |
| −1 | 5,686 | 16.8% |
| 0 | 14,737 | 12.8% |
| 1 | 3,688 | 33.9% |
| 2 | 2,667 | 69.1% |
| 3 | 322 | 75.8% |

Code 0 is *safer* than code −1. If you feed these codes to a model as numbers, you're telling it risk goes up steadily from −2 to 8, and the data says otherwise at the low end. That observation drove the main feature decision.

**Credit limit works the way you'd expect.** It's the bank's own earlier judgement of the customer. The lowest-limit decile (10k–30k) defaults at 35.8%, and the highest (370k–1M) at 11.9%.

**Paying nothing is the warning sign.** I grouped customers by the share of their September bill they paid:

| share of September bill paid | customers | default rate |
|---|---|---|
| nothing | 3,630 | 39.6% |
| under 5% | 8,094 | 19.2% |
| 5–20% | 7,954 | 22.2% |
| 20–50% | 2,301 | 18.7% |
| 50–99% | 1,548 | 13.3% |
| all of it | 3,875 | 15.5% |

Above zero the bands are noisy, so this is a cliff rather than a slope.

**Demographics barely move the rate.** Between the large groups, the default rate differs by about 6 points at most.

## Features

I made one feature-engineering change and measured it.

- `PAY_0 … PAY_6` become categories (−2, −1, 0, 1, 2, 3+) and are one-hot encoded. Codes 3 and above are pooled because each one holds very few customers, and their default rates jump around rather than rising.
- `EDUCATION` becomes graduate / university / high_school / other, and `MARRIAGE` becomes married / single / other. The undocumented codes go into "other".
- Numeric columns are scaled for logistic regression only. Trees don't need it.

All of this lives inside a scikit-learn `Pipeline`, so during cross-validation the encoder and scaler are fit only on the training part of each fold.

To measure the change, I trained the same logistic regression twice on the same split:

| version | encoding | ROC-AUC | PR-AUC |
|---|---|---|---|
| A | categoricals as plain integers | 0.7305 | 0.5107 |
| B | categoricals collapsed and one-hot encoded | 0.7770 | 0.5549 |

That's **+0.046 ROC-AUC** with no new features and no tuning. Version B also one-hot encodes education, marriage and sex, so strictly this is the gain from the categorical encoding as a whole. Given how weak the demographics are, most of it should come from the repayment codes, but I haven't separated the two. I also only measured this on logistic regression.

## Class imbalance

With 22% defaults, the problem statement asks for imbalance to be handled. I tried `class_weight="balanced"`, the cheapest of the options it lists, on every model family:

| model | Brier, unweighted | Brier, balanced | change in validation PR-AUC |
|---|---|---|---|
| decision tree | 0.1361 | 0.1820 | +0.0072 |
| HistGradientBoosting | 0.1312 | 0.1736 | −0.0055 |
| LightGBM | 0.1321 | 0.1639 | +0.0012 |
| logistic regression | 0.1344 | 0.1812 | −0.0010 |
| random forest | 0.1322 | 0.1602 | −0.0014 |

The probabilities got worse for every family (a higher Brier score), and the ranking (PR-AUC) didn't really change. Weighting does catch more defaulters at a 0.5 cutoff: HistGradientBoosting goes from 38% to 63%. But that's only the effective cutoff moving, and I move the cutoff on purpose later anyway. So I kept the unweighted models.

I didn't use SMOTE. It creates synthetic defaulters by blending neighbouring rows, and on one-hot repayment codes that produces customers who are, say, 40% "two months late" and 60% "paid on time". I didn't test it, so that's reasoning rather than a result.

## Models

I compared five model types, each with and without class weights. I ran 5-fold stratified cross-validation on train, then fit on train and scored on validation. Unweighted results:

| model | CV PR-AUC | validation ROC-AUC | validation PR-AUC | Brier |
|---|---|---|---|---|
| HistGradientBoosting | 0.5418 ± 0.0164 | 0.7952 | 0.5782 | 0.1312 |
| random forest | 0.5471 ± 0.0228 | 0.7887 | 0.5742 | 0.1322 |
| LightGBM | 0.5391 ± 0.0155 | 0.7882 | 0.5707 | 0.1321 |
| logistic regression | 0.5361 ± 0.0241 | 0.7770 | 0.5549 | 0.1344 |
| decision tree (depth 6) | 0.5013 ± 0.0153 | 0.7654 | 0.5164 | 0.1361 |

The weighted variants are in `experiments/model_comparison.csv`.

I picked **HistGradientBoosting**, which was best on validation PR-AUC. To be fair to the random forest: it's slightly ahead in cross-validation, but the two are well within one standard deviation of each other, so they're effectively tied. LightGBM is the same kind of model and scored a little lower here, so it didn't justify an extra dependency. The single decision tree is clearly behind; that's what you pay for a model you can draw on one page.

Tuning: 30 random draws, scored on cross-validated PR-AUC. The best settings were learning rate 0.0148, 234 iterations, 16 leaves per tree, at least 81 rows per leaf, and L2 regularisation 0.027. Cross-validated PR-AUC rose from 0.5418 to 0.5488, while validation PR-AUC went from 0.5782 to 0.5769. In other words, tuning didn't buy much, because the defaults were already close. At this data size HistGradientBoosting's built-in early stopping is on by default and keeps 10% of the training rows aside to watch the loss. With these settings it never stopped early and used all 234 iterations.

I removed `SEX` from the final model, because sex can't be used in credit decisions in most places. On validation it cost nothing: ROC-AUC 0.7965 without it vs 0.7958 with it.

## Metrics

- **PR-AUC** is the main one. A review team works from the top of the ranked list, and PR-AUC measures how clean that top is. A model with no skill scores the base rate (0.22), not 0.5, so I always quote it next to the base rate.
- **ROC-AUC** for comparison with other work.
- **Brier score**, because it's the only one here that notices when probabilities are distorted rather than just mis-ordered. That's what caught the class-weight problem.
- **Precision and recall at the chosen cutoff**, because that's what the review team actually experiences.

Accuracy doesn't work here. Predicting "no default" for everyone gives 77.9% accuracy on the test set and catches nobody.

## Choosing the cutoff

The model gives a probability. The bank needs a yes or no. The two kinds of mistake cost different amounts:

- **Missed defaulter (false negative):** the bank loses part of the balance.
- **False alarm (false positive):** a good customer gets an unnecessary call or a lower limit. That costs staff time and some goodwill.

I compared three ways to set the cutoff on the validation set:

| rule | cutoff | share flagged | precision | recall |
|---|---|---|---|---|
| default 0.5 | 0.5000 | 11.7% | 70.0% | 37.1% |
| a miss costs 10× a false alarm | 0.0909 | 77.2% | 27.1% | 94.7% |
| **review the riskiest 20%** | 0.3176 | 20.0% | 57.8% | 52.2% |

- The 0.5 cutoff misses 833 of the validation defaulters.
- The cost rule (flag anyone with probability above 1/11) is correct on paper, but it flags 4,624 of 5,989 customers, which no team could work through. It also assumes the probabilities are calibrated, and I haven't checked that.
- On the validation sweep, F1 peaks around a cutoff of 0.25, which flags about a quarter of customers.

I went with **reviewing the riskiest 20%**. It assumes the team can handle about a fifth of the book each month. That 20% is my assumption; the problem statement doesn't give a number. Writing the rule as "the riskiest 20%" rather than a fixed probability means the queue size stays the same if the model is retrained.

The price is in the recall column: about half of all defaulters don't make the list. `experiments/threshold_sweep.csv` shows every other option.

## What drives the predictions (SHAP)

I ran SHAP's `TreeExplainer` on 2,000 validation customers. Values are in log-odds.

| feature | mean \|SHAP\| |
|---|---|
| PAY_0_2 (two months late in September) | 0.298 |
| PAY_2_2 (two months late in August) | 0.164 |
| LIMIT_BAL | 0.156 |
| PAY_AMT2 | 0.123 |
| PAY_AMT1 | 0.116 |
| BILL_AMT1 | 0.078 |

Repayment-status columns account for 50.3% of all attribution and payment amounts for 25.6%. The demographics left in the model (age, education, marriage) account for 2.9%. This matches the EDA, which is reassuring.

For individual customers:
- The **riskiest** one (probability 0.83) is pushed up mostly by being two months late in August and three or more months late in September.
- The **safest** one (probability 0.04) is pushed down by large recent payments.

That's the kind of reason a reviewer can write in a case note.

## Results on the test set

The test set wasn't used for any decision above and was scored once.

| | ROC-AUC | PR-AUC | accuracy | precision | recall | F1 |
|---|---|---|---|---|---|---|
| always predict "no default" | 0.5000 | 0.2211 | 0.7789 | — | 0.0000 | 0.0000 |
| **model, riskiest-20% cutoff** | **0.7815** | **0.5612** | 0.8063 | **0.5695** | 0.5076 | 0.5367 |
| model, 0.5 cutoff | 0.7815 | 0.5612 | 0.8217 | 0.6928 | 0.3474 | 0.4628 |

At the chosen cutoff the model flagged 19.7% of test customers, and 57% of those defaulted. That's 2.58× better than flagging the same number at random. Validation scored a bit higher (ROC-AUC 0.7965, PR-AUC 0.5795). That's expected, since validation was used to make choices.

The deliverable is `outputs/final_predictions.csv`: 5,016 Finance Club customers with `Customer` and `next_month_default` columns, and 916 (18.3%) flagged. Probabilities are in `outputs/final_prediction_probabilities.csv`. Most of these customers also appear in the UCI data I trained on, so treat this file as the deliverable and the test table as the real performance estimate.

## Limitations

- **One month ahead only.** The target is October 2005. It says nothing about a year out.
- **Taiwan, 2005.** A different market and a different time. The approach carries over; the fitted model doesn't.
- **No out-of-time test.** Every customer covers the same six months, so the test checks how the model does on new *customers*, not in a future *period*. A real bank would also validate on a later cohort.
- **Uncalibrated probabilities.** They rank customers well, but I haven't checked that a 0.3 means a 30% chance. The 20% rule only needs the ranking.
- **The 20% capacity is assumed.**
- **The 56 duplicate rows** might be data-entry errors or genuinely identical customers. There's no way to tell from the file.
- **A missing repayment code** (`PAY_*` empty) makes scoring fail. The UCI data has none, but new data could.

## Running it

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python run_compare.py     # downloads UCI, builds the splits, compares models, tunes
./.venv/bin/python run_final.py       # final model, cutoff, SHAP, test results, predictions
./.venv/bin/python -m pytest tests/ -q
```

`run_final.py` writes the prediction file only if the Finance Club validation file is at `data/raw/validate_dataset_final.csv`. Without it, the script skips that step and everything else still runs.

To re-run the notebooks:

```bash
./.venv/bin/python -m ipykernel install --user --name creditcard
./.venv/bin/jupyter execute --inplace notebooks/*.ipynb
```

```
src/             data loading, split, features, models, metrics, cutoff, experiment log
run_compare.py   model comparison, class weights, encoding test, tuning
run_final.py     final model, cutoff, SHAP, test results, predictions
notebooks/       01 EDA · 02 features and split · 03 model comparison · 04 cutoff and SHAP · 05 results
experiments/     every table quoted above, plus log.jsonl
outputs/         prediction file
tests/           data and cutoff checks
```
