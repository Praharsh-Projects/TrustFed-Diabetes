# Results

## Chapter Introduction

This chapter presents the final empirical results of **TrustFed-Diabetes** on the full CDC Diabetes Health Indicators dataset. The chapter is intentionally organized around the final dashboard and the exported full-CDC result package, because the main contribution of the project is not a single score but a reproducible evidence surface that combines predictive performance, calibrated probabilities, subgroup auditing, explanation summaries, explanation stability, and federated communication tracking.

The chapter therefore follows the same evidence order used in the dashboard: `Showcase`, `Model Comparison`, `Calibration`, `Explainability`, `Fairness`, `Federated Training`, and `Full Study Comparison`. Every key dashboard term is kept exactly as it appears in the interface. At first mention, each term is explained in very simple language so that the chapter can be read without the dashboard being open.

The chapter is also deliberately more detailed than a short article-style results section. For each major chart, the text explains what the chart shows, how to read it, what the full-CDC results show, how the pattern changes across settings, what the chart implies for the thesis, and what the chart does not prove. This makes the chapter long, but it also makes the result package easier to defend in a viva, because each graph is tied directly to the question it is meant to answer.

All quantitative claims in this chapter come from the final full-CDC evidence package:

- [results/full_cdc_polished_summary](C:/Users/praha/Desktop/Joshna/results/full_cdc_polished_summary)
- [results/full_cdc_visual_summary](C:/Users/praha/Desktop/Joshna/results/full_cdc_visual_summary)
- [thesis_assets/full_cdc_polished/tables](C:/Users/praha/Desktop/Joshna/thesis_assets/full_cdc_polished/tables)

No numbers in this chapter come from the older capped CDC development package.

## 1. Result Package and Evaluation Basis

### 1.1 Full CDC scope

The final result package uses the full CDC Diabetes Health Indicators dataset, which contains approximately **253,680 rows**. The final evaluation retains the project's existing split policy: approximately **60% training**, **20% calibration**, and **20% test**. This means that the full-CDC evaluation uses roughly **152,208 training cases**, **50,736 calibration cases**, and **50,736 test cases**. The calibration split is used for probability adjustment and threshold selection, while the test split is used for final evaluation.

This point matters because some dashboard charts show counts that are much smaller than 253,680. Those charts are not wrong. They show either the held-out full-CDC **test fold** or a normalized view of the test-fold distribution. The confusion matrices, for example, are meant to report out-of-sample decisions on the held-out test set rather than the entire dataset mixed together. That is the correct methodology.

### 1.2 Broad audit package versus visual verification package

The final full-CDC result package contains two linked but slightly different evidence sources.

The first source is the **broad audit package** in [results/full_cdc_polished_summary](C:/Users/praha/Desktop/Joshna/results/full_cdc_polished_summary). This is the main scientific comparison package. It contains the broad matrix across `Centralized`, `Local-Only`, `Average Local-Only`, and `Federated` settings, together with summary metrics, calibration summaries, fairness summaries, threshold summaries, round-wise federated summaries, and communication summaries. This package reported:

- `runs_found = 80`
- `metric_rows = 7336`
- `summary_rows = 740`

The second source is the **visual verification package** in [results/full_cdc_visual_summary](C:/Users/praha/Desktop/Joshna/results/full_cdc_visual_summary). This package stores the saved prediction-backed summaries needed for confusion matrices, ROC curves, PR curves, score distributions, and other chart types that depend on prediction-level artifacts. This package reported:

- `runs_found = 12`
- `prediction_rows = 9,741,312`

The relationship between the two packages should be stated clearly. The broad audit package is the source of the final ranking and comparison claims. The visual verification package is the source of the prediction-backed charts. When the two packages differ in scope, the chapter treats the prediction-backed view as an illustration and the broad audit package as the source of the final headline comparison claim.

### 1.3 How to read the dashboard selections

Because this chapter mirrors the dashboard, the dashboard filter terms are defined here.

`Data Source` means the dataset currently being analyzed. In the final main evidence package, the active source is the CDC Diabetes Health Indicators dataset.

`Run Type` means the training setup being compared. `Centralized` means a pooled-data reference model trained on all eligible training data together. `Local-Only` means a single site is trained independently. `Average Local-Only` means the average summary across those independent local site models. `Federated` means that sites keep their raw data local and exchange only model updates.

`Model Family` means the predictive model type. In the full-CDC broad audit package, this includes logistic regression, random forest, gradient boosting, decision tree, shallow neural network, and XGBoost, with federated training applied to logistic and shallow neural network families.

`Algorithm` means the federated update rule. `FedAvg` averages site updates. `FedProx` adds a proximal penalty so that each site's update stays closer to the shared model, which can help when site data differ more strongly.

`Probability Adjustment` means the post-training calibration choice. `None` means the model's raw risk scores are used directly. `Sigmoid` and `Isotonic` are standard post-hoc calibration methods. `Global Isotonic`, `Global Sigmoid`, `Federated Isotonic`, and `Federated Sigmoid` refer to the saved calibrated probability variants produced for federated outputs.

`Number Of Sites` means the number of simulated client sites. The final full-CDC package uses `3` and `5`.

`Data Split Type` means whether client data were partitioned as `iid` or `non_iid`. `iid` means the site distributions are made similar. `non_iid` means they are intentionally different.

`Data Difference Level` is the dashboard's human-readable name for the heterogeneity control `alpha`. Lower values mean stronger differences between sites. Higher values mean milder differences.

`Decision Rule` means the threshold used to convert a predicted diabetes risk into a binary prediction. `fixed_0p5` means a fixed 50% cutoff. `calib_f1_optimal` means the threshold was chosen on the calibration split to maximize `Balanced Detection Score (F1)`.

### 1.4 How to read the metric labels

The dashboard keeps the metric names explicit, but they are still worth explaining in plain language.

`Ranking Quality (AUROC)` measures how well a model ranks higher-risk cases above lower-risk cases across all possible thresholds. A higher value is better. It is a ranking measure, not a fixed-threshold accuracy measure.

`Positive-Case Ranking (PR AUC)` measures how well the model concentrates true positive cases above false positives when positive cases are relatively less frequent. This metric is especially useful for CDC because diabetes-positive cases are not the majority class.

`Balanced Detection Score (F1)` is the harmonic mean of precision and recall. In simple terms, it favors decision rules that both catch positive cases and avoid too many false alarms.

`Accuracy` is the overall proportion of correct yes/no predictions at the chosen decision rule.

`Precision` is the proportion of predicted positive cases that are truly positive.

`Recall` is the proportion of truly positive cases that are successfully identified.

`Specificity` is the proportion of truly negative cases that are correctly identified as negative.

`Balanced Accuracy` averages recall and specificity. It is useful when one class is more common than the other.

`Risk Match Error (ECE)` measures how closely predicted percentages match observed event frequencies. Lower is better.

`Probability Error (Brier)` measures the average squared probability error. Lower is better.

`Confidence Penalty (Log Loss)` penalizes wrong probabilities more sharply, especially when the model is confident and wrong. Lower is better.

`Positive Prediction Rate` is how often a subgroup is predicted positive under the current decision rule.

`Positive Rate Gap` is the subgroup disparity version of `Demographic Parity Difference`.

`Error Gap Between Groups` is the subgroup disparity version of `Equalized Odds Difference`.

### 1.5 Reproducibility basis

The final full-CDC result package was produced from a reproducible configuration-driven pipeline rather than from manual notebook edits. The broad audit run covered the CDC dataset without a `max_rows` cap, used five seeds, included both `3` and `5` sites, included `iid` plus three `non_iid` settings, compared `Centralized`, `Local-Only`, `Average Local-Only`, and `Federated` rows, and retained both the fixed threshold and the F1-tuned threshold.

The core commands used to build the final package were:

```powershell
py scripts/run_matrix.py --config configs/full_cdc_polished.json
py scripts/aggregate_results.py --results-dir results/full_cdc_polished_runs --output-dir results/full_cdc_polished_summary
py scripts/run_matrix.py --config configs/full_cdc_visual_verify.json
py scripts/aggregate_results.py --results-dir results/full_cdc_visual_runs --output-dir results/full_cdc_visual_summary
py scripts/export_thesis_assets.py --summary-dir results/full_cdc_polished_summary --output-dir thesis_assets/full_cdc_polished
```

These commands matter in the Results chapter because the contribution of the thesis is not only a dashboard or only a classifier. It is a reproducible evaluation framework in which the tables, figures, and narrative all come from saved outputs.

### 1.6 Computation and Reading Guide

This subsection explains the notation, aggregation rules, threshold rules, and split conventions used throughout the remainder of the chapter. Its purpose is to make the later chart sections readable without forcing the reader to reverse-engineer the code or the exported CSV files.

#### Notation and symbols

The chapter uses a small set of recurring symbols. `y_true` means the true binary label, with `1` representing the positive class and `0` representing the negative class. `y_pred` means the thresholded binary prediction. `p_i` means the predicted probability for case `i`. `t` means the decision threshold used to convert a predicted probability into a yes/no prediction. `B_m` means the `m`-th probability bin in calibration analysis. `g` means a subgroup such as an age category, BMI category, or sex category. `n` means the number of rows contributing to the summary being reported.

The confusion-matrix notation also follows the standard convention. `TP` means true positives, `TN` means true negatives, `FP` means false positives, and `FN` means false negatives. In simple terms, these four quantities count all correct and incorrect binary decisions after a threshold has been applied.

#### How metric values are aggregated

The dashboard is not plotting one raw run at a time unless the chart is explicitly based on a single saved explanation or a single saved prediction-backed visual. Most dashboard summaries are aggregated across grouped rows. The aggregation code uses grouped means, sample standard deviations, and 95% confidence intervals.

For a grouped metric value `x_1, x_2, ..., x_n`, the reported mean is:

`mean = (1 / n) * sum_(i=1 to n) x_i`

The reported sample standard deviation is:

`std = sqrt( sum_(i=1 to n) (x_i - mean)^2 / (n - 1) )`

The 95% confidence interval used in the exported summaries is:

`mean +/- 1.96 * std / sqrt(n)`

This is the exact rule implemented in [metrics.py](C:/Users/praha/Desktop/Joshna/src/fl_diabetes/metrics.py) through `aggregate_with_confidence_intervals` and `_ci95`. If only one grouped row is available, the confidence interval collapses to the mean itself. In practical terms, this means the confidence interval is a summary of run-to-run variation within the grouped slice, not a claim of clinical uncertainty.

#### How thresholds are selected

The dashboard uses two `Decision Rule` settings.

`fixed_0p5` means that the positive class is predicted whenever `p_i >= 0.5`. This is the simplest thresholding rule and acts as a transparent baseline.

`calib_f1_optimal` means that a threshold is chosen from a candidate set to maximize `Balanced Detection Score (F1)` on the **calibration split only**. This is important enough to state explicitly: the test split is not used to choose the threshold. The test split is only used for final evaluation after the threshold has already been chosen.

In the code, the candidate thresholds come from a combined grid-and-quantile set: a fixed grid from 0.05 to 0.95, calibration-probability quantiles at the same 19 points, and the explicit value `0.5`. The candidates are clipped into `(0, 1)`, rounded to four decimals, and then evaluated on the calibration split. If two thresholds achieve the same F1, the tie is broken in favor of the threshold closest to `0.5`. This tie-break rule is written directly in [metrics.py](C:/Users/praha/Desktop/Joshna/src/fl_diabetes/metrics.py) inside `select_decision_threshold`.

#### How to interpret split-specific visuals

Some of the most common reader confusions come from mixing together different split roles.

The confusion matrix is built from the **held-out test fold only**. It does not sum to the full dataset size because it is not supposed to include training or calibration rows. In the final full-CDC package, that means the total count is about **50,736**, not 253,680.

The score-distribution chart is also test-based, but it is not a raw-count chart. It is a **normalized within-class distribution**. That means the positive and negative class distributions are each scaled so that their shapes can be compared fairly. The chart therefore shows how scores spread, not how many raw rows are in each class.

The broad audit package and the visual verification package also play different roles. The broad audit package is the source of the final best-row claims. The visual verification package is the source of prediction-backed charts such as confusion matrices, ROC curves, PR curves, score distributions, and representative explanation snapshots. When the visual package is narrower than the broad package, the Results chapter states that distinction explicitly instead of silently pretending they are identical.

### 1.7 Metric formulas and implementation basis

This subsection defines the dashboard metrics formally. Unless stated otherwise, probability-based metrics are computed from the **test probabilities**, while threshold-based metrics are computed from the **test labels and thresholded test predictions**. Where the implementation relies on a library-backed function, that is stated directly so that the reader knows whether the metric is handwritten or delegated to a standard metric implementation.

`Accuracy` is the overall fraction of correct binary predictions:

`Accuracy = (TP + TN) / (TP + TN + FP + FN)`

`Precision` is the fraction of predicted positives that are correct:

`Precision = TP / (TP + FP)`

`Recall` is the fraction of true positives that are found:

`Recall = TP / (TP + FN)`

`Specificity` is the fraction of true negatives that remain negative:

`Specificity = TN / (TN + FP)`

`Balanced Accuracy` is the arithmetic mean of recall and specificity:

`Balanced Accuracy = (Recall + Specificity) / 2`

`Balanced Detection Score (F1)` is the harmonic mean of precision and recall:

`Balanced Detection Score (F1) = 2 * Precision * Recall / (Precision + Recall)`

`Ranking Quality (AUROC)` is the area under the receiver operating characteristic curve computed from the test probabilities. In the implementation, this is metric-library-backed through `roc_auc_score` in scikit-learn. In simple language, it asks whether truly positive cases are ranked above truly negative cases across all thresholds.

`Positive-Case Ranking (PR AUC)` is the area under the precision-recall curve, implemented through scikit-learn's `average_precision_score`. It emphasizes how well the model concentrates truly positive cases near the top of the ranking.

`Probability Error (Brier)` is the mean squared probability error:

`Brier = (1 / n) * sum_(i=1 to n) (p_i - y_true,i)^2`

`Confidence Penalty (Log Loss)` is the negative average log-likelihood of the true labels under the predicted probabilities:

`Log Loss = -(1 / n) * sum_(i=1 to n) [ y_true,i * log(p_i) + (1 - y_true,i) * log(1 - p_i) ]`

In the implementation, Brier and Log Loss use scikit-learn's `brier_score_loss` and `log_loss`.

`Risk Match Error (ECE)` is the expected calibration error. The code uses the exact 10-bin formulation:

`ECE = sum_(m=1 to 10) (|B_m| / n) * | observed_event_rate(B_m) - avg_predicted_risk(B_m) |`

This formulation is implemented directly in [metrics.py](C:/Users/praha/Desktop/Joshna/src/fl_diabetes/metrics.py) by `expected_calibration_error`. Probabilities are clipped into `(1e-6, 1 - 1e-6)` before binning, and only non-empty bins contribute to the sum.

`Positive Prediction Rate` is the average thresholded predicted-positive label in a subgroup:

`Positive Prediction Rate = (1 / n_g) * sum_(i in g) y_pred,i`

`Positive Rate Gap` is the largest subgroup difference in positive prediction rate:

`Positive Rate Gap = max_g PositivePredictionRate(g) - min_g PositivePredictionRate(g)`

This corresponds to the exported `Demographic Parity Difference`.

`Error Gap Between Groups` is the larger of the true-positive-rate gap and the false-positive-rate gap across subgroups:

`Error Gap Between Groups = max( max_g TPR(g) - min_g TPR(g), max_g FPR(g) - min_g FPR(g) )`

This corresponds to the exported `Equalized Odds Difference`.

`Explanation Stability` is the Spearman-style rank correlation between successive rounds' absolute feature-importance vectors. In the implementation, absolute scores are ranked first and then correlated. In simple language, it measures whether the feature ranking changes dramatically from one round to the next.

`Cross-Client Top-k Overlap` is the fraction of top-k features shared by two clients:

`Cross-Client Top-k Overlap = |TopK(left) intersect TopK(right)| / k`

`Communication bytes` is the amount of model-update traffic sent in a single round. `Cumulative communication bytes` is the cumulative sum of those bytes over rounds. These values are saved directly in the communication summaries rather than being inferred from the plots.

### 1.8 Chart construction and source-table conventions

The dashboard charts are built from exported CSV summaries rather than from live training objects. This distinction is important because the Results chapter is meant to be reproducible from saved outputs.

The broad audit package provides the main grouped tables:

- [dashboard_metrics.csv](C:/Users/praha/Desktop/Joshna/results/full_cdc_polished_summary/dashboard_metrics.csv)
- [dashboard_calibration.csv](C:/Users/praha/Desktop/Joshna/results/full_cdc_polished_summary/dashboard_calibration.csv)
- [dashboard_fairness.csv](C:/Users/praha/Desktop/Joshna/results/full_cdc_polished_summary/dashboard_fairness.csv)
- [dashboard_rounds.csv](C:/Users/praha/Desktop/Joshna/results/full_cdc_polished_summary/dashboard_rounds.csv)
- [dashboard_shap.csv](C:/Users/praha/Desktop/Joshna/results/full_cdc_polished_summary/dashboard_shap.csv)
- [dashboard_thresholds.csv](C:/Users/praha/Desktop/Joshna/results/full_cdc_polished_summary/dashboard_thresholds.csv)
- [dashboard_showcase_metrics.csv](C:/Users/praha/Desktop/Joshna/results/full_cdc_polished_summary/dashboard_showcase_metrics.csv)
- [dashboard_score_ceiling.csv](C:/Users/praha/Desktop/Joshna/results/full_cdc_polished_summary/dashboard_score_ceiling.csv)

The visual verification package provides the prediction-backed plot sources:

- [dashboard_curves.csv](C:/Users/praha/Desktop/Joshna/results/full_cdc_visual_summary/dashboard_curves.csv)
- [dashboard_confusion.csv](C:/Users/praha/Desktop/Joshna/results/full_cdc_visual_summary/dashboard_confusion.csv)
- [dashboard_local_explanations.csv](C:/Users/praha/Desktop/Joshna/results/full_cdc_visual_summary/dashboard_local_explanations.csv)
- [dashboard_stability.csv](C:/Users/praha/Desktop/Joshna/results/full_cdc_visual_summary/dashboard_stability.csv)

Unless a chart explicitly shows direct confusion-cell counts or a representative local explanation, the dashboard normally plots **grouped means** rather than single raw rows. Where confidence intervals are available, they come from the grouped summary tables described above. Where the chapter discusses a specific visual chart that does not use confidence bands, the relevant subsection states whether the plotted value is a mean, a normalized share, or a direct count.

### 1.9 Reader clarification notes

Several misunderstandings are likely enough that they are worth addressing before the chart-by-chart walkthrough.

**Why the confusion matrix does not sum to 253,680.** The confusion matrix is based on the held-out test fold only. It therefore sums to the test-fold size, not to the full dataset size.

**Why the score-distribution chart does not show raw full-data counts.** The score-distribution view uses normalized within-class shares so that the shapes of the positive and negative score distributions can be compared fairly.

**Why the best federated headline row may differ from the federated visual chart row.** The broad audit package determines the final best-row claims. The visual verification package determines which slices have saved prediction-backed charts. When the visual package is narrower, the chapter uses the visual slice as an illustration and states that explicitly.

**Why AUROC can stay almost unchanged while calibration improves a great deal.** AUROC measures ranking quality, while calibration measures whether the risk percentages themselves match reality. A model can rank cases almost identically before and after calibration even if its probabilities become much more trustworthy.

**Why `fixed_0p5` and `calib_f1_optimal` can make the same model look very different.** The decision threshold changes the operating point dramatically. The trained model is the same, but the deployed yes/no rule changes recall, specificity, precision, and subgroup gaps.

## 2. Showcase

The `Showcase` page is the dashboard's headline page. It is designed to answer the first practical question a reader asks: *What is the clearest top-level story of the project once the full CDC data have been processed?* The page therefore combines headline cards with three main evidence surfaces: case classification, score distribution, and decision-rule tradeoff.

| Chart / Card | Source file | Primary grouping | Primary value(s) | Split basis | Aggregation meaning |
| --- | --- | --- | --- | --- | --- |
| `Best Centralized Model`, `Best Federated Model`, `How Close Federated Gets`, `Best Probability-Quality Setup`, `Best Low False-Positive Setup`, `Best Probability Adjustment`, `Best Overall Evidence Slice` | `dashboard_showcase_metrics.csv`, `dashboard_score_ceiling.csv`, thesis summary tables | selected `run_type`, `model`, `algorithm`, `calibration`, `threshold_strategy`, and scenario metadata | grouped means for AUROC, PR AUC, F1, ECE, Brier, specificity, and chosen thresholds | test metrics, with calibration-derived thresholds where applicable | headline cards summarize grouped means across matching saved runs |
| `How The Best Models Classify Cases` | `dashboard_confusion.csv` | selected showcase slice and confusion cell | mean cell count | held-out test fold only | each cell is the grouped mean confusion count for the visual slice |
| `How Risk Scores Spread Across Classes` | `dashboard_curves.csv` | selected showcase slice, class label, and score bin | normalized within-class share | held-out test fold only | each line or density trace shows class-wise normalized score mass |
| `What Changes When We Raise Or Lower The Decision Rule` | `dashboard_thresholds.csv` | selected slice, threshold, and `group_feature` | grouped mean F1, precision, recall, specificity, `Positive Rate Gap`, and `Error Gap Between Groups` | threshold sweep over calibration-derived candidate thresholds | each trace is a grouped mean metric value by threshold |

### 2.1 Showcase cards

#### `Best Centralized Model`

`Best Centralized Model` identifies the strongest pooled-data reference row under the current full-CDC package. In simple language, it is the best fully pooled benchmark available in the study.

In the final full-CDC package, the dashboard's `Best Centralized Model` is `Centralized / XGBoost / None / calib_f1_optimal`. Its headline values are:

- `Ranking Quality (AUROC) = 0.830`
- `Positive-Case Ranking (PR AUC) = 0.430`
- `Balanced Detection Score (F1) = 0.470`
- `Accuracy = 0.815`
- `Precision = 0.391`
- `Recall = 0.589`
- `Specificity = 0.851`
- `Balanced Accuracy = 0.720`

This card matters because it defines the performance upper-bound reference used throughout the thesis. The final interpretation of the project does not depend on Federated outperforming this row. Instead, the important question is how close Federated can get while preserving data locality and supporting a richer trustworthiness audit.

#### `Best Federated Model`

`Best Federated Model` identifies the strongest federated row chosen for the dashboard headline view. In the final dashboard summary, that headline row is `Federated / FedAvg / Shallow Neural Network / Federated Isotonic / calib_f1_optimal` with `3` sites, `iid`, and `alpha = 0.5`.

Its headline values are:

- `Ranking Quality (AUROC) = 0.826`
- `Positive-Case Ranking (PR AUC) = 0.416`
- `Balanced Detection Score (F1) = 0.467`
- `Risk Match Error (ECE) = 0.004`
- `Probability Error (Brier) = 0.098`

This card shows that the strongest federated decision slice remains very close to the centralized reference. It does not prove that Federated is universally better than Centralized. What it proves is that, on the full CDC data, the gap can remain small enough to make Federated technically credible as a privacy-preserving decision-support framework.

#### `How Close Federated Gets`

`How Close Federated Gets` is the gap card. It does not present a raw model score. Instead, it reports the difference between the best dashboard `Centralized` and `Federated` headline slices.

For the final full-CDC package, the headline gap is approximately:

- `Ranking Quality (AUROC)` gap: `-0.004`
- `Balanced Detection Score (F1)` gap: `-0.003`

The negative sign means that the federated slice trails the centralized reference slightly. The key point is that the gap is small, not that it disappears. This card supports the thesis claim that Federated can remain near the pooled-data reference without requiring raw-data pooling.

#### `Best Probability-Quality Setup`

`Best Probability-Quality Setup` identifies the strongest row for trustworthy risk percentages rather than raw ranking alone. In plain language, it asks: *Which setup gives the most believable predicted probabilities?*

The strongest exported probability-quality row is `Federated / FedAvg / Logistic / Global Isotonic / calib_f1_optimal` with `5` sites, `non_iid`, and `alpha = 1.0`. Its main values are:

- `Risk Match Error (ECE) = 0.003`
- `Probability Error (Brier) = 0.099`
- `Confidence Penalty (Log Loss) = 0.319`
- `Ranking Quality (AUROC) = 0.821`
- `Balanced Detection Score (F1) = 0.457`

This row matters because it shows that the most trustworthy risk percentages do not necessarily come from the absolute top ranking row. The thesis therefore benefits from explicitly separating "best ranking" and "best calibrated probability" findings.

#### `Best Low False-Positive Setup`

`Best Low False-Positive Setup` identifies the operating point that keeps false positive predictions as low as possible. In simple language, this card answers the question: *If the application is especially worried about falsely flagging non-diabetic cases, which setup is strongest?*

The exported low-false-positive recommendation is `Federated / FedProx / Logistic / Federated Isotonic / fixed_0p5` with `5` sites, `non_iid`, and `alpha = 0.5`. Its key values are:

- `Specificity = 0.994`
- `False-Positive Rate = 0.006`

This card is important because it shows that different decision goals lead to different preferred settings. The strongest low-false-positive setting is not the same as the strongest F1 setting.

#### `Best Probability Adjustment`

`Best Probability Adjustment` summarizes which probability-adjustment family gives the strongest calibrated output. In the current dashboard logic, the answer is effectively the probability-adjustment row with the best combination of low `Risk Match Error (ECE)` and competitive predictive quality.

In the full-CDC result package, the strongest adjustment family is the **Isotonic** family, especially `Global Isotonic` for federated logistic outputs and standard `Isotonic` for the strongest centralized rows. This result is visible again in the detailed calibration section, where raw `None` probabilities retain strong AUROC but much worse calibration error than the isotonic variants.

#### `Best Overall Evidence Slice`

`Best Overall Evidence Slice` is the dashboard's summary card that chooses the strongest overall headline row across the best centralized, federated, and probability-quality representatives. In the current logic, it prefers stronger `Balanced Detection Score (F1)`, then stronger `Ranking Quality (AUROC)`, then lower `Risk Match Error (ECE)`.

Under that rule, the final `Best Overall Evidence Slice` is still the `Centralized / XGBoost / None / calib_f1_optimal` row. This is expected. It confirms that pooled-data training remains the raw upper bound for overall prediction quality in the current study.

#### How these cards were computed and plotted

The showcase cards are built primarily from [dashboard_showcase_metrics.csv](C:/Users/praha/Desktop/Joshna/results/full_cdc_polished_summary/dashboard_showcase_metrics.csv), with supporting rows from [dashboard_score_ceiling.csv](C:/Users/praha/Desktop/Joshna/results/full_cdc_polished_summary/dashboard_score_ceiling.csv) and the exported best-row thesis tables in [thesis_assets/full_cdc_polished/tables](C:/Users/praha/Desktop/Joshna/thesis_assets/full_cdc_polished/tables). The `Best Centralized Model` and `Best Federated Model` cards come from grouped summary rows ordered by the dashboard's showcase selection logic. `How Close Federated Gets` is computed by subtracting the selected federated headline row from the selected centralized headline row on the key summary metrics. `Best Probability-Quality Setup` comes from the best calibration-oriented grouped row, and `Best Low False-Positive Setup` comes from the row that minimizes the false-positive rate while preserving the reported operating-point metadata. These cards are therefore not hand-picked screenshots; they are rendered from exported grouped tables.

[Insert Figure R1 here: Showcase cards and headline values]

Figure title: `Showcase` overview.

Caption: The `Showcase` page summarises the strongest centralized, federated, probability-quality, and low false-positive results in the full-CDC package.

Why this figure belongs in the Results chapter: It gives the reader the project's top-line evidence before the more detailed chart-by-chart audit.

### 2.2 `How The Best Models Classify Cases`

[Insert Figure R2 here: How The Best Models Classify Cases]

Figure title: `How The Best Models Classify Cases`.

Caption: Side-by-side confusion matrices show how the selected `Centralized` and `Federated` showcase slices classify the held-out full-CDC test cases.

Why this figure belongs in the Results chapter: It turns the headline card values into concrete counts of true positives, false positives, true negatives, and false negatives.

#### What this chart shows

`How The Best Models Classify Cases` is the confusion-matrix chart. The x-axis shows the predicted class (`Pred 0` and `Pred 1`), and the y-axis shows the true class (`True 0` and `True 1`). Each cell contains the average count for one outcome type. The four cells therefore correspond to true negatives, false positives, false negatives, and true positives.

This chart is displayed on the full-CDC held-out test fold, not on the entire dataset. That is why the counts sum to approximately **50,736** rather than 253,680.

#### How to read this chart

The first thing to inspect is the lower-left cell, which is the true-negative count. On CDC, this cell is expected to be large because the non-diabetic class is more common. The next important cells are the upper-right cell, which shows true positives, and the two off-diagonal cells, which show mistakes. A stronger decision setting will either increase the true-positive cell without letting the false-positive cell expand too sharply, or will lower the false-positive cell without collapsing the true-positive cell too severely.

#### What the full-CDC results show in this chart

For the showcase `Centralized` slice, the full-CDC held-out test fold is classified with approximately:

- `TN = 36,968`
- `FP = 6,699`
- `FN = 2,847`
- `TP = 4,222`

These counts match the dashboard's reported `Accuracy = 0.815`, `Precision = 0.391`, `Recall = 0.589`, and `Specificity = 0.851`. The central pattern is easy to read: the `Centralized` XGBoost row catches a meaningful number of positive cases, but it still makes a visible number of false-positive decisions, which is expected when the threshold is tuned for F1 rather than for maximal specificity.

For the prediction-backed federated visual, the dashboard uses the closest saved full-CDC federated visual slice because the visual verification package is narrower than the broad audit package. In the saved full-CDC federated logistic visual slice, the same test-fold scale is visible, with roughly:

- `TN = 36,330`
- `FP = 7,337`
- `FN = 2,778`
- `TP = 4,291`

The important pattern is that the federated visual remains on the same scale of errors and correct calls as the centralized visual. The federated side does not collapse into a visibly different error regime.

#### How this chart changes under different settings

This chart changes materially with the `Decision Rule`. For the best centralized XGBoost row under `None`, moving from `fixed_0p5` to `calib_f1_optimal` raises `Balanced Detection Score (F1)` from **0.324** to **0.470** and raises `Recall` from **0.411** to **0.589**, but it reduces `Specificity` from **0.872** to **0.851**. In other words, the F1-tuned decision rule catches more positive cases, but it accepts more false positives.

The same pattern is even more dramatic on the federated MLP row. Under `fixed_0p5`, `Precision` is high at about **0.597** and `Specificity` is very high at about **0.989**, but `Recall` falls to only **0.095** and `F1` falls to about **0.158**. Under `calib_f1_optimal`, the same model moves to `Recall` about **0.619**, `Specificity` about **0.832**, and `F1` about **0.466**. This is exactly the kind of threshold tradeoff the thesis is meant to surface.

The chart also changes across `Centralized` and `Federated` settings. The centralized reference keeps a modest performance edge, but the federated confusion structure remains comparable rather than qualitatively weaker.

#### What this chart implies for the thesis

This chart supports the thesis claim about decision-support readiness. It shows that the project is not only ranking cases; it is also evaluating concrete operating points. The visual comparison demonstrates that `Federated` remains close enough to `Centralized` to be practically discussable, while still letting the thesis inspect recall-sensitive and specificity-sensitive decision rules explicitly.

#### What this chart does not prove

This chart does not prove causal validity, clinical utility, or fairness by itself. A confusion matrix only reports classification outcomes at one chosen threshold. It cannot explain whether the probabilities are well calibrated, whether subgroup disparities are acceptable, or whether the selected threshold is clinically preferable.

#### How this chart was computed and plotted

This chart is built from [dashboard_confusion.csv](C:/Users/praha/Desktop/Joshna/results/full_cdc_visual_summary/dashboard_confusion.csv), which is sourced from saved prediction artifacts in the full-CDC visual verification package. The grouping is the selected showcase slice plus the confusion `cell` label (`TN`, `FP`, `FN`, `TP`). The plotted value is `count_mean`, so the displayed numbers are grouped mean cell counts rather than one raw run's counts. The split basis is the held-out **test fold only**. No training or calibration rows are included. The centralized side uses the saved full-CDC centralized prediction-backed visual, while the federated side uses the closest available saved federated visual slice when the broad headline row itself does not have a matching prediction-backed artifact in the visual package.

### 2.3 `How Risk Scores Spread Across Classes`

[Insert Figure R3 here: How Risk Scores Spread Across Classes]

Figure title: `How Risk Scores Spread Across Classes`.

Caption: The score-distribution chart shows how the selected models spread predicted risk across positive and negative test cases.

Why this figure belongs in the Results chapter: It helps the reader see whether the model meaningfully separates higher-risk and lower-risk cases before a fixed threshold is applied.

#### What this chart shows

`How Risk Scores Spread Across Classes` plots predicted risk on the x-axis and the normalized share within each class on the y-axis. The chart is not a raw count plot. It is a normalized distribution plot, so each class is shown as a comparable shape rather than as its raw frequency.

The chart therefore answers a different question from the confusion matrix. Instead of showing yes/no outcomes, it shows how the risk scores themselves are distributed across true positive and true negative cases.

#### How to read this chart

The first thing to check is whether the positive class curve sits meaningfully to the right of the negative class curve. Stronger separation means the model assigns higher probabilities to genuinely higher-risk cases. If the curves overlap too heavily, then the model may still achieve some performance, but its class separation is weaker.

#### What the full-CDC results show in this chart

In the full-CDC visual package, the negative class is concentrated toward the lower-risk end of the scale, while the positive class spreads more broadly into moderate and high predicted-risk regions. The chart therefore supports the ranking scores already reported by the headline cards. It does not show perfect class separation, which is consistent with the fact that `Ranking Quality (AUROC)` is around **0.830** rather than near 1.000.

The central visible pattern is that the selected `Centralized` reference produces a smoother and slightly cleaner separation, while the federated visual remains broadly similar in shape. This is exactly what one would expect from the small AUROC gap between the two headline slices.

#### How this chart changes under different settings

Under raw `None` probabilities, the risk distribution tends to retain the model's original spread. Under `Sigmoid`, the central portion of the distribution usually becomes smoother and less extreme. Under `Isotonic` and `Global Isotonic`, the score distribution can become more step-like because isotonic calibration is piecewise constant. This is why calibration can improve `Risk Match Error (ECE)` substantially without visibly transforming the broad ranking behavior.

The chart also changes under threshold settings, although the threshold itself is not drawn directly here. A model whose positive-class distribution shifts farther right will often support a stricter decision rule with less loss of recall. A more overlapping distribution usually forces a sharper tradeoff between recall and specificity.

#### What this chart implies for the thesis

This chart supports the thesis argument that the selected models genuinely rank cases rather than merely memorizing a single cutoff. It visually aligns with the AUROC and PR AUC values by showing that positive and negative cases are not randomly intermixed across the risk scale.

#### What this chart does not prove

This chart does not prove that the assigned risk percentages are correctly calibrated. A model can separate classes reasonably well and still give poorly matched probabilities. That is why the calibration section remains necessary.

#### How this chart was computed and plotted

This chart is built from [dashboard_curves.csv](C:/Users/praha/Desktop/Joshna/results/full_cdc_visual_summary/dashboard_curves.csv). The grouping is the selected showcase slice, the true class label, and the score bin. The plotted value is a normalized within-class share rather than a raw count. In other words, each class distribution is normalized so that the reader can compare the shapes of the positive-class and negative-class score profiles directly. The split basis is again the held-out **test fold only**. The chart therefore visualizes class-wise score spread, not total dataset size.

### 2.4 `What Changes When We Raise Or Lower The Decision Rule`

[Insert Figure R4 here: What Changes When We Raise Or Lower The Decision Rule]

Figure title: `What Changes When We Raise Or Lower The Decision Rule`.

Caption: The threshold tradeoff chart shows how detection quality and subgroup gap measures change as the decision rule moves.

Why this figure belongs in the Results chapter: It makes threshold choice explicit instead of hiding it behind one summary score.

#### What this chart shows

`What Changes When We Raise Or Lower The Decision Rule` is the threshold-sweep chart for the showcase page. The x-axis is the decision threshold. The y-axis is a bounded performance or disparity value. Depending on the panel, the plotted lines represent `Balanced Detection Score (F1)`, precision, recall, specificity, `Positive Rate Gap`, and `Error Gap Between Groups`.

The chart therefore unifies two ideas that are often separated in shorter studies: operating-point performance and operating-point disparity.

#### How to read this chart

The first thing to look for is the threshold region where `Balanced Detection Score (F1)` peaks. The next step is to check what happens to precision, recall, specificity, and the subgroup gap lines around that same region. A threshold is not appealing simply because one line is high. It is appealing when the overall tradeoff aligns with the goal of the application.

#### What the full-CDC results show in this chart

The full-CDC results show that the threshold matters a great deal. For the showcase `Centralized` XGBoost row under `None`, the `fixed_0p5` rule gives `F1 = 0.324`, `Recall = 0.411`, and `Specificity = 0.872`. Moving to `calib_f1_optimal` gives `F1 = 0.470`, `Recall = 0.589`, and `Specificity = 0.851`. This means that the F1-tuned rule gains substantial positive-case detection at the cost of a modest loss in specificity.

For the strong federated MLP row, the threshold effect is even stronger. Under `fixed_0p5`, the model becomes conservative: `Precision = 0.597`, `Recall = 0.095`, `Specificity = 0.989`, and `F1 = 0.158`. Under `calib_f1_optimal`, the same row shifts to `Recall = 0.619`, `Specificity = 0.832`, and `F1 = 0.466`. The visual implication is that the threshold-sweep panel is not decorative. It is essential for understanding how the same trained model can look either highly specific or broadly balanced depending on the chosen cutoff.

#### How this chart changes under different settings

Across `fixed_0p5` and `calib_f1_optimal`, the threshold chart changes the most. The fixed rule favors a default interpretation, while the F1-tuned rule favors balanced detection. Across `Centralized` and `Federated`, the general tradeoff shape remains similar, but the exact threshold at which the best compromise appears can shift slightly.

Across calibration settings, `Sigmoid` and `Isotonic` often move the threshold that optimizes F1 even when AUROC remains almost unchanged. Across `FedAvg` and `FedProx`, the broad tradeoff shape is similar for logistic slices, but the low-false-positive recommendation emerges from a `FedProx` logistic row because `FedProx` can preserve a slightly more conservative decision profile in some non-IID settings.

#### What this chart implies for the thesis

This chart directly supports the thesis argument that performance alone is not enough. The chosen decision rule changes recall, specificity, and subgroup gaps. A thesis that stopped at AUROC would miss this entirely.

#### What this chart does not prove

This chart does not prove that one threshold is universally best. The preferred threshold depends on whether the goal is balanced detection, fewer false positives, or lower subgroup disparity. The chart therefore supports decision-making, but it does not replace that decision.

#### How this chart was computed and plotted

This chart is built from [dashboard_thresholds.csv](C:/Users/praha/Desktop/Joshna/results/full_cdc_polished_summary/dashboard_thresholds.csv). The grouping is the selected slice, threshold value, and subgroup family where applicable. The plotted values are grouped means such as `f1_mean`, `precision_mean`, `recall_mean`, `specificity_mean`, `demographic_parity_difference_mean`, and `equalized_odds_difference_mean`. The threshold sweep itself comes from calibration-derived candidate thresholds generated by the threshold-selection code described earlier, while the final plotted traces summarize the exported threshold rows. Where the dashboard shows a selected threshold marker, it corresponds to the slice's chosen decision threshold in the grouped summary.

## 3. Model Comparison

The `Model Comparison` page answers a broader question than the showcase page. Instead of focusing on one selected reference pair, it asks how the trained model families compare across `Centralized`, `Local-Only`, `Average Local-Only`, and `Federated` settings.

| Chart / Card | Source file | Primary grouping | Primary value(s) | Split basis | Aggregation meaning |
| --- | --- | --- | --- | --- | --- |
| `How The Trained Models Compare` | `dashboard_metrics.csv` | `run_type`, `model`, `algorithm`, `calibration`, `threshold_strategy`, site and partition metadata | `roc_auc_mean`, `f1_mean`, `pr_auc_mean`, `ece_mean` | test metrics under the chosen threshold strategy | each point is a grouped mean across matching runs and seeds |

### 3.1 `How The Trained Models Compare`

[Insert Figure R5 here: How The Trained Models Compare]

Figure title: `How The Trained Models Compare`.

Caption: The model-comparison chart places trained rows on the joint space of ranking quality and balanced detection, while still showing run-type differences.

Why this figure belongs in the Results chapter: It is the clearest chart for explaining how model families and training setups compare in the broad audit package.

#### What this chart shows

`How The Trained Models Compare` plots a model-comparison summary rather than a single scenario. The x-axis is `Ranking Quality (AUROC)`, and the y-axis is `Balanced Detection Score (F1)`. Each point or symbol represents one summary row under the selected filter state, and the color identifies the `Run Type`.

In simple language, this chart asks which trained rows are jointly strongest on ranking and thresholded balanced detection.

#### How to read this chart

The best rows move toward the upper-right portion of the figure. Points farther right rank cases better. Points higher up produce stronger balanced detection at the selected decision rule. The most useful comparison is not simply the topmost point. It is the relative arrangement of `Centralized`, `Federated`, and `Average Local-Only` points.

#### What the full-CDC results show in this chart

The broad audit package places the strongest `Centralized` row at `AUROC = 0.830` and `F1 = 0.470` for `XGBoost / None / calib_f1_optimal`. The strongest `Federated` headline row sits at `AUROC = 0.826` and `F1 = 0.467` for `FedAvg / Shallow Neural Network / Federated Isotonic / calib_f1_optimal`.

The strongest `Local-Only` and `Average Local-Only` rows are both gradient boosting under `iid`, `3` sites, and `alpha = 0.5`, with `AUROC` around **0.827** and `F1` around **0.467**. That means the strongest local-only average is also very competitive in this full-data package.

The central pattern of the chart is therefore not a huge separation between run types. It is a tight cluster of strong rows in the 0.826-0.830 AUROC and 0.466-0.470 F1 range. That is exactly why a nuanced thesis conclusion is needed.

#### How this chart changes under different settings

Across `Centralized` versus `Federated`, the performance gap remains small, with the best federated slice trailing the strongest centralized slice by about **0.004 AUROC** and about **0.003 F1** in the headline comparison.

Across `Local-Only` and `Average Local-Only`, the best local gradient-boosting slice becomes highly competitive under `iid` conditions. However, local-only rows are site-specific models rather than a shared model family that can be deployed across sites without pooling or coordinating updates. This distinction matters in the thesis interpretation.

Across model families, `XGBoost` and `Gradient Boosting` dominate the strongest centralized and local-only rows, while the strongest federated rows come from `Shallow Neural Network` and strong calibrated `Logistic` settings. Across calibration settings, AUROC usually changes little, while ECE and Brier can shift meaningfully.

#### What this chart implies for the thesis

This chart supports the central project claim that the correct headline is "Federated stays close to Centralized" rather than "Federated beats Centralized." It also shows that `Average Local-Only` is informative but is not the thesis endpoint because it does not deliver a single shared cross-site model.

#### What this chart does not prove

This chart does not prove that the top-right model is automatically the best operational choice. It does not include communication cost, calibration quality, or subgroup gaps by itself.

#### How this chart was computed and plotted

This chart is built from [dashboard_metrics.csv](C:/Users/praha/Desktop/Joshna/results/full_cdc_polished_summary/dashboard_metrics.csv). The grouping is the selected scenario metadata, including `run_type`, `model`, `algorithm`, `calibration`, `threshold_strategy`, and the site/partition descriptors that remain under the active filters. The x-axis uses `roc_auc_mean`, the y-axis uses `f1_mean`, and companion values such as `pr_auc_mean` and `ece_mean` are pulled from the same grouped rows for tooltips or card text. The values are grouped means across matching runs and seeds, not one raw run. This chart therefore belongs to the broad audit package rather than the visual verification package.

## 4. Calibration

The `Calibration` page addresses whether the predicted risk percentages can be trusted as probabilities, not only whether the cases are ranked well. This distinction is central to the thesis because decision support requires meaningful probabilities, not only high AUROC.

| Chart / Card | Source file | Primary grouping | Primary value(s) | Split basis | Aggregation meaning |
| --- | --- | --- | --- | --- | --- |
| `Do The Risk Percentages Match Reality?` | `dashboard_calibration.csv` | selected slice and reliability bin | `avg_predicted_risk_mean`, `observed_event_rate_mean` | held-out test probabilities binned into ten reliability bins | each point is a grouped bin mean |
| `How Many Cases Fall Into Each Risk Range` | `dashboard_calibration.csv` | selected slice and reliability bin | `count_mean` | held-out test probabilities binned into ten reliability bins | each bar or heatmap cell is the grouped mean count in a probability bin |
| `Which Probability Adjustment Gives The Most Trustworthy Risks?` | `dashboard_metrics.csv`, `dashboard_showcase_metrics.csv` | `calibration` with selected model and run metadata | `ece_mean`, `brier_mean`, `log_loss_mean` | test probabilities after the selected probability-adjustment method | each point or lollipop is a grouped mean summary row |

### 4.1 `Do The Risk Percentages Match Reality?`

[Insert Figure R6 here: Do The Risk Percentages Match Reality?]

Figure title: `Do The Risk Percentages Match Reality?`

Caption: Reliability curves compare the average predicted risk in each bin with the observed event rate in the same bin.

Why this figure belongs in the Results chapter: It is the most direct visual answer to whether the model's risk percentages are trustworthy.

#### What this chart shows

`Do The Risk Percentages Match Reality?` is the reliability chart. The x-axis is the average predicted risk in a probability bin. The y-axis is the observed event rate in that bin. A diagonal reference line represents perfect calibration.

If a curve lies close to the diagonal, then the model's stated risk percentages align well with the observed outcomes. If the curve sits above or below the diagonal, the model is underconfident or overconfident in that region.

#### How to read this chart

The first thing to inspect is the distance between each curve and the diagonal. The second thing is whether that distance is systematic or only local. A slight deviation in one sparse bin is less concerning than a consistent bias across many bins.

#### What the full-CDC results show in this chart

The full-CDC package shows a very strong contrast between raw and calibrated probabilities. For the strongest centralized XGBoost row:

- `None`: `ECE = 0.097`, `Brier = 0.128`, `Log Loss = 0.391`
- `Sigmoid`: `ECE = 0.020`, `Brier = 0.098`, `Log Loss = 0.320`
- `Isotonic`: `ECE = 0.004`, `Brier = 0.097`, `Log Loss = 0.313`

These values show that the raw XGBoost probabilities rank cases very well but are much less trustworthy as literal percentages. The isotonic-adjusted version keeps AUROC essentially unchanged at about **0.829** while sharply lowering calibration error.

The strongest federated probability-quality row is `FedAvg / Logistic / Global Isotonic / calib_f1_optimal` with:

- `AUROC = 0.821`
- `F1 = 0.457`
- `ECE = 0.003`
- `Brier = 0.099`
- `Log Loss = 0.319`

This means the strongest federated calibrated row actually yields a lower ECE than the strongest centralized isotonic XGBoost row, even though its AUROC is lower.

#### How this chart changes under different settings

Across `None`, `Sigmoid`, and `Isotonic`, the biggest shift is almost always in `ECE`, `Brier`, and `Log Loss`, not in AUROC. For the centralized XGBoost row, AUROC stays about **0.830** across `None` and `Sigmoid`, drops only slightly to **0.829** under `Isotonic`, and yet ECE falls dramatically from **0.097** to **0.004**.

Across federated calibration settings, `Global Isotonic` repeatedly appears at the top of the probability-quality table. For example, in `FedAvg / Logistic / Global Isotonic` with `5` sites and `non_iid alpha = 1.0`, the model reaches `ECE = 0.00318`, while similar logistic rows under weaker or no adjustment retain higher calibration error. Across `FedAvg` and `FedProx`, the best calibrated logistic rows remain extremely close in AUROC and F1, so the calibration method, not the federated method, explains most of the visible probability-quality gain.

#### What this chart implies for the thesis

This chart strongly supports the thesis argument that calibration is not an optional cosmetic step. It changes whether the model's predicted risk percentages can be interpreted as meaningful risk estimates.

#### What this chart does not prove

This chart does not prove clinical actionability by itself. A well-calibrated risk score is easier to interpret, but calibration alone does not determine whether a threshold is ethically or clinically appropriate.

#### How this chart was computed and plotted

This chart is built from [dashboard_calibration.csv](C:/Users/praha/Desktop/Joshna/results/full_cdc_polished_summary/dashboard_calibration.csv). The grouping is the selected slice together with the reliability `bin`. The x-axis uses `avg_predicted_risk_mean`, and the y-axis uses `observed_event_rate_mean`. Each point therefore represents a grouped mean probability bin in the held-out test data. The diagonal reference line is a plotting aid rather than a measured series; it represents perfect calibration where predicted risk matches observed event frequency exactly.

### 4.2 `How Many Cases Fall Into Each Risk Range`

[Insert Figure R7 here: How Many Cases Fall Into Each Risk Range]

Figure title: `How Many Cases Fall Into Each Risk Range`.

Caption: The probability-range chart shows how many cases fall into each calibrated or uncalibrated risk bin.

Why this figure belongs in the Results chapter: It shows whether the reliability story is supported by enough cases in each part of the probability scale.

#### What this chart shows

`How Many Cases Fall Into Each Risk Range` is the risk-bin count view. The x-axis is the probability bin. The y-axis is the `Probability Adjustment` setting. The cell intensity or bar height shows how many cases fall into that range.

This chart is important because a reliability curve can look neat even when very few cases occupy the most extreme bins.

#### How to read this chart

The first thing to inspect is where most cases accumulate. The second is whether the highest-risk bins are supported by many or few cases. If most cases sit in the lower bins, then the low-risk calibration region is statistically better supported than the most extreme high-risk region.

#### What the full-CDC results show in this chart

The full-CDC package shows a strong concentration of cases in lower predicted-risk bins. This is consistent with the class balance of the CDC dataset and with the fact that most cases are not positive. The higher-risk bins contain fewer cases, which means calibration statements in the upper-right region of the reliability curve should be read more carefully.

This is also why the project reports both calibration curves and probability-range counts. The counts show where the visual calibration story is backed by dense evidence and where it rests on thinner support.

#### How this chart changes under different settings

Under `None`, the score mass can remain more spread or more jagged depending on the model family. Under `Sigmoid`, the mid-range usually becomes smoother. Under `Isotonic` and `Global Isotonic`, the mass can appear more structured because isotonic calibration is monotonic and piecewise constant.

Across `Centralized` and `Federated`, the general picture is similar: lower bins contain the largest number of cases. What changes more substantially is whether the bin-wise observed frequencies align with those predicted probabilities, which is why the reliability chart and the calibration quality summary remain necessary.

#### What this chart implies for the thesis

This chart supports responsible interpretation of the calibration figures. It reminds the reader that calibration quality should be read together with the density of cases across the score range.

#### What this chart does not prove

This chart does not prove good or bad calibration by itself. It only shows the support structure of the risk scale.

#### How this chart was computed and plotted

This chart also comes from [dashboard_calibration.csv](C:/Users/praha/Desktop/Joshna/results/full_cdc_polished_summary/dashboard_calibration.csv). The grouping is the selected slice and calibration `bin`, and the plotted value is `count_mean`. In practical terms, the chart shows the mean number of held-out test cases falling into each probability bin under the selected `Probability Adjustment`. The chart does not normalize the counts across bins; it reports grouped mean bin occupancy. That is why it answers a different question from the normalized score-distribution chart on the `Showcase` page.

### 4.3 `Which Probability Adjustment Gives The Most Trustworthy Risks?`

[Insert Figure R8 here: Which Probability Adjustment Gives The Most Trustworthy Risks?]

Figure title: `Which Probability Adjustment Gives The Most Trustworthy Risks?`

Caption: The probability-quality comparison chart summarizes `Risk Match Error (ECE)`, `Probability Error (Brier)`, and `Confidence Penalty (Log Loss)` across adjustment choices.

Why this figure belongs in the Results chapter: It converts the reliability story into a direct cross-setting comparison.

#### What this chart shows

`Which Probability Adjustment Gives The Most Trustworthy Risks?` compares calibration settings on three probability-quality metrics: `Risk Match Error (ECE)`, `Probability Error (Brier)`, and `Confidence Penalty (Log Loss)`. Lower values are better for all three.

#### How to read this chart

The first thing to look for is which adjustment method sits lowest across the three metrics. The second is whether that improvement comes with a large or small change in AUROC and F1 on the associated summary rows.

#### What the full-CDC results show in this chart

The full-CDC results show that the raw `None` probabilities are repeatedly dominated by calibrated alternatives. For the strongest centralized XGBoost row, moving from `None` to `Isotonic` reduces `ECE` from **0.097** to **0.004**, reduces `Brier` from **0.128** to **0.097**, and reduces `Log Loss` from **0.391** to **0.313**, while AUROC changes only slightly.

The strongest federated probability-quality row is `FedAvg / Logistic / Global Isotonic`, with `ECE = 0.003`, `Brier = 0.099`, and `Log Loss = 0.319`. This means the most trustworthy risk percentages in the full-CDC package are delivered by a federated calibrated logistic slice rather than by the absolute top centralized ranking row.

#### How this chart changes under different settings

Across centralized settings, `Sigmoid` is a strong middle ground: it preserves AUROC exactly for XGBoost while reducing ECE from **0.097** to **0.020**. `Isotonic` goes further, reducing ECE to about **0.004**, but with a slight drop in PR AUC and F1.

Across federated settings, `Global Isotonic` is consistently strongest for low ECE. Across heterogeneity settings, the differences are modest. For `FedAvg / Logistic / Global Isotonic / calib_f1_optimal` with `5` sites and `non_iid`, AUROC remains around **0.821** across `alpha = 1.0`, `0.5`, and `0.1`, while ECE stays very low in the range of roughly **0.0032** to **0.0038**. This suggests that heterogeneity affects the calibrated logistic rows much less than one might fear.

#### What this chart implies for the thesis

This chart supports one of the strongest trustworthiness claims in the thesis: the final framework is not merely accurate, but explicitly calibration-aware, and it can identify settings where federated probabilities are especially reliable.

#### What this chart does not prove

This chart does not prove that the same calibration method is universally best for every model family, dataset, or deployment objective. It only proves what was strongest within the current result package.

#### How this chart was computed and plotted

This chart is constructed from the grouped metric summaries in [dashboard_metrics.csv](C:/Users/praha/Desktop/Joshna/results/full_cdc_polished_summary/dashboard_metrics.csv), with the showcase rows cross-checked against [dashboard_showcase_metrics.csv](C:/Users/praha/Desktop/Joshna/results/full_cdc_polished_summary/dashboard_showcase_metrics.csv). The grouping is the selected slice and `calibration` method. The primary plotted columns are `ece_mean`, `brier_mean`, and `log_loss_mean`. All three are grouped means computed from test probabilities after the chosen probability adjustment. The chart therefore compares calibration methods on their exported summary metrics rather than recomputing calibration live inside the dashboard.

## 5. Explainability

The `Explainability` page asks a different kind of question from the performance and calibration pages. It asks whether the selected models produce understandable explanation patterns and whether those patterns remain stable across sites and rounds.

| Chart / Card | Source file | Primary grouping | Primary value(s) | Split basis | Aggregation meaning |
| --- | --- | --- | --- | --- | --- |
| `Which Features Matter Most?` | `dashboard_shap.csv` | selected slice and feature | `mean_abs_shap_mean`, `rank_mean` | saved explanation summaries from evaluation artifacts | each bar is a grouped mean feature-attribution magnitude |
| `What Pushed This One Prediction Up Or Down?` | `dashboard_local_explanations.csv` | selected slice and feature | `contribution_mean` | representative saved local explanation from the visual verification package | each bar is a grouped mean contribution for one representative explanation row |
| `Do The Feature Explanations Stay Consistent Over Time?` | `dashboard_stability.csv` | selected slice and round | `spearman_top_feature_stability_mean` | round-wise explanation summaries from federated runs | each point is a grouped mean stability score by round |
| `Do Different Sites Focus On Similar Features?` | `dashboard_stability.csv` | selected slice and client pair | `top_k_overlap_mean`, `spearman_rank_correlation_mean` | cross-client explanation summaries from the visual verification package | each cell or point summarizes grouped client-pair overlap |

### 5.1 `Which Features Matter Most?`

[Insert Figure R9 here: Which Features Matter Most?]

Figure title: `Which Features Matter Most?`

Caption: The feature-importance chart summarizes which inputs carry the largest average explanatory weight.

Why this figure belongs in the Results chapter: It shows whether the final models depend on clinically plausible public variables and whether the explanation story is stable across strong slices.

#### What this chart shows

`Which Features Matter Most?` is the global feature-attribution chart. The x-axis is average feature impact, usually measured through the exported `mean absolute SHAP impact` summary. The y-axis lists the features. Larger values indicate that the model depends more heavily on that feature on average.

#### How to read this chart

The first thing to inspect is the top-ranked feature. The next is whether the same feature continues to appear across multiple strong settings rather than only once.

#### What the full-CDC results show in this chart

The strongest recurring pattern in the full-CDC explainability tables is the dominance of `GenHlth` (general health status). In the exported top attribution table, `GenHlth` repeatedly appears with mean absolute attribution values around **0.93** to **1.02** and with mean rank near **1.0** to **1.2** across strong federated MLP slices.

This is not an isolated row. The dominance of `GenHlth` appears in both `FedAvg` and `FedProx` MLP settings and across several `non_iid` levels. For example:

- `FedAvg / MLP / none / 5 sites / non_iid / alpha 0.5`: `GenHlth` mean absolute impact about **1.018**
- `FedProx / MLP / none / 5 sites / non_iid / alpha 0.5`: `GenHlth` mean absolute impact about **0.997**
- `FedAvg / MLP / none / 5 sites / non_iid / alpha 1.0`: `GenHlth` mean absolute impact about **0.959**

This repeated pattern is more persuasive than a single explanation snapshot.

#### How this chart changes under different settings

Across `FedAvg` and `FedProx`, the explanation ranking is very similar for the strong MLP rows. Across heterogeneity settings, the leading feature remains the same even when the exact impact value shifts slightly. This is important because it means the explanatory narrative is not completely unstable under changing site distributions.

The feature-attribution chart differs more strongly across model families than across calibration settings. Calibration changes the probability mapping more than the underlying feature dependence. The strongest explanation comparison is therefore between `Logistic` and `MLP`, not between `None` and `Isotonic`.

#### What this chart implies for the thesis

This chart supports the thesis claim that the project includes explainable evidence rather than a black-box performance report. It also supports the stability claim that at least some major explanatory patterns survive across strong federated settings.

#### What this chart does not prove

This chart does not prove causal importance. A feature can have a large attribution without being a causal driver of diabetes risk.

#### How this chart was computed and plotted

This chart is built from [dashboard_shap.csv](C:/Users/praha/Desktop/Joshna/results/full_cdc_polished_summary/dashboard_shap.csv). The grouping is the selected slice and `feature` name. The primary plotted value is `mean_abs_shap_mean`, with `rank_mean` used to preserve the feature ordering. The values are grouped mean absolute attribution magnitudes exported from the saved SHAP summaries. This chart belongs to the broad audit package because global feature-importance rows are already aggregated and do not require prediction-level plotting artifacts.

### 5.2 `What Pushed This One Prediction Up Or Down?`

[Insert Figure R10 here: What Pushed This One Prediction Up Or Down?]

Figure title: `What Pushed This One Prediction Up Or Down?`

Caption: The representative local explanation chart shows which features raised or lowered one selected prediction.

Why this figure belongs in the Results chapter: It complements the global importance chart by showing how one prediction is assembled locally.

#### What this chart shows

`What Pushed This One Prediction Up Or Down?` is a local explanation panel. Instead of averaging over the whole dataset, it shows how one selected prediction was pushed upward or downward by its input features. Positive contributions move the predicted risk upward, and negative contributions move it downward.

#### How to read this chart

The first thing to inspect is which features have the largest positive or negative contributions. The second is whether those features are also prominent in the global importance chart. Strong agreement between the two charts makes the explanation story easier to follow.

#### What the full-CDC results show in this chart

The exported local explanation rows for the full-CDC federated logistic visual slices repeatedly show `Age`, `HighBP`, `GenHlth`, and `HighChol` among the strongest local drivers. In the saved `FedAvg / Logistic / none / calib_f1_optimal` rows, `Age` repeatedly appears with contribution values around **-0.45** to **-0.47**, while `HighBP` appears around **-0.07** to **-0.09**, and `GenHlth` appears around **-0.07** to **-0.08**. The exact sign depends on the selected representative example, but the important point is that the same clinically sensible variables keep reappearing.

The centralized gradient boosting local explanation view also shows plausible variables such as `Age`, `HighBP`, `GenHlth`, `BMI`, `HighChol`, and `PhysActivity`. This gives the chapter a defensible explanation narrative rather than a list of opaque latent features.

#### How this chart changes under different settings

Across `FedAvg` and `FedProx` logistic slices, the local explanation rankings remain very close in the saved representative examples. Across model families, the exact contribution magnitudes differ because the functional form differs, but the recurrent variable set remains clinically recognizable.

Calibration settings have less effect on which feature is ranked first in a local explanation than on how the final probability is expressed. In that sense, local explanations are more model- and case-dependent than calibration-dependent.

#### What this chart implies for the thesis

This chart supports the practical interpretability claim of the project. It shows that the framework can provide one-case explanations that are readable and connected to actual CDC variables.

#### What this chart does not prove

This chart does not prove that the local explanation is causal, complete, or unique. It is an interpretation aid, not a proof of mechanism.

#### How this chart was computed and plotted

This chart is built from [dashboard_local_explanations.csv](C:/Users/praha/Desktop/Joshna/results/full_cdc_visual_summary/dashboard_local_explanations.csv). The grouping is the selected visual slice and `feature`. The primary plotted value is `contribution_mean`, which summarizes the saved representative explanation rows. The split basis is the visual verification package because representative local explanations require prediction-backed artifact files. The chart therefore illustrates how one saved example is decomposed rather than averaging feature effects over the whole test set.

### 5.3 `Do The Feature Explanations Stay Consistent Over Time?`

[Insert Figure R11 here: Do The Feature Explanations Stay Consistent Over Time?]

Figure title: `Do The Feature Explanations Stay Consistent Over Time?`

Caption: The explanation-stability chart tracks how similar the leading feature ranking remains across federated rounds.

Why this figure belongs in the Results chapter: It shows whether the explanation story is stable or changes erratically during federated training.

#### What this chart shows

`Do The Feature Explanations Stay Consistent Over Time?` is the round-wise explanation-stability chart. The x-axis is training round. The y-axis is average explanation stability, measured through Spearman-style rank agreement for the top explanatory features.

#### How to read this chart

The first thing to inspect is whether the line rises, falls, or remains noisy. A more stable explanation trace approaches 1.0 and stays there. A more unstable trace fluctuates heavily.

#### What the full-CDC results show in this chart

For the representative `FedAvg / Logistic / none / calib_f1_optimal / 3 sites / iid / alpha 0.5` slice, the stability line rises quickly after the early rounds:

- round 3: about **0.963**
- round 4: about **0.984**
- round 8: about **0.998**
- round 11: about **0.998**
- round 13: about **0.999**

This is a strong stability pattern. It means that once the early optimization settles, the top explanatory features stop changing much.

#### How this chart changes under different settings

Across `FedAvg` and `FedProx`, the overall stability level remains high. Across stronger heterogeneity, the path to stability can be slightly noisier, but the broad result is still one of convergence rather than drift. Across `3` and `5` sites, the same pattern holds: the later rounds show very high feature-rank stability once the optimization settles.

#### What this chart implies for the thesis

This chart supports the thesis claim that the framework audits not only explanation content but also explanation stability. That is a more demanding standard than simply providing a single explanation plot.

#### What this chart does not prove

This chart does not prove that the explanations are correct in a causal sense. It only proves that they are internally stable.

#### How this chart was computed and plotted

This chart is built from [dashboard_stability.csv](C:/Users/praha/Desktop/Joshna/results/full_cdc_visual_summary/dashboard_stability.csv). The grouping is the selected slice and `round`. The plotted value is `spearman_top_feature_stability_mean`, which comes from the round-wise feature-rank stability summaries. These values are grouped means over matching saved visual rows. The x-axis is the federated training round, and the y-axis is the stability score, so the line represents how consistently the ranked feature-importance pattern is preserved over time.

### 5.4 `Do Different Sites Focus On Similar Features?`

[Insert Figure R12 here: Do Different Sites Focus On Similar Features?]

Figure title: `Do Different Sites Focus On Similar Features?`

Caption: The cross-site overlap chart shows whether different client sites emphasize similar top features.

Why this figure belongs in the Results chapter: It connects explanation quality to the federated setting rather than only to pooled-data modeling.

#### What this chart shows

`Do Different Sites Focus On Similar Features?` is the cross-client top-k overlap chart. Each cell compares one site with another site. Higher overlap means the sites place similar features in their top explanatory set.

#### How to read this chart

The first thing to inspect is whether the overlap stays uniformly high across the matrix or whether there are isolated site pairs with much lower agreement.

#### What the full-CDC results show in this chart

For the representative `FedAvg / Logistic / none / calib_f1_optimal / 3 sites / iid / alpha 0.5` slice, the exported overlap values are **1.0** for the site pairs shown in the saved cross-site matrix. That means the saved top-k feature sets are perfectly aligned across those sites in the selected illustrative slice.

#### How this chart changes under different settings

Across more heterogeneous site splits, perfect overlap should not be assumed, but the broad pattern in the visual package is still one of strong agreement rather than fragmentation. Across `FedAvg` and `FedProx`, the shared explanation story remains similar, which is consistent with the stability chart.

#### What this chart implies for the thesis

This chart supports the idea that the federated model is not learning totally different explanation stories at different sites. That matters for auditability.

#### What this chart does not prove

This chart does not prove that all sites are interchangeable or that no subgroup-specific differences exist. It only addresses overlap in the top explanatory features.

#### How this chart was computed and plotted

This chart also uses [dashboard_stability.csv](C:/Users/praha/Desktop/Joshna/results/full_cdc_visual_summary/dashboard_stability.csv), but the grouping is now the selected slice plus the client pair (`client_left`, `client_right`). The primary plotted value is `top_k_overlap_mean`, with `spearman_rank_correlation_mean` available as a companion agreement statistic where present. The values summarize cross-client overlap in the saved top-k feature sets. This makes the chart a direct federated explanation-consistency visualization rather than a pooled-model importance chart.

## 6. Fairness

The `Fairness` page does not claim formal fairness certification. Instead, it provides subgroup disparity analysis on publicly available variables such as `age_group`, `bmi_category`, and `sex`. This is why the chapter stays careful with wording. The result package identifies where the model behaves differently across groups, but it does not claim that these checks are the final ethical verdict.

| Chart / Card | Source file | Primary grouping | Primary value(s) | Split basis | Aggregation meaning |
| --- | --- | --- | --- | --- | --- |
| `How Often Each Group Is Flagged Positive` | `dashboard_fairness.csv` | selected slice, `group_feature`, and `group_value` | `selection_rate_mean` | thresholded test predictions | each point or bar is a grouped mean subgroup positive rate |
| `How Different The Results Are Between Groups` | `dashboard_fairness.csv` | selected slice and `group_feature` | `demographic_parity_difference_mean`, `equalized_odds_difference_mean` | thresholded test predictions summarized by subgroup family | each cell or bar is a grouped mean disparity score |
| `What Changes For Each Group When We Raise Or Lower The Decision Rule?` | `dashboard_thresholds.csv` | selected slice, threshold, and `group_feature` | `selection_rate_mean`, `demographic_parity_difference_mean`, `equalized_odds_difference_mean`, plus overall F1 and specificity | threshold sweep over calibration-derived candidate thresholds | each line is a grouped mean subgroup trace by threshold |

### 6.1 `How Often Each Group Is Flagged Positive`

[Insert Figure R13 here: How Often Each Group Is Flagged Positive]

Figure title: `How Often Each Group Is Flagged Positive`.

Caption: The subgroup positive-rate chart shows how frequently each group receives a positive prediction under the selected decision rule.

Why this figure belongs in the Results chapter: It is the simplest visual entry point for subgroup behavior.

#### What this chart shows

`How Often Each Group Is Flagged Positive` plots `Positive Prediction Rate` for each subgroup. In simple language, it shows how often a person in each subgroup is flagged as diabetic by the selected model and threshold.

#### How to read this chart

The first thing to look for is whether the subgroup rates are close together or widely separated. Large separations do not automatically prove unfairness, but they do indicate behavior that deserves audit attention.

#### What the full-CDC results show in this chart

For the representative federated logistic fairness slice shown in the exported tables, the subgroup positive rates differ most across `bmi_category` and `age_group`, and much less across `sex`. For example, in the `FedAvg / Logistic / Federated Isotonic / calib_f1_optimal / 3 sites / iid / alpha 0.5` slice:

- `age_group` rates range from about **0.031** for `under_40` to about **0.346** for `60_plus`
- `bmi_category` rates range from about **0.092** for `healthy` to about **0.430** for `obese`
- `sex` rates are closer, roughly **0.207** for `female` and **0.250** for `male`

These are meaningful differences and they are consistent with known diabetes risk structure, but they still require explicit auditing.

#### How this chart changes under different settings

The chart changes strongly with the `Decision Rule`. A lower threshold raises positive prediction rates across all groups and can widen some subgroup differences. A stricter threshold reduces positive prediction rates and can either narrow or widen the gaps depending on where the score distributions differ.

Across subgroup dimensions, `BMI` shows the widest spread, followed by `age_group`, while `sex` usually shows the narrowest spread in the reported slices.

#### What this chart implies for the thesis

This chart supports the thesis claim that the framework includes subgroup auditing rather than only one overall score.

#### What this chart does not prove

This chart does not prove whether the observed subgroup differences are justified or unjustified. It only shows that they exist and need interpretation.

#### How this chart was computed and plotted

This chart is built from [dashboard_fairness.csv](C:/Users/praha/Desktop/Joshna/results/full_cdc_polished_summary/dashboard_fairness.csv). The grouping is the selected slice, `group_feature`, and `group_value`. The primary plotted value is `selection_rate_mean`, which is the grouped mean positive prediction rate for each subgroup under the selected threshold. The split basis is thresholded **test predictions**. This means the chart is a post-threshold subgroup summary rather than a raw score-distribution view.

### 6.2 `How Different The Results Are Between Groups`

[Insert Figure R14 here: How Different The Results Are Between Groups]

Figure title: `How Different The Results Are Between Groups`.

Caption: The group-gap chart summarizes subgroup disparity using `Positive Rate Gap` and `Error Gap Between Groups`.

Why this figure belongs in the Results chapter: It turns subgroup behavior into direct disparity metrics that can be compared across settings.

#### What this chart shows

`How Different The Results Are Between Groups` summarizes the gap metrics rather than the raw subgroup rates. `Positive Rate Gap` is the practical dashboard label for `Demographic Parity Difference`. `Error Gap Between Groups` is the practical dashboard label for `Equalized Odds Difference`.

#### How to read this chart

The first thing to inspect is which subgroup dimension has the largest gap. The second is whether the `Positive Rate Gap` and the `Error Gap Between Groups` tell the same story or diverge.

#### What the full-CDC results show in this chart

In the representative federated fairness slice, the largest subgroup gaps occur in `bmi_category`, followed by `age_group`, while `sex` is much smaller. For the same `FedAvg / Logistic / Federated Isotonic / calib_f1_optimal / 3 sites / iid / alpha 0.5` slice:

- `age_group`: `Positive Rate Gap = 0.315`, `Error Gap Between Groups = 0.385`
- `bmi_category`: `Positive Rate Gap = 0.339`, `Error Gap Between Groups = 0.405`
- `sex`: `Positive Rate Gap = 0.044`, `Error Gap Between Groups = 0.038`

This means that the subgroup differences are not evenly distributed across group definitions. BMI-based subgroup structure is the most visibly different in the reported slice.

#### How this chart changes under different settings

Across `None`, `Sigmoid`, and `Isotonic`, the calibration choice can move the threshold-sensitive group rates slightly, but threshold choice usually produces the larger visible difference. Across `fixed_0p5` and `calib_f1_optimal`, the gap profile can change because the positive prediction rate changes.

Across `Centralized` and `Federated`, there is no simple single winner on every subgroup metric. That is exactly why the fairness tab exists. The thesis should not overclaim a uniform fairness advantage where the results are setting-dependent.

#### What this chart implies for the thesis

This chart supports the thesis claim that the framework measures subgroup disparity explicitly and does not treat predictive performance as sufficient.

#### What this chart does not prove

This chart does not prove formal fairness certification, legal compliance, or ethical adequacy. It is an audit signal, not a final normative judgment.

#### How this chart was computed and plotted

This chart also uses [dashboard_fairness.csv](C:/Users/praha/Desktop/Joshna/results/full_cdc_polished_summary/dashboard_fairness.csv). The grouping is the selected slice and `group_feature`. The plotted values are `demographic_parity_difference_mean` and `equalized_odds_difference_mean`, which the dashboard presents as `Positive Rate Gap` and `Error Gap Between Groups`. These are grouped mean disparity values computed from thresholded test predictions. The chart therefore summarizes subgroup disparities at the current operating point rather than over the full threshold sweep.

### 6.3 `What Changes For Each Group When We Raise Or Lower The Decision Rule?`

[Insert Figure R15 here: What Changes For Each Group When We Raise Or Lower The Decision Rule?]

Figure title: `What Changes For Each Group When We Raise Or Lower The Decision Rule?`

Caption: The group-threshold chart shows how subgroup positive rates and disparity gaps move when the threshold changes.

Why this figure belongs in the Results chapter: It links fairness auditing directly to threshold selection.

#### What this chart shows

`What Changes For Each Group When We Raise Or Lower The Decision Rule?` is the subgroup threshold-sweep chart. The x-axis is the decision threshold. The y-axis shows the changing subgroup rates and gap metrics. The chart answers a precise operational question: how do subgroup disparities move when the model becomes more permissive or more conservative?

#### How to read this chart

The first thing to look for is whether the subgroup-gap lines flatten or steepen as the threshold moves. The next is whether the threshold region with stronger `Balanced Detection Score (F1)` is also the region with acceptable group gaps.

#### What the full-CDC results show in this chart

The full-CDC package shows that subgroup gaps are threshold-sensitive. The clearest exported fairness-conscious row is:

- `Local-Only / Random Forest / Isotonic / calib_f1_optimal / 3 sites / non_iid / alpha 0.1`
- threshold about **0.0649**
- `Balanced Detection Score (F1) = 0.429`
- `Specificity = 0.885`
- `Positive Rate Gap = 0.013`
- `Error Gap Between Groups = 0.013`

This row is not the strongest predictive row in the entire project, but it is the strongest row in the exported fairness-threshold table because it combines useful detection performance with very small reported subgroup gap values.

#### How this chart changes under different settings

Across thresholds, the chart changes a great deal. Lower thresholds usually increase recall and positive prediction rate while sometimes increasing subgroup gaps. Higher thresholds reduce positive calls and can reduce or intensify different gap components depending on where the subgroup score distributions lie.

Across subgroup definitions, `bmi_category` tends to move more strongly than `sex`. Across `FedAvg` and `FedProx`, the fairness picture remains mixed rather than uniformly dominated by one method. Across `iid` and `non_iid`, heterogeneity can change how stable the subgroup curves appear, which is why the dashboard keeps this chart separate from the headline performance view.

#### What this chart implies for the thesis

This chart supports a key practical thesis point: the fairness story depends not only on the trained model but also on the deployed threshold. A threshold-aware framework is therefore more defensible than a framework that reports fairness only at a default 0.5 cutoff.

#### What this chart does not prove

This chart does not prove that the exported fairness-conscious threshold is the clinically best threshold. It only proves that threshold choice changes subgroup disparity and that those changes can be measured.

#### How this chart was computed and plotted

This chart is built from [dashboard_thresholds.csv](C:/Users/praha/Desktop/Joshna/results/full_cdc_polished_summary/dashboard_thresholds.csv). The grouping is the selected slice, threshold, and `group_feature`. The plotted values are grouped means such as `selection_rate_mean`, `demographic_parity_difference_mean`, and `equalized_odds_difference_mean`, together with overall metrics like `f1_mean` and `specificity_mean` for context. The threshold values come from the calibration-derived candidate set described earlier, and the dashboard's selected threshold marker corresponds to the slice's chosen decision rule. This chart therefore combines performance and fairness over a threshold sweep rather than reporting one fixed operating point only.

## 7. Federated Training

The `Federated Training` page turns the project back toward the federated optimization process itself. It answers two questions: how the shared model evolves round by round, and how much communication that process requires.

| Chart / Card | Source file | Primary grouping | Primary value(s) | Split basis | Aggregation meaning |
| --- | --- | --- | --- | --- | --- |
| `How Federated Training Improves Round By Round` | `dashboard_rounds.csv` | selected federated slice and `round` | `global_eval_auc_mean`, `global_eval_f1_mean`, `global_eval_ece_mean` | round-wise evaluation summaries | each point is a grouped mean metric value at a given round |
| `How Much Data Is Sent During Training?` | `dashboard_rounds.csv`, `communication_summary.csv` | selected federated slice and `round` | `communication_bytes_mean`, `cumulative_communication_bytes_mean` | round-wise communication summaries | each bar or line is a grouped mean communication quantity |

### 7.1 `How Federated Training Improves Round By Round`

[Insert Figure R16 here: How Federated Training Improves Round By Round]

Figure title: `How Federated Training Improves Round By Round`.

Caption: The round-wise federated chart tracks predictive quality and calibration quality as training rounds accumulate.

Why this figure belongs in the Results chapter: It shows whether federated optimization converges steadily rather than only reporting the final row.

#### What this chart shows

`How Federated Training Improves Round By Round` plots training round on the x-axis and a performance quantity on the y-axis. The chart includes round-wise `Ranking Quality (AUROC)`, round-wise `Balanced Detection Score (F1)`, and round-wise `Risk Match Error (ECE)` or related evaluation traces depending on the selected slice.

#### How to read this chart

The first thing to inspect is whether the AUROC and F1 traces rise and then stabilize. The second is whether ECE falls or remains controlled as the rounds continue. A stronger federated process is one that converges smoothly rather than improving erratically.

#### What the full-CDC results show in this chart

For the representative `FedAvg / Logistic / none / 3 sites / iid / alpha 0.5` slice, the round-wise trace shows a clean early rise:

- round 1: `AUROC = 0.813`, `F1 = 0.097`, `ECE = 0.037`
- round 5: `AUROC = 0.821`, `F1 = 0.214`, `ECE = 0.020`
- round 10: `AUROC = 0.822`, `F1 = 0.230`, `ECE = 0.015`

The key visual point is that the largest improvement happens early, after which the shared model stabilizes.

#### How this chart changes under different settings

Across `FedAvg` and `FedProx`, the final logistic results for the `3`-site `iid` slice are almost identical:

- `FedAvg`: `AUROC = 0.821419`, `F1 = 0.458295`, `ECE = 0.012934`
- `FedProx`: `AUROC = 0.821417`, `F1 = 0.458244`, `ECE = 0.012961`

This means the two methods behave very similarly in this relatively mild setting. The reason to keep both in the thesis is not that they diverge dramatically everywhere. The reason is that `FedProx` gives a principled heterogeneity-aware baseline whose behavior can be compared when data become more different.

Across heterogeneity levels under `Global Isotonic` logistic rows with `5` sites and `non_iid`, the final metrics also remain remarkably stable:

- `alpha = 1.0`: `AUROC = 0.8210`, `F1 = 0.4573`, `ECE = 0.00318`
- `alpha = 0.5`: `AUROC = 0.8211`, `F1 = 0.4580`, `ECE = 0.00378`
- `alpha = 0.1`: `AUROC = 0.8207`, `F1 = 0.4585`, `ECE = 0.00357`

This is an important result. It shows that stronger non-IID splitting does not destroy performance in the selected calibrated logistic family.

#### What this chart implies for the thesis

This chart supports the thesis claim that federated optimization is not only possible on the chosen public data, but stable enough to yield interpretable round-wise behavior and competitive final performance.

#### What this chart does not prove

This chart does not prove that the same convergence behavior would hold on real hospital infrastructure. It is still a public-data federation simulation.

#### How this chart was computed and plotted

This chart is built from [dashboard_rounds.csv](C:/Users/praha/Desktop/Joshna/results/full_cdc_polished_summary/dashboard_rounds.csv). The grouping is the selected federated slice and `round`. The primary plotted columns are `global_eval_auc_mean`, `global_eval_f1_mean`, and `global_eval_ece_mean`. Each point is therefore the grouped mean of a round-wise evaluation quantity across matching runs. The chart belongs to the broad audit package because the round-wise metrics are already exported as grouped federated summaries.

### 7.2 `How Much Data Is Sent During Training?`

[Insert Figure R17 here: How Much Data Is Sent During Training?]

Figure title: `How Much Data Is Sent During Training?`

Caption: The communication chart shows both per-round and cumulative bytes exchanged during federated optimization.

Why this figure belongs in the Results chapter: It makes the data-locality versus communication-cost tradeoff explicit.

#### What this chart shows

`How Much Data Is Sent During Training?` plots communication bytes over federated rounds. Depending on the view, it shows per-round exchange, cumulative communication, or both. The chart does not measure raw-data movement. It measures parameter-update traffic.

#### How to read this chart

The first thing to inspect is the cumulative line. The second is whether more sites or more rounds increase communication sharply. A communication-efficient setting is one that stays competitive without excessive cumulative bytes.

#### What the full-CDC results show in this chart

The most communication-efficient competitive federated choice in the exported thesis table is:

- `Federated / FedProx / Logistic / None / calib_f1_optimal / 3 sites / non_iid / alpha 1.0`
- `cumulative communication = 31,680 bytes`
- `Ranking Quality (AUROC) = 0.821`

For the `FedAvg / Logistic / none / 3 sites / iid / alpha 0.5` slice, the round communication is about **1,056 bytes** per round and the cumulative communication reaches about **42,240 bytes** over the run. When the same logistic family is expanded to `5` sites, the cumulative communication rises to about **70,400 bytes**. This makes the site-count effect immediately visible.

#### How this chart changes under different settings

Across `3` and `5` sites, communication rises substantially because more clients send updates. Across `iid` and `non_iid`, cumulative communication can stay the same if the run length is fixed, but the performance return on those bytes may shift slightly. Across `FedAvg` and `FedProx`, communication in this implementation is driven more by the model and site count than by the choice of federated algorithm, because both methods exchange similarly sized parameter updates.

#### What this chart implies for the thesis

This chart supports a central clarification in the thesis: federated learning reduces centralized raw-data movement, but it does not eliminate communication cost. Instead, it replaces raw-data pooling with iterative model-update exchange.

#### What this chart does not prove

This chart does not prove total deployment cost, wall-clock cost, or energy cost by itself. It only proves the communication burden within the current simulation framework.

#### How this chart was computed and plotted

This chart uses [dashboard_rounds.csv](C:/Users/praha/Desktop/Joshna/results/full_cdc_polished_summary/dashboard_rounds.csv) together with [communication_summary.csv](C:/Users/praha/Desktop/Joshna/thesis_assets/full_cdc_polished/tables/communication_summary.csv) or its exported summary equivalent where needed for final totals. The grouping is the selected federated slice and `round`. The plotted values are per-round communication bytes and cumulative communication bytes, both as grouped means. The bars or lines therefore represent model-update traffic rather than any estimate of raw-data movement.

## 8. Full Study Comparison

The `Full Study Comparison` page is where the broad audit package becomes easiest to interpret as a thesis result. This page compares `Centralized`, `Local-Only`, `Average Local-Only`, and `Federated` directly.

| Chart / Card | Source file | Primary grouping | Primary value(s) | Split basis | Aggregation meaning |
| --- | --- | --- | --- | --- | --- |
| `How Centralized, Local-Only, and Federated Compare` | `dashboard_metrics.csv` | `run_type` plus selected model and scenario filters | setup-level `f1_mean`, with AUROC and ECE available as companion values | test metrics under the chosen threshold strategy | each setup summary is a grouped mean across matching rows in the broad audit package |

### 8.1 `How Centralized, Local-Only, and Federated Compare`

[Insert Figure R18 here: How Centralized, Local-Only, and Federated Compare]

Figure title: `How Centralized, Local-Only, and Federated Compare`.

Caption: The study-comparison chart summarizes how the main training setups differ on balanced detection and related metrics.

Why this figure belongs in the Results chapter: It provides the clearest single visualization of the thesis comparison logic across training setups.

#### What this chart shows

`How Centralized, Local-Only, and Federated Compare` places the main training setups next to one another using aggregated summary rows. It is not a single-model figure. It is a broad comparison figure that shows how the major training paradigms differ in the full-CDC package.

#### How to read this chart

The first comparison is `Centralized` versus `Federated`. The next is whether `Average Local-Only` sits closer to the centralized reference or farther away. The broad question is not "who wins one score?" but "how do the training paradigms differ once the result package is read as a whole?"

#### What the full-CDC results show in this chart

The strongest `Centralized` row remains `XGBoost / None / calib_f1_optimal` with `AUROC = 0.830`, `PR AUC = 0.430`, and `F1 = 0.470`.

The strongest `Federated` headline row remains `FedAvg / Shallow Neural Network / Federated Isotonic / calib_f1_optimal` with `AUROC = 0.826`, `PR AUC = 0.416`, and `F1 = 0.467`.

The strongest `Average Local-Only` row remains `Gradient Boosting / Sigmoid / calib_f1_optimal / 3 sites / iid / alpha 0.5` with `AUROC = 0.827`, `PR AUC = 0.423`, and `F1 = 0.467`.

This means the broad audit package supports a nuanced interpretation:

1. `Centralized` remains the pooled-data performance upper bound.
2. `Federated` stays very close.
3. `Average Local-Only` can also be highly competitive in some mild settings, especially under `iid`.
4. The final thesis claim should therefore emphasize closeness, calibration, explanation, and auditability rather than claiming a universal federated win.

#### How this chart changes under different settings

Across `Centralized` and `Federated`, the gap remains small but real. Across `Federated` and `Average Local-Only`, the relation depends on the chosen slice. Local gradient boosting can be very strong under `iid`, while federated MLP and calibrated logistic rows become more attractive when the thesis asks for one shared cross-site framework rather than independent site-specific models.

Across `iid` and `non_iid`, the broad federated performance remains surprisingly stable in the strongest rows, which strengthens the robustness part of the thesis argument. Across `3` and `5` sites, performance changes are modest compared with the corresponding increase in communication cost.

#### What this chart implies for the thesis

This chart supports the core thesis result in one sentence: **Centralized remains the raw upper bound, but Federated stays close while adding calibration-aware, subgroup-aware, explainable, and communication-aware evidence that isolated Local-Only baselines do not provide as cleanly in one shared framework.**

#### What this chart does not prove

This chart does not prove that Federated is always the best choice for every institutional setting. It only proves that it is technically competitive and audit-ready in the current public-data framework.

#### How this chart was computed and plotted

This chart is built from [dashboard_metrics.csv](C:/Users/praha/Desktop/Joshna/results/full_cdc_polished_summary/dashboard_metrics.csv). The grouping is primarily `run_type`, together with any active model, calibration, site, and partition filters. The plotted values are setup-level grouped means such as `f1_mean`, while companion values like `roc_auc_mean`, `pr_auc_mean`, and `ece_mean` provide additional context in the chapter text and dashboard tooltips. Because the chart comes from the broad audit package, it is the correct source for the final comparison claims across `Centralized`, `Local-Only`, `Average Local-Only`, and `Federated`.

## 9. Overall Results Synthesis

The results in this chapter support a clear and defensible final synthesis.

First, the strongest `Centralized` result remains the pooled-data reference. In the full-CDC package, `Centralized / XGBoost / None / calib_f1_optimal` provides the strongest headline combination of `Ranking Quality (AUROC) = 0.830`, `Positive-Case Ranking (PR AUC) = 0.430`, and `Balanced Detection Score (F1) = 0.470`. This is the correct upper-bound reference for the project.

Second, the strongest `Federated` result stays close to that reference. The dashboard headline `Federated` slice, `FedAvg / Shallow Neural Network / Federated Isotonic / calib_f1_optimal`, reaches `AUROC = 0.826`, `PR AUC = 0.416`, and `F1 = 0.467`, leaving only a small headline gap. The broad audit package and the visual verification package differ slightly in scope, but they tell the same broad story: federated performance remains near the centralized reference rather than collapsing under the site-split design.

Third, calibration changes the practical meaning of the model output. Raw `None` probabilities often preserve strong AUROC but much worse `Risk Match Error (ECE)`. `Sigmoid` improves probability quality substantially, and `Isotonic` or `Global Isotonic` provides the strongest full-CDC probability-quality rows. The strongest exported probability-quality slice is `Federated / FedAvg / Logistic / Global Isotonic / calib_f1_optimal`, with `ECE = 0.003`, `Brier = 0.099`, and `Log Loss = 0.319`.

Fourth, threshold choice changes the story dramatically. For both centralized and federated rows, moving from `fixed_0p5` to `calib_f1_optimal` can raise `Balanced Detection Score (F1)` by a large margin while changing precision, recall, specificity, and subgroup gaps. This is why the dashboard and the Results chapter treat the `Decision Rule` as a first-class part of the result package.

Fifth, subgroup disparity is not uniform across group definitions. The full-CDC results show larger visible subgroup differences for `bmi_category` and `age_group` than for `sex` in the representative federated fairness slices. The strongest exported fairness-conscious row comes from `Local-Only / Random Forest / Isotonic / calib_f1_optimal / 3 sites / non_iid / alpha 0.1` with `F1 = 0.429`, `Positive Rate Gap = 0.013`, and `Error Gap Between Groups = 0.013`. This does not overturn the overall federated story, but it does show why fairness auditing must remain explicit rather than assumed.

Sixth, explainability and explanation stability remain meaningful additions rather than decorative extras. `GenHlth` repeatedly emerges as the dominant global feature in the federated MLP attribution summaries, while representative local explanation rows repeatedly surface variables such as `Age`, `HighBP`, `GenHlth`, and `HighChol`. Round-wise explanation stability rises quickly toward 1.0 in the representative federated logistic slices, and cross-site top-k overlap reaches 1.0 in the saved illustrative `3`-site `iid` slice.

Seventh, communication cost is real and measurable. The project therefore avoids claiming that Federated is automatically "more efficient" in a simplistic sense. Instead, it shows the tradeoff directly: the most communication-efficient competitive federated choice reaches about `31,680` cumulative bytes while retaining `AUROC = 0.821`, and increasing the number of sites raises the cumulative communication substantially.

Taken together, these findings support the thesis claim in its intended form. The thesis does **not** show that federated learning universally outperforms centralized pooled-data training. It shows something more careful and more defensible: a public-data federated diabetes prediction framework can remain close to the centralized reference while also providing calibrated probabilities, explanation summaries, explanation-stability checks, subgroup disparity analysis, and communication-cost evidence inside one reproducible audit surface.

The final full-CDC package therefore supports the following results-only summary:

- **Best for ranking:** `Centralized / XGBoost / None / calib_f1_optimal`
- **Best for balanced detection:** `Centralized / XGBoost / None / calib_f1_optimal`
- **Best for low false positives:** `Federated / FedProx / Logistic / Federated Isotonic / fixed_0p5`
- **Best for probability quality:** `Federated / FedAvg / Logistic / Global Isotonic / calib_f1_optimal`
- **Best federated close-to-centralized slice:** `Federated / FedAvg / Shallow Neural Network / Federated Isotonic / calib_f1_optimal`
- **Best communication-efficient federated choice:** `Federated / FedProx / Logistic / None / calib_f1_optimal / 3 sites / non_iid / alpha 1.0`

This is the main empirical conclusion of the Results chapter. The strongest central result remains the pooled-data upper bound, but the federated framework remains close enough, calibrated enough, explainable enough, and auditable enough to justify the thesis contribution as a trustworthy federated diabetes prediction framework rather than as a simple privacy-versus-accuracy exercise.
