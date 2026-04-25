# Thesis Conclusion Summary

This summary is generated from the active evidence package and keeps every claim tied to the real saved metrics.

## Main Thesis Statement

- Centralized training is the raw upper-bound reference for predictive performance on the active public dataset package.
- Federated learning gets close to that Centralized reference while preserving data locality and outperforming isolated Local-Only baselines in the broader audit.
- Different operating points are best for different goals: ranking quality, balanced detection, low false positives, calibrated probabilities, fairness tradeoffs, and communication efficiency.

## CDC

- **Best for ranking:** Centralized / XGBoost / None / Use F1-Tuned Cutoff with AUROC 0.830, PR AUC 0.430, and F1 0.470.
- **Best for balanced detection:** Centralized / XGBoost / None / Use F1-Tuned Cutoff with balanced detection 0.470, ranking quality 0.830, and decision rule 0.416.
- **Best for low false positives:** Federated / FedProx / Logistic / Federated Isotonic / Use 50% Risk Cutoff with specificity 0.994, false-positive rate 0.006, and decision rule nan.
- **Best for trustworthy probabilities:** Federated / FedAvg / Logistic / Global Isotonic / Use F1-Tuned Cutoff with risk-match error 0.003, probability error 0.099, and confidence penalty 0.319.
- **Best federated close-to-centralized slice:** Federated / FedAvg / Shallow Neural Network / Federated Isotonic / Use F1-Tuned Cutoff with ranking gap -0.004 and balanced-detection gap -0.003 versus the strongest Centralized slice.
- **Best group-aware decision rule:** Local-Only / Random Forest / Isotonic / Use F1-Tuned Cutoff with decision rule 0.065, balanced detection 0.429, positive-rate gap 0.013, and error gap 0.013.
- **Best explanation stability slice:** Federated / FedProx / Logistic / None / Use 50% Risk Cutoff with Explanation Stability 1.000.
- **Most communication-efficient competitive federated choice:** Federated / FedProx / Logistic / None / Use F1-Tuned Cutoff with cumulative communication 31680.000 bytes while keeping ranking quality 0.821.

## PIMA

- **Best for low false positives:** Federated / FedProx / Logistic / Federated Sigmoid / Use 50% Risk Cutoff with specificity 1.000, false-positive rate 0.000, and decision rule nan.

## Interpretation

- The strongest honest story is not that Federated learning universally beats Centralized training.
- The strongest honest story is that Federated learning can stay close to Centralized performance while adding calibration, fairness, explainability, and communication-aware evidence that isolated Local-Only models do not provide as cleanly.
- Threshold choice materially changes recall, specificity, and false-positive behavior, so any final recommendation should name the operating threshold together with the metric it optimizes.
