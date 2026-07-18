# Email Threat Classifier with Differential Privacy

A pipeline that classifies emails as **Ham / Spam / Phishing** and compares standard
(non-private) models against several **differentially private (DP)** training strategies,
to measure the real-world privacy-utility tradeoff on high-dimensional text data.

## What this does

1. **Feature extraction** — hand-built features from subject/body text (keyword density,
   URL/HTML structure, urgency signals, etc.) plus TF-IDF.
2. **Non-DP baselines** — ExtraTrees, RandomForest, XGBoost, VotingClassifier.
3. **DP strategies** — compared side by side:
   - DP Logistic Regression (diffprivlib)
   - DP-ANN trained with DP-SGD (Opacus)
   - DP Dual-Layer ANN
   - PCA + Improved DP-ANN (dimensionality reduction before DP training)
   - PATE (teacher ensemble + noisy aggregation)
   - DP Ensemble voting
4. **Evaluation** — accuracy/F1 comparison across privacy budgets (ε), confusion
   matrices, and a privacy-utility tradeoff curve.

## Key findings

- Best non-DP model: **~98% accuracy**, near-perfect phishing recall.
- Traditional DP methods (DP LogReg/NaiveBayes/RF) **fail** on this high-dimensional
  data — accuracy collapses toward random guessing.
- **DP-SGD with neural nets (Opacus)** is the only DP approach that stays usable:
  ~93–95% at ε=10, ~85–90% at ε=1.
- Privacy cost is roughly **3–13% accuracy**, scaling with how strict ε is.
- **DP-ANN at ε=5.0** offers the best privacy-utility tradeoff for this task.

## Setup

```bash
pip install -r requirements.txt
```

Requires a `final_dataset.csv` with `subject`, `body`, and `label` columns
(0 = Ham, 1 = Spam, 2 = Phishing). This dataset is not included in the repo.

## Run

The script was originally built as a single Colab cell (`email_classifier_fixed.py`).
Update the `filepath` in Step 1 to point to your local dataset, then run:

```bash
python email_classifier_fixed.py
```

Outputs: accuracy/F1 tables in console, plus saved plots
(`final_comparison.png`, `final_confusion.png`, `final_privacy_utility.png`).

## Notes

- Uses `opacus` for DP-SGD, so per-sample gradient computation makes DP-ANN
  training noticeably slower than standard training — this is expected.
- No data leakage: TF-IDF, scaling, and feature selection are fit on train data only.
