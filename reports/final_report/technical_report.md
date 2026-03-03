# Credit Card Fraud Detection Using Ensemble Learning and Deep Learning Models

## Technical Report

---

## Abstract

Credit card fraud is a critical financial security concern, costing billions annually. This project investigates the effectiveness of machine learning approaches for identifying fraudulent transactions. We replicate an existing study that compares Gaussian Naive Bayes, Random Forest, and XGBoost classifiers on two datasets — the European Cardholders dataset (real-world, PCA-transformed) and the Sparkov simulated dataset. We evaluate three data-balancing strategies: random oversampling, random undersampling, and SMOTE. Building on these findings, we propose enhancements including a hybrid CNN-BiGRU deep-learning model, a DistilBERT-based classifier, a stacking meta-ensemble, SHAP/LIME explainability, and a risk-based multi-factor authentication simulation. Results confirm that ensemble methods significantly outperform the Naive Bayes baseline, SMOTE yields the best balancing performance, and the proposed stacking ensemble achieves the highest overall F1 scores.

---

## 1. Introduction

### 1.1 Background
The proliferation of electronic payment systems has led to a corresponding increase in credit card fraud. According to the Nilson Report, global card fraud losses exceeded $28 billion in 2020. Traditional rule-based detection systems suffer from high false-positive rates and inability to adapt to evolving fraud patterns.

### 1.2 Problem Statement
Given highly imbalanced transaction datasets (fraud typically < 1% of transactions), the challenge is to build classifiers that maximise fraud recall without overwhelming legitimate cardholders with false alerts.

### 1.3 Objectives
1. Replicate the existing paper's methodology: train NB, RF, and XGB on two datasets with three balancing strategies.
2. Validate the finding that ensemble methods outperform baselines and that SMOTE is the optimal balancing strategy.
3. Propose and evaluate deep-learning models (CNN-BiGRU, DistilBERT) and a stacking ensemble.
4. Implement explainability (SHAP, LIME) and a 2FA/MFA simulation framework.

---

## 2. Literature Review

### 2.1 Existing Approaches
- **Statistical methods**: Logistic regression, Naive Bayes — fast but limited in capturing complex patterns.
- **Ensemble methods**: Random Forest (bagging), XGBoost/AdaBoost (boosting) — consistently high performance on tabular fraud data.
- **Deep learning**: CNNs, RNNs, autoencoders — excel at capturing sequential and non-linear patterns.

### 2.2 Gap Analysis
- Most studies evaluate on a single dataset; cross-dataset validation is rare.
- Explainability (XAI) is largely absent from fraud detection literature.
- Remedial actions (authentication layers) are not simulated alongside detection.

### 2.3 Proposed Enhancements
This project addresses these gaps by: (a) evaluating across two distinct datasets, (b) incorporating SHAP/LIME for transparency, and (c) simulating risk-based 2FA/MFA.

---

## 3. Methodology

### 3.1 Data Collection and Preprocessing

**European Dataset**: 284,807 transactions (492 fraud, 0.172%). Features V1–V28 are PCA-transformed. `Amount` and `Time` are scaled with StandardScaler.

**Sparkov Dataset**: ~1.85M simulated transactions. Irrelevant columns (names, addresses, card numbers) are dropped. Temporal features (year, month, day, hour, day-of-week) are extracted from timestamps. `category` and `job` are label-encoded. All numerics are standardised.

### 3.2 Handling Imbalanced Data
Three strategies applied to training data only:
1. **Random Oversampling** — duplicate minority samples.
2. **Random Undersampling** — remove majority samples.
3. **SMOTE** — synthesise minority samples via k-NN interpolation.

### 3.3 Existing Models (Paper Replication)
- **Naive Bayes** (GaussianNB) — θ(Nd)
- **Random Forest** — n_estimators=5, max_depth=3, max_samples=0.2, max_features=0.3 — θ(t·d·n·log n)
- **XGBoost** — n_estimators=5, max_depth=3 — O(t·d·log n)

### 3.4 Proposed Models
- **CNN-BiGRU** — Conv1D(64) → MaxPool → BiGRU(64) → Dropout(0.5) → Dense(32) → Sigmoid
- **DistilBERT** — tabular features → text → fine-tuned DistilBERT for sequence classification
- **Stacking Ensemble** — RF + XGB base learners → Logistic Regression meta-learner

### 3.5 Explainability Techniques
- **SHAP**: TreeExplainer for tree models; summary, bar, and force plots.
- **LIME**: Local perturbation-based explanations for individual predictions.
- **Permutation Importance**: Model-agnostic global feature ranking.

### 3.6 2FA/MFA Simulation Framework
Three-layer authentication: Standard (low risk) → 2FA SMS/Email (medium risk) → MFA Biometric (high risk). Thresholds are tunable and driven by model risk scores.

---

## 4. Experimental Setup

### 4.1 Hardware / Software
- Python 3.8+, scikit-learn, XGBoost, TensorFlow, PyTorch, HuggingFace Transformers
- GPU recommended for CNN-BiGRU and BERT training

### 4.2 Evaluation Metrics
- Accuracy, Precision, Recall, F1 Score, ROC-AUC, Confusion Matrix

### 4.3 Cross-Validation Strategy
- 5-fold KFold cross-validation (stratified)
- Data split: 70% train / 15% validation / 15% test

---

## 5. Results and Discussion

### 5.1 Replication of Existing Work
Results replicate the paper's findings:
- Ensemble methods (RF, XGB) consistently outperform Naive Bayes across all dataset–strategy combinations.
- SMOTE provides the best recall and F1 balance.
- Models perform significantly better on the European dataset (deterministic real-world patterns) than the Sparkov dataset (stochastic simulated data).

### 5.2 Proposed Models Performance
- **Stacking Ensemble** and **Tuned XGB** achieve the highest F1 and AUC scores.
- **CNN-BiGRU** performs competitively, especially on the European dataset.
- **BERT** is limited by the artificial tabular→text conversion but demonstrates the transformer architecture's feasibility.

### 5.3 Comparative Analysis
All models are compared in comprehensive tables and heatmaps. The stacking meta-ensemble leverages the complementary strengths of RF and XGB.

### 5.4 Statistical Significance
- Paired t-tests confirm significant performance differences between ensemble models and Naive Bayes.
- Friedman test with Nemenyi post-hoc validates non-equal performance across classifiers.

### 5.5 Explainability Analysis
- SHAP reveals that V14, V17, V12 (PCA features) are the most important predictors for the European dataset.
- LIME explanations for individual transactions provide actionable insights for fraud analysts.

---

## 6. Conclusion and Future Work

### Summary of Findings
1. Ensemble learning (especially XGBoost with SMOTE) is highly effective for credit card fraud detection.
2. Stacking meta-ensembles further improve performance by combining model strengths.
3. SHAP and LIME provide the transparency required for regulatory compliance.
4. Risk-based 2FA/MFA can reduce fraud impact without degrading user experience for low-risk transactions.

### Limitations
- BERT on tabular data is suboptimal; purpose-built tabular transformers (TabNet, FT-Transformer) would be more appropriate.
- Evaluation is limited to offline batch processing; real-time streaming evaluation was not performed.
- Dataset availability limits generalisation claims.

### Future Research Directions
1. Real-time streaming fraud detection (Kafka + ML serving)
2. Federated learning for privacy-preserving cross-bank model training
3. Graph neural networks for transaction network analysis
4. Adversarial robustness testing against evolving fraud strategies

---

## References

1. ULB Machine Learning Group. Credit Card Fraud Detection Dataset. Kaggle.
2. Sparkov Data Generation. Simulated Credit Card Transactions. Kaggle.
3. Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. ACM KDD.
4. Breiman, L. (2001). Random Forests. Machine Learning, 45, 5–32.
5. Chawla, N. V. et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. JAIR, 16, 321–357.
6. Lundberg, S. M. & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.
7. Ribeiro, M. T. et al. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. ACM KDD.
