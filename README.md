# Credit Risk Modeling - Bati Bank

## 📌 Credit Scoring Business Understanding

### 1. Basel II and Model Interpretability
The Basel II Accord emphasizes the need for banks to use advanced risk management practices to assess credit risk. Therefore, any model developed must be transparent, auditable, and interpretable—particularly when it influences financial decisions and regulatory capital requirements.

### 2. Importance of Proxy Variables
As our dataset lacks an explicit "default" label, we must engineer a proxy variable (e.g., using behavioral metrics like Recency, Frequency, and Monetary value). This proxy allows us to approximate credit risk. However, relying on proxies introduces the risk of false classification and biased decision-making.

### 3. Trade-offs: Simple vs Complex Models
- **Simple models (e.g., Logistic Regression with WoE)**: Highly interpretable, easy to audit, preferred in regulatory settings.
- **Complex models (e.g., Gradient Boosting, Random Forest)**: Better predictive performance, harder to explain. Risk of regulatory scrutiny unless explainability techniques are used (e.g., SHAP).
