# Heart Disease Risk Prediction for Precision Underwriting

This project applies machine learning techniques to predict an individual's risk of heart attack using health survey data. It is aimed at improving **precision underwriting**, **product design**, and **personalized health management** for the insurance industry.

> 📍 Case Partner: Cathay Life Insurance  
> 📊 Dataset: CDC Behavioral Risk Factor Surveillance System (BRFSS)  

---

## 🎯 Business Problem

**How can we improve the accuracy and efficiency of cardiovascular insurance underwriting?**

Traditional underwriting primarily relies on age and pre-existing conditions. However, these methods:
- Fail to reflect nuanced personal health risks
- Lead to **information asymmetry and adverse selection**
- Miss early intervention opportunities

**Solution**: Build a machine learning model to predict heart disease risk, enabling:
- Precision risk segmentation
- Personalized pricing
- Smarter health engagement programs

---

## 🗃️ Data Overview

- Source: Public health survey (2022)
- Records: ~450,000 respondents
- Target variable: `HadHeartAttack` (Yes/No)

### Key Variables:
- Demographics: Age, Sex, State
- Health history: Angina, Diabetes, Arthritis
- Behavior: Sleep, Smoking, Drinking, Exercise
- Perception: General health rating, mental & physical health days
- Screening: Chest scan, Pneumonia vaccine, COVID test

---

## 🧹 Data Preprocessing

- **Missing values**:
  - Numerical: `IterativeImputer` + BayesianRidge
  - Categorical: Random Forest–based imputation
- **Outlier handling**: BMI, Height, Weight → IQR clipping
- **Feature engineering**:
  - BMI binning
  - LDA for supervised dimensionality reduction
- **Feature selection**:
  - Use IV > 0.02 + Random Forest importance > 0.01

---

## ⚖️ Class Imbalance Handling

- **Target**: Only ~8% of individuals had heart attacks
- Applied **SMOTEENN**:
  - SMOTE: Over-sample minority class
  - ENN: Remove noisy majority samples
- Used **stratified sampling** during train-test split

---

## 🤖 Models & Evaluation

Trained and tuned via `GridSearchCV`:

| Model              | Recall | Precision | F1-Score | Accuracy |
|--------------------|--------|-----------|----------|----------|
| Logistic Regression| 0.82   | 0.18      | 0.28     | 0.76     |
| XGBoost            | 0.51   | 0.47      | 0.49     | 0.94     |
| LightGBM           | 0.51   | 0.47      | 0.49     | 0.94     |
| KNN                | 0.63   | 0.22      | 0.33     | 0.84     |
| Random Forest      | 0.52   | 0.44      | 0.48     | 0.93     |
| SVM                | 0.75   | 0.19      | 0.30     | 0.79     |

> ⚠️ Trade-off observed: high recall models detect more true positives (important for **early intervention**), but precision remains a challenge.

---

## 📌 Application Scenarios

### 1. 🔍 Precision Underwriting & Pricing
Use predicted risk scores to:
- Adjust premiums
- Enable automatic approval for low-risk clients
- Require health checks or loading for high-risk applicants

### 2. 🎯 Product Design & Market Segmentation
- Segment clients by risk level
- Offer different products:
  - High risk → early protection plans
  - Low risk → long-term wellness incentives

### 3. 🧠 Personalized Health Engagement
- Recommend behavior change programs (exercise, diet, screening)
- Reward risk improvement with cashback or premium discount

### 4. 🛡️ Fraud Detection
- Cross-check predicted low-risk customers with high claims
- Trigger review or investigation if mismatch found

---

## 🧩 Key Insights

- **Self-perception** of health (e.g., general health rating) is a strong predictor
- **Chronic conditions** (e.g., diabetes, arthritis, angina) significantly increase risk
- **Physical limitations** (e.g., difficulty walking, hearing loss) also correlate with heart attacks
- Lifestyle indicators like **smoking**, **sleep duration**, and **BMI** matter

---

## 📈 Future Improvements

- Add wearable device data (e.g., heart rate, activity tracking)
- Use time-series models to capture behavioral trends
- Apply explainable AI (e.g., SHAP) for transparent underwriting
- Integrate real-world claim data for calibration

Chinese report：[MLAI final ppt.pptx](https://github.com/user-attachments/files/23201145/MLAI.final.ppt.pptx)
