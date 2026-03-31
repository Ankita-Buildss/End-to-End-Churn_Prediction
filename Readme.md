# Bank Customer Churn Analysis and Prediction
## End-to-End ANN & Streamlit Deployment  

---

## Table of Contents
1. [Business Impact](#business-impact)  
2. [Objective](#objective)  
3. [Dataset & Use Case](#dataset--use-case)  
4. [Architecture](#architecture)  
5. [Methodology](#methodology)  
6. [Tech Stack](#tech-stack)  
7. [Results](#results)  
8. [Model Performance](#model-performance)  
9. [Challenges](#challenges)  
10. [Future Scope](#future-scope)  
11. [Demo Video](#demo-video)  

---

## Business Impact
Customer churn leads to revenue loss and higher acquisition costs. Predicting churn helps businesses take proactive retention actions and improve profitability.

- Enables proactive retention strategies  
- Reduces revenue loss  
- Improves customer engagement  
- Supports data-driven decision-making  

---

## Objective
Develop a machine learning system using **Artificial Neural Networks (ANN)** to predict customer churn and identify high-risk customers for timely retention actions.

---

## Dataset & Use Case
The dataset contains **bank customer data**, including demographics, account details, and transaction-related features.

**Use Case:** Predict whether a customer will leave the bank (churn) or stay.

- **Target Variable (Exited):**  
  - 1 → Customer will leave (churn)  
  - 0 → Customer will stay  

---

## Methodology

- Preprocess data (drop irrelevant columns, encode categorical features, scale numerical values)  
- Use all remaining features as predictors for ANN  
- Train **ANN model** for binary classification  
- Deploy using **Streamlit** for real-time prediction  
- Evaluate using Accuracy, Precision, Recall, F1-score, ROC-AUC  

---

## Tech Stack

- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Streamlit  
- **Model:** Artificial Neural Network (ANN)  
- **Deployment:** Streamlit  

---

## Results

- High accuracy in churn prediction  
- Real-time predictions via Streamlit  
- Identified high-risk customers  

---

## Challenges

- Handling class imbalance  
- Dropping irrelevant features and encoding categoricals  
- Preventing overfitting in ANN  
- Designing an intuitive user interface  

---

## Future Scope

- Cloud deployment with API access  
- Incorporate additional customer behavioral features  
- Automated alerts for at-risk customers  
- Explore ensemble or sequential models (RNNs)  

---

## Demo Video

<video width="700" controls>
  <source src="videos/churn_demo.mov" type="video/quicktime">
  Your browser does not support the video tag.
</video>

---

