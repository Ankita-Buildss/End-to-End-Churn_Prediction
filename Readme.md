---
title: "Customer Churn Prediction System"
output: html_document
---

## End-to-End ANN & Streamlit Deployment  
### Bank Customer Churn Analysis and Prediction

---

## Business Impact
Customer churn leads to revenue loss and higher acquisition costs. Predicting churn helps businesses take proactive retention actions and improve profitability.

- Enables proactive retention strategies  
- Reduces revenue loss  
- Improves customer engagement  
- Supports data-driven decision-making  

---

## Objective
Develop a machine learning system using **Artificial Neural Networks (ANN)** to predict customer churn and identify high-risk customers.

---

## Dataset & Use Case
The dataset contains **bank customer data**, including demographics, account details, and transaction-related features.

**Use Case:** Predict whether a customer will leave the bank (churn) or stay.

- **Target Variable (Exited):**  
  - 1 → Customer will leave (churn)  
  - 0 → Customer will stay  

---

## Architecture

![Churn Pipeline](images/churn/flowchart.png)

**Workflow:** Data preprocessing → ANN training → Prediction → Streamlit deployment  

---

## Methodology

- Data preprocessing (encoding + scaling)  
- Feature selection  
- ANN model training  
- Streamlit deployment for real-time prediction  
- Evaluation using Accuracy, Precision, Recall, F1-score, ROC-AUC  

---

## Tech Stack

- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Streamlit  
- **Model:** ANN  
- **Deployment:** Streamlit  

---

## Results

- High accuracy in churn prediction  
- Real-time predictions via Streamlit  
- Identified high-risk customers  
- Improved model performance  

---

## Model Performance

![Result 1](images/churn/result1.png)
![Result 2](images/churn/result2.png)

![Result 3](images/churn/result3.png)
![Result 4](images/churn/result4.png)

---

## Challenges

- Handling class imbalance  
- Feature selection  
- Preventing overfitting  
- Designing user-friendly UI  

---

## Future Scope

- Cloud deployment with API  
- Add more behavioral features  
- Automated churn alerts  
- Explore advanced models  

---

## Demo Video

<!-- Add your video here -->
<!-- Example:
<video width="700" controls>
  <source src="videos/churn_demo.mp4" type="video/mp4">
</video>
-->

---

© 2026 Ankita Shelke