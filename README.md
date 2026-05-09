# Lead_Conversion_Prediction_Model

This project builds a machine learning model to predict which leads are likely to convert into potential customers. By identifying high-potential leads early, businesses can allocate sales and marketing resources more efficiently and improve conversion performance.

The project focuses on comparing Decision Tree and Random Forest classifiers to determine the most effective predictive model for lead conversion analysis.

---

# Project Overview

This project builds a **machine learning classification model** to predict whether a lead will convert into a customer based on engagement behavior, demographic information, and interaction patterns.

### What Problem Does It Solve?

Companies often receive a large number of leads from multiple marketing channels but have limited sales resources. This project helps:

* **Identify** high-conversion leads
* **Prioritize** leads for sales follow-up
* **Improve** marketing efficiency
* **Reduce** wasted effort on low-quality leads
* **Increase** overall conversion performance

### How Does It Help?

Using machine learning predictions, businesses can:

* ✅ Focus on leads with the highest conversion probability
* ✅ Improve lead qualification processes
* ✅ Increase sales efficiency
* ✅ Optimize marketing channel performance
* ✅ Better understand customer conversion behavior

---

# Objective

This project aims to:

### Primary Goal

Build a machine learning model capable of accurately predicting lead conversion.

### Secondary Goals

1. Identify the most important factors influencing conversion
2. Compare Decision Tree and Random Forest performance
3. Reduce model overfitting through tuning and regularization
4. Improve minority class prediction using balancing techniques
5. Generate interpretable business insights from feature importance

### Success Criteria

| Metric             | Importance                                       |
| ------------------ | ------------------------------------------------ |
| **Recall**         | Identify as many potential customers as possible |
| **Precision**      | Keep false positives manageable                  |
| **F1-Score**       | Balance recall and precision                     |
| **ROC-AUC**        | Evaluate overall classification capability       |
| **Generalization** | Reduce overfitting between train and test sets   |

---

# Dataset Description

### Dataset Overview

* **Total Records**: ~4,600 leads
* **Target Variable**: `status`
* **Class Distribution**:

  * 0 → Not Converted
  * 1 → Converted
* **Problem Type**: Binary Classification

### Important Features

#### User Engagement Features

| Feature                 | Description                     |
| ----------------------- | ------------------------------- |
| `time_spent_on_website` | Total time spent on the website |
| `page_views_per_visit`  | Average pages viewed per visit  |
| `website_visits`        | Number of visits                |

#### Interaction Features

| Feature             | Description                      |
| ------------------- | -------------------------------- |
| `first_interaction` | Initial contact channel          |
| `last_activity`     | Most recent customer interaction |

#### Profile Features

| Feature              | Description                 |
| -------------------- | --------------------------- |
| `profile_completed`  | Level of profile completion |
| `current_occupation` | Occupation category         |
| `age`                | Customer age                |

#### Marketing Features

| Feature                | Description                   |
| ---------------------- | ----------------------------- |
| `referral`             | Referral status               |
| `digital_media`        | Digital media interaction     |
| `educational_channels` | Educational platform exposure |

---

# Project Workflow

## 1. Data Preprocessing

* Removed irrelevant identifier columns
* Handled categorical variables using one-hot encoding
* Split data into training and testing sets
* Used stratification to preserve class balance
* Applied balancing techniques for minority class improvement

---

## 2. Exploratory Data Analysis (EDA)

EDA was performed to:

* Analyze feature distributions
* Detect imbalance
* Study customer behavior patterns
* Identify influential variables
* Explore correlations between features

---

## 3. Decision Tree Classifier

A Decision Tree classifier was initially trained using default parameters.

### Observations

* The tree became very deep and complex
* Training accuracy was extremely high
* Strong overfitting was observed

### Improvements

Hyperparameter tuning was performed using:

* `GridSearchCV`

Tuned parameters included:

* `max_depth`
* `min_samples_split`
* `min_samples_leaf`

This reduced overfitting and improved model generalization.

---

## 4. Random Forest Classifier

A Random Forest classifier was implemented to improve predictive performance and reduce variance.

### Advantages

* Combines multiple decision trees
* Reduces overfitting
* Improves stability
* Better generalization on unseen data

### Model Optimization

* Cross-validation was applied
* Recall was used as the primary scoring metric
* Balancing techniques improved minority class prediction
* Regularization reduced train-test performance gap

---

# Model Performance

## Final Random Forest Results

| Metric       | Score |
| ------------ | ----- |
| **Accuracy** | 0.83  |
| **Recall**   | 0.80  |
| **F1-Score** | 0.74  |
| **ROC-AUC**  | 0.892 |

### Why Random Forest Was Selected

✅ Better generalization
✅ Lower overfitting
✅ Strong recall performance
✅ Better minority class detection
✅ More stable predictions
✅ Strong ROC-AUC performance

---

# Feature Importance

Feature importance analysis revealed that the most influential predictors were:

1. `time_spent_on_website`
2. `page_views_per_visit`
3. `first_interaction_Website`
4. `age`
5. `profile_completed`

### Key Insight

User engagement variables were the strongest indicators of lead conversion. Leads spending more time on the platform and interacting more frequently were significantly more likely to convert into customers.

---

# Technologies Used

| Technology       | Purpose                   |
| ---------------- | ------------------------- |
| Python           | Programming language      |
| Pandas           | Data manipulation         |
| NumPy            | Numerical operations      |
| Scikit-learn     | Machine learning          |
| Matplotlib       | Data visualization        |
| Seaborn          | Statistical visualization |
| Jupyter Notebook | Interactive analysis      |

---

# Project Structure

```bash
Lead_Conversion_Prediction/
│
├── README.md
├── Potential_Customers_Prediction.ipynb
├── data/
│   └── dataset.csv
├── outputs/
│   ├── feature_importance.png
│   ├── confusion_matrix.png
│   └── evaluation_metrics.txt
└── requirements.txt
```

---

# Installation & Setup

## Prerequisites

* Python 3.10+
* Jupyter Notebook
* pip

---

## Install Required Packages

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

---

## Run the Notebook

```bash
jupyter notebook
```

Open:

```bash
Potential_Customers_Prediction.ipynb
```

---

# Quick Start

## Train the Model

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=15,
    min_samples_leaf=8,
    random_state=1
)

rf.fit(X_train, y_train)
```

---

## Generate Predictions

```python
y_pred = rf.predict(X_test)
```

---

## Evaluate Model

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

---

# Key Findings

### 1. User Engagement is Critical

Time spent on the website and page views were the strongest predictors of conversion.

### 2. Balanced Models Improve Recall

Balancing techniques significantly improved detection of converted leads.

### 3. Random Forest Outperformed Decision Tree

The Random Forest model provided:

* better stability,
* lower overfitting,
* and improved predictive performance.

### 4. Generalization Improved After Regularization

Reducing model complexity produced closely aligned train and test metrics.

---

# Business Recommendations

## Recommendation 1: Prioritize High-Engagement Leads

Focus sales efforts on users with:

* high website activity,
* multiple page visits,
* and strong interaction behavior.

---

## Recommendation 2: Improve User Engagement

Enhancing website experience may directly improve lead conversion probability.

---

## Recommendation 3: Use Predictive Lead Scoring

Deploy the model to:

* rank incoming leads,
* prioritize sales follow-ups,
* and optimize conversion strategies.

---

# Conclusion

The Random Forest classifier was selected as the final model because it achieved the best balance between recall, accuracy, and generalization performance.

The model successfully identified potential customers while maintaining low overfitting and strong predictive capability. Feature importance analysis further demonstrated that customer engagement behavior plays a major role in conversion prediction.

Overall, this project demonstrates how machine learning can support lead prioritization and improve business decision-making through predictive analytics.

---

# Future Improvements

Potential future enhancements include:

* XGBoost implementation
* Threshold optimization
* SHAP value interpretation
* Model deployment with FastAPI or Flask
* Real-time lead scoring dashboard
* Hyperparameter optimization using RandomizedSearchCV

---

# License

This project is for educational and portfolio purposes.

---

# Author

Machine Learning Classification Project
Lead Conversion Prediction using Decision Tree & Random Forest
