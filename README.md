# 🏠 Tehran House Price Prediction using Deep Learning

An AI-powered project to predict real estate prices in Tehran using Deep Learning (Neural Networks).
This model analyzes over **230,000 real estate ads** from 2023 and 2024 to estimate property values with high accuracy.

## 🚀 Project Overview
Finding the "fair price" in Tehran's volatile housing market is challenging. This project leverages **Machine Learning** and **Deep Learning** to discover patterns between location, age, amenities, and price.

**Key Features:**
- **Data Cleaning:** Handling missing values, outliers, and Persian text processing.
- **Feature Engineering:** One-Hot Encoding for neighborhoods and Scaling for Neural Networks.
- **Model:** Scikit-Learn `MLPRegressor` (Multi-Layer Perceptron).
- **Performance:** Achieved an **R2 Score of ~89%** on test data.

## 📊 Results
The model successfully improved accuracy significantly compared to traditional Linear Regression.

| Model | R2 Score | MAE (Mean Absolute Error) |
| :--- | :--- | :--- |
| Linear Regression | 79% | 4.2 Billion Tomans |
| **Deep Learning (MLP)** | **88.68%** | **2.75 Billion Tomans** |

## 🛠️ Installation & Usage
1. Clone the repo:
```bash
git clone [https://github.com/YOUR_USERNAME/Tehran-House-Price-Prediction.git](https://github.com/YOUR_USERNAME/Tehran-House-Price-Prediction.git)
