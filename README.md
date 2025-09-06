# Stock Price Predictor

This project predicts short-term stock price changes using intraday data and an iterative feature mutation process. It simulates an “evolutionary” approach: generating features, mutating them with mathematical transformations, selecting the best-performing models, and repeating the process.

---

## 📌 Features
- **Data Fetching**  
  Pulls 1-minute bar data for a chosen ticker (default: GOOGL) from the [Polygon.io](https://polygon.io) API.  

- **Target Variable**  
  Percent change in price 30 minutes into the future.  

- **Feature Engineering**  
  - Lagged percent changes (1–380 minutes)  
  - Transformations: logarithms, polynomials, square roots  
  - Pairwise interactions (adjacent and random)  

- **Model Tournament**  
  - Trains linear regression models on multiple feature sets  
  - Evaluates models using **Mean Squared Error (MSE)** and **Polarity Accuracy** (direction correctness)  
  - Selects top performers and mutates their features across iterations  

---
## Sample Output

<img width="1110" height="282" alt="Screenshot 2025-09-06 at 14 53 40" src="https://github.com/user-attachments/assets/ebf1ad18-ae4c-487f-958d-9dde2f16fd8e" />


<img width="643" height="70" alt="Screenshot 2025-08-28 at 18 16 00" src="https://github.com/user-attachments/assets/6441f92c-44ae-4584-8a55-5d8374c4f3a8" />

---

## 📂 Project Structure
├── main.py # Driver script (fetches data, runs modeling loop)

├── model_operations.py # Helper functions (feature mutations & evaluation)

└── README.md # Project documentation



