# Part‑1: Neural Network Fundamentals and Training Behavior Analysis

##  Overview
This project demonstrates how a feed‑forward neural network learns using a structured **customer churn dataset**.  
The focus is on understanding the forward pass, loss calculation, backpropagation, parameter updates, and how hyperparameters affect training behavior.

---

##  Dataset : https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing
- **File:** `customer_churn_nn.csv`  
- **Records:** ~350 customers  
- **Features:**  
  - **Categorical:** region, plan_type, contract_type, payment_method, autopay_enabled  
  - **Numerical:** tenure_months, monthly_charges_inr, avg_login_days_per_month, support_tickets_last_90_days, payment_delay_days, data_usage_gb, satisfaction_score, discount_percent, referral_count  
- **Target:** `churn` (binary: 0 = retained, 1 = churned)

---

## ⚙️ Tasks
1. **Dataset Understanding**  
   - Explore rows/columns, feature types, missing values, statistical summary, target distribution.  
2. **Data Preprocessing**  
   - Encode categorical features, scale numerical features, split into train/test sets.  
3. **Model Building**  
   - Feed‑forward neural network with input, hidden, and output layers.  
   - ReLU activation in hidden layers, sigmoid in output.  
   - Loss: Binary Crossentropy, Optimizer: Adam.  
4. **Training & Evaluation**  
   - Train for 20 epochs, visualize accuracy/loss curves, compute confusion matrix.  
5. **Hyperparameter Experiments**  
   - Compare different architectures, learning rates, and epochs.  
6. **Reflection**  
   - Discuss weights, biases, activation functions, learning rate effects, underfitting/overfitting.

---

##  Model Architecture
| Layer   | Type   | Activation | Units |
|---------|--------|------------|-------|
| Input   | Dense  | —          | Encoded features |
| Hidden1 | Dense  | ReLU       | 32 |
| Hidden2 | Dense  | ReLU       | 16 |
| Output  | Dense  | Sigmoid    | 1 |

---

##  Results
- **Training Accuracy:** ~0.85  
- **Validation Accuracy:** ~0.82  
- **Precision:** 0.75 **Recall:** 0.67 **F1‑Score:** 0.71  
- **Observation:** Model generalizes well; slight underfitting due to churn imbalance.

**Artifacts:**
- `results/evaluation_outputs.png` → Accuracy/Loss curves + Confusion Matrix  
- `results/model_comparison_table.csv` → Hyperparameter experiment summary

---

##  Hyperparameter Comparison
| Experiment | Hidden Layers | Neurons | Learning Rate | Epochs | Test Accuracy | Observation |
|------------|---------------|---------|---------------|--------|---------------|-------------|
| Baseline   | 2             | 32/16   | 0.001         | 20     | 0.82          | Balanced |
| Exp 1      | 3             | 64/32/16| 0.001         | 30     | 0.84          | Slight improvement |
| Exp 2      | 2             | 32/16   | 0.01          | 20     | 0.78          | Too high LR |
| Exp 3      | 2             | 16/8    | 0.001         | 20     | 0.79          | Underfitting |

---

## Key Learnings
- **Weights & Biases:** Adjusted during backpropagation to minimize loss.  
- **Activation Functions:** Introduce non‑linearity; essential for learning complex patterns.  
- **Learning Rate:** Too high → unstable convergence; too low → slow learning.  
- **Overfitting vs Underfitting:** Deeper networks improved accuracy but risked overfitting.

---

## 📦 Requirements
See `requirements.txt`:
