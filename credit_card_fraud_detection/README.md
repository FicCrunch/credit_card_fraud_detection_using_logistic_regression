# Fraudulent Credit Card Detection

## Project Overview
Built a machine learning model to detect fraudulent credit card transactions from a highly imbalanced dataset.  
The goal is to accurately identify fraudulent transactions while minimizing false negatives, since missing fraud can be costly.

## Dataset
- Real-world credit card transactions dataset  
- ~284,000 transactions, only ~0.17% fraud  
- Features: anonymized numeric values (V1–V28) + transaction Amount  
- Highly imbalanced → special handling needed

## Key Challenges
- Extreme class imbalance (fraud = 0.17%)  
- Small fraction of frauds → accuracy alone is misleading  
- Need for careful evaluation: precision, recall, ROC-AUC

## Approach
1. **Data preprocessing**  
   - Scaled `Amount` feature  
   - Stratified train/test split to preserve class distribution  

2. **Model**  
   - Logistic Regression with `class_weight='balanced'`  
   - Trained on training set, evaluated on test set  

3. **Evaluation Metrics**  
   - Precision, recall, F1-score  
   - Confusion matrix visualization  
   - ROC-AUC curve  

4. **Insights**  
   - Lowering probability threshold can improve recall for fraud detection  
   - Feature scaling and class weighting significantly improve performance  

## Results (Example)
- ROC-AUC Score: 0.97  
- Recall for fraud class: 0.78  
- Precision for fraud class: 0.85  

## Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  

## How to Run
1. Clone repo  
2. Install dependencies: `pip install -r requirements.txt`  
3. Run `notebook.ipynb` or scripts in `src/`  

## Key Takeaway
In fraud detection, **recall matters more than accuracy**. Handling class imbalance and proper feature scaling are crucial for reliable predictions.
