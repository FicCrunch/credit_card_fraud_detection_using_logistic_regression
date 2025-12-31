## Explanation of Logistic Regression (BTS)

## Overview
This is a supervised regression technique used to determine or predict likelihoods of certain events happening(this case specifically)
This algorithm uses a sigmoid function:
    $$f(x) = \dfrac{1}{1 + e^{\beta_0 x + \beta_1}}$$

# Flow
## Initialize Weights
- all weights, $w_i$, are initialized to 0 

## Make predictions with initial weights
- weighted sum is calculated: $z = w_1*x_1 + w_2*x_2 * w_3*x_3 + ... + b = 0$
- Sigmoid function comes into play: $\sigma(z = 0) = Pr(z = 0)$, In this case the model says this is the probability of having a fraudulent transaction initially

## Loss Function comes in
- Each prediction is compared to actual labels $y_i$
- the loss function is calculated:
     $L = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]$
- This tells the model "How wrong am i?" or "How off am I?"

## Gradient Descent
- GOAL: We want weights that can produce the most minimum loss 👍
- the weights are adjusted
- Formula: $\frac{\partial L}{\partial w_j} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i) x_{ij}$
             ,where $ (\hat{y}_i - y_i)$ represents the difference the prediction and the true label, $x_{ij} represents how important that feature was for that weight$
- after gradient is produced, we nudge the weights: $$w_j = w_j - \eta \cdot \frac{\partial L}{\partial w_j}$$
