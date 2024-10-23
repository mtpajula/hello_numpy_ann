
### **1. Loss Function**

**Where is the loss function calculated?**

```python
# Compute loss (mean squared error)
loss = np.mean((output - y) ** 2)
```

**Explanation:**

- **Loss Function:** The code uses the **Mean Squared Error (MSE)** loss function.
- **Calculation:** It computes the average of the squared differences between the predictions (`output`) and the true values (`y`).
- **Formula:** \( \text{Loss} = \frac{1}{n} \sum_{i=1}^{n} (y_{\text{predicted}}^{(i)} - y_{\text{true}}^{(i)})^2 \)

**Summary:**

- **The loss function in this code is Mean Squared Error (MSE).**

---

### **2. Optimizer**

**Where is the optimizer implemented?**

The optimizer is implicitly implemented in the `backward` function, where the model's weights are updated based on the gradients.

```python
def backward(self, d_output, learning_rate):
    # Backward pass through all layers in reverse order
    for layer in reversed(self.layers):
        d_output = layer.backward(d_output, learning_rate)
```

**Explanation:**

- **Gradient Calculation:** Gradients are computed with respect to the loss function concerning the model parameters.
- **Weight Updates:** Each layer updates its weights and biases using the learning rate (`learning_rate`) and the computed gradients.
- **Optimization Algorithm:** This implements **Stochastic Gradient Descent (SGD)**.

**Weight Update in the Layer's `backward` Method:**

Layer's backward-method typically includes logic like:

```python
def backward(self, d_output, learning_rate):
    # Compute gradients w.r.t weights and biases
    self.d_weights = np.dot(self.input.T, d_output)
    self.d_biases = np.sum(d_output, axis=0, keepdims=True)
    # Update weights and biases
    self.weights -= learning_rate * self.d_weights
    self.biases -= learning_rate * self.d_biases
    # Compute gradient w.r.t input for previous layers
    d_input = np.dot(d_output, self.weights.T)
    return d_input
```

**Summary:**

- **The optimizer in this code is Stochastic Gradient Descent (SGD) using the specified learning rate.**

---

### **3. Metrics**

**Where are the metrics calculated, and what metric is used?**

```python
# Compute loss (mean squared error)
loss = np.mean((output - y) ** 2)
self.losses.append(loss)
if epoch % 100 == 0 or epoch == epochs - 1:
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")
```

**Explanation:**

- **Metric:** The code uses the **loss function value** itself as the metric, specifically the Mean Squared Error (MSE) loss.
- **Tracking:** The loss is appended to a list `self.losses` at each epoch.
- **Output:** The loss value is printed periodically during training.

**Summary:**

- **The metric in this code is the Mean Squared Error (MSE), monitored during training.**

---

### **Comprehensive Summary:**

- **Loss Function:** Mean Squared Error (MSE)
  - Calculates the average squared difference between predicted and true values.
- **Optimizer:** Stochastic Gradient Descent (SGD)
  - Updates the model's weights based on the learning rate and gradients.
- **Metrics:** Mean Squared Error (MSE)
  - The loss function value is used as the metric to evaluate model performance during training.

---

**Why are these choices significant?**

- **Mean Squared Error (MSE):**
  - **Applicability:** Commonly used loss function for regression problems where continuous values are predicted.
  - **Sensitivity to Outliers:** MSE penalizes larger errors more than smaller ones, which can be beneficial for training.

- **Stochastic Gradient Descent (SGD):**
  - **Simplicity:** Easy to implement and understand.
  - **Efficiency:** Updates weights after each sample or mini-batch, which can speed up learning.
  - **Learning Rate (`learning_rate`):** Controls the step size during weight updates; too high can cause instability, too low can slow learning.

- **Metric Selection:**
  - **Monitoring the Loss Function:** Provides direct feedback on how well the model is minimizing the error during training.
  - **Simplicity:** MSE is straightforward to compute and interpret.

---

**What are the common alternatives for each of these?**

#### **Loss Functions:**

- **Mean Absolute Error (MAE):**
  - Measures the average absolute difference between predictions and true values.
- **Huber Loss:**
  - Combines properties of MSE and MAE; less sensitive to outliers than MSE.
- **Cross-Entropy Loss:**
  - Used in classification problems.

#### **Optimizers:**

- **Momentum:**
  - Enhances SGD by adding a momentum term to accelerate convergence.
- **AdaGrad:**
  - Adapts learning rates for each parameter individually.
- **RMSprop:**
  - Similar to AdaGrad but uses exponential decay.
- **Adam:**
  - Combines Momentum and RMSprop algorithms; adaptive learning rate.

#### **Metrics:**

- **Accuracy:**
  - Used in classification problems to measure the proportion of correct predictions.
- **Precision, Recall, F1-Score:**
  - Provide deeper insights in classification, especially with imbalanced datasets.
- **R^2 Score:**
  - Used in regression to measure the proportion of variance explained by the model.
- **Mean Absolute Percentage Error (MAPE):**
  - Expresses error as a percentage of the true values.
