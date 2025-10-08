# Sine Function Approximation with Neural Networks
## Description:
A PyTorch-based neural network project that demonstrates function approximation using a Multi-Layer Perceptron (MLP). The model learns to approximate the sine function sin(x) through supervised learning, featuring:
### Architecture: 3-layer MLP (1→16→16→1) with ReLU activation and Xavier initialization
### Training: Mini-batch gradient descent with Adam optimizer and adaptive learning rate scheduling
### Visualization: Real-time animated plotting showing:
- Training/validation data points and model predictions
- Live training and validation loss curves (MSE)
- Progressive learning visualization over 1000 epochs
### Technologies used: PyTorch, NumPy, Matplotlib