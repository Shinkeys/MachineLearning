# Sine Function Approximation with Neural Networks

## Description:
A PyTorch-based neural network project that demonstrates function approximation using a Multi-Layer Perceptron (MLP). The model learns to approximate the sine function sin(x) through supervised learning, featuring:

### Architecture: 
4-layer MLP (1→16→16→1) with 2 hidden layers of 16 neurons each, ReLU activation, and Xavier initialization

### Training: 
- 512 training samples and 128 validation samples
- Mini-batch gradient descent (batch size: 64) with Adam optimizer
- Adaptive learning rate scheduling (ReduceLROnPlateau)
- Initial learning rate: 0.01

### Visualization: 
Real-time animated plotting showing:
- Training/validation data points and model predictions
- Live training and validation loss curves (MSE)
- Progressive learning visualization over 1000 epochs

### Technologies used: 
PyTorch, NumPy, Matplotlib

## Results
![Training Results](https://github.com/user-attachments/assets/b7f5f681-a72c-43a7-996f-8d7ded99a49f)
*1 hidden layer with 16 neurons*
