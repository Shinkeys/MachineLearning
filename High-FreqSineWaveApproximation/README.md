# High-Frequency Sine Wave Approximation with Fourier Features

## Description:
A PyTorch-based neural network project that demonstrates high-frequency function approximation using a Multi-Layer Perceptron (MLP) enhanced with **Fourier Feature encoding**. The model successfully learns to approximate high-frequency sine waves (e.g., sin(32x)) that would be difficult or impossible for standard MLPs to capture due to spectral bias

### Key Idea: Fourier Feature Encoding
Traditional neural networks struggle with high-frequency signals due to spectral bias (tendency to learn low-frequency patterns first). This project solves this by transforming the input through Fourier features:
- Transforms scalar input `x` into multiple frequency components: `[sin(x), cos(x), sin(2x), cos(2x), sin(4x), cos(4x), ...]`
- Uses powers of 2 for frequency scaling: `[1, 2, 4, 8, 16, 32, ...]`
- Enables the network to directly access high-frequency basis functions

### Training: 
- **Dataset**: 512 training samples and 128 validation samples
- **Optimization**: Mini-batch gradient descent (batch size: 64) with Adam optimizer
- **Learning rate**: Initial LR of 0.01 with ReduceLROnPlateau scheduler (factor=0.7, patience=100)
- **Loss function**: Mean Squared Error (MSE)
- **Training duration**: 1000 epochs with real-time visualization

### Visualization: 
Real-time animated plotting showing:
- True high-frequency sine wave (e.g., sin(32x))
- Model predictions evolving during training
- Training and validation data points
- Dual-panel view with function approximation and loss curves
- Live metrics: Batch Loss, Smooth MSE, Validation MSE, and Learning Rate

### Technologies used: 
PyTorch, NumPy, Matplotlib

### Results:
![Training Results](https://github.com/user-attachments/assets/123f6fbc-cb7a-4fa4-b2e8-3926b9fd9b82)
