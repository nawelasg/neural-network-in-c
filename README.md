# MNIST Digit Recognition Model

## Overview
This project implements a neural network model from scratch in C to classify handwritten digits from the MNIST dataset. The model consists of a fully connected feedforward neural network with one hidden layer using ReLU activation and a softmax output layer.

The program trains the network using stochastic gradient descent with momentum and evaluates the accuracy on the test set.

## Features
- Supports configurable hyperparameters, including learning rate, batch size, and the number of epochs.
- Uses ReLU activation for the hidden layer and softmax for the output layer.
- Implements momentum-based optimization for faster convergence.
- Includes shuffling of the training data to improve training.

## Requirements
To compile and run this code, you need:
- A GCC compiler supporting C99 or later
- The MNIST dataset files: `train-images.idx3-ubyte` and `train-labels.idx1-ubyte`
- A system capable of running C programs

## Files
- **`MNIST_Recog.c`**: The source code containing the implementation of the neural network.
- **MNIST Data Files**: Required for input:
  - `train-images.idx3-ubyte`
  - `train-labels.idx1-ubyte`

## Compilation and Execution

### Compilation
To compile the program, use the following command:
```bash
gcc -O3 -march=native -ffast-math -o nn MNIST_Recog.c -lm
```

### Execution
To run the program, use the following command:
```bash
./nn.exe
```
Ensure that the MNIST data files are in the `data/` directory.

## How It Works

### Architecture
1. **Input Layer**: 784 nodes (28x28 pixels per image)
2. **Hidden Layer**: 256 nodes with ReLU activation
3. **Output Layer**: 10 nodes with softmax activation

### Training
- The network is trained on the MNIST dataset using mini-batches.
- The loss function used is cross-entropy.
- Stochastic gradient descent with momentum is employed for optimization.

### Testing
- The trained model is evaluated on the test set (20% of the MNIST dataset).
- The accuracy is calculated and displayed after every epoch.

## Code Walkthrough

### Key Components
1. **Neural Network Structure**:
   - Defined using `Layer` and `Network` structures.
   - Layers are initialized with weights, biases, and momentum arrays.

2. **Forward Propagation**:
   - Implemented in the `forward` function.
   - Includes matrix-vector multiplication and ReLU activation.

3. **Backward Propagation**:
   - Implemented in the `backward` function.
   - Computes gradients for weights, biases, and input.

4. **Training Loop**:
   - Processes data in mini-batches.
   - Applies shuffling to improve generalization.

5. **Prediction**:
   - Uses the `predict` function to classify test samples.

6. **Data Loading**:
   - Functions `read_mnist_images` and `read_mnist_labels` handle the MNIST file format.

7. **Performance**:
   - Shuffles data using `shuffle_data` to reduce overfitting.
   - Tracks training time and accuracy per epoch.

### Hyperparameters
The following parameters can be adjusted in the code:
- **Input Size**: `784` (fixed for MNIST images)
- **Hidden Size**: `256`
- **Output Size**: `10`
- **Learning Rate**: `0.0005`
- **Momentum**: `0.9`
- **Epochs**: `20`
- **Batch Size**: `64`
- **Mini Batch Size**: `5`

## MNIST Dataset
The MNIST dataset is a collection of 60,000 training images and 10,000 test images of handwritten digits (0-9). Each image is 28x28 pixels.

### File Format
- **Images**: Stored in `train-images.idx3-ubyte`
- **Labels**: Stored in `train-labels.idx1-ubyte`

### Preprocessing
- Pixel values are normalized to the range `[0, 1]`.
- Labels are converted to one-hot encoding for training.

## Output
During execution, the program outputs:
- Accuracy on the test set after each epoch
- Time taken for each epoch

Example output:
```plaintext
Epoch 1, Accuracy: 85.67%, Time: 20.45 seconds
Epoch 2, Accuracy: 88.12%, Time: 18.32 seconds
...
Epoch 20, Accuracy: 92.45%, Time: 19.28 seconds
```

## Limitations
- This implementation uses a basic feedforward network and does not include convolutional layers.
- The learning rate and other hyperparameters need manual tuning for optimal results.
- Processing is single-threaded and may not leverage multi-core CPUs.

## Future Enhancements
- Implementing additional activation functions like sigmoid or tanh.
- Adding convolutional layers to improve performance.
- Using a more advanced optimizer like Adam.
- Supporting GPU acceleration with CUDA.

## References
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Softmax Function](https://en.wikipedia.org/wiki/Softmax_function)
- [ReLU Activation](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))

## License
This project is distributed under the MIT License.

---

Feel free to report issues or suggest improvements to the code!

