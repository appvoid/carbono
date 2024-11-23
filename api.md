# API Guide

## Constructor
```javascript
new carbono(debug = true)
```
Creates a new neural network instance.
- `debug`: Boolean to enable/disable console logging (default: true)

## Core Methods

### layer(inputSize, outputSize, activation)
Adds a new layer to the network.
- **Parameters:**
  - `inputSize`: Number of input neurons
  - `outputSize`: Number of output neurons
  - `activation`: Activation function ("tanh", "sigmoid", "relu", "selu", "softmax")
- **Returns:** void

### train(trainSet, options)
Trains the neural network.
- **Parameters:**
  - `trainSet`: Array of objects with format: `{input: [], output: []}`
  - `options`: Object with training parameters:
    ```javascript
    {
      epochs: 200,              // Number of training iterations
      learningRate: 0.212,      // Learning rate
      printEveryEpochs: 100,    // Log interval
      earlyStopThreshold: 1e-6, // Early stopping condition
      testSet: null,            // Validation dataset
      callback: null,           // Function called after each epoch
      optimizer: "sgd",         // Optimizer ("sgd" or "adam")
      lossFunction: "mse"       // Loss function ("mse" or "cross-entropy")
    }
    ```
- **Returns:** Promise with training summary:
    ```javascript
    {
      trainLoss: number,
      testLoss: number,
      parameters: number,
      training: {
        time: number,
        epochs: number,
        learningRate: number
      },
      layers: [{
        inputSize: number,
        outputSize: number,
        activation: string
      }]
    }
    ```

### predict(input, tags)
Makes predictions using the trained network.
- **Parameters:**
  - `input`: Array of input values
  - `tags`: Boolean to return labeled predictions (default: true)
- **Returns:** 
  - If `tags=true` and using classification: Array of `{label: string, probability: number}`
  - Otherwise: Array of raw output values

### save(name)
Saves the model to a .uai file.
- **Parameters:**
  - `name`: Filename (default: "model")
- **Returns:** Promise

### load(callback)
Loads a model from a .uai file.
- **Parameters:**
  - `callback`: Function called after successful load
- **Returns:** Promise

## Supported Features

### Activation Functions
- `tanh`: Hyperbolic tangent
- `sigmoid`: Sigmoid function
- `relu`: Rectified Linear Unit
- `selu`: Scaled Exponential Linear Unit
- `softmax`: Softmax function (typically used in output layer for classification)

### Loss Functions
- `mse`: Mean Squared Error
- `cross-entropy`: Cross Entropy Loss

### Optimizers
- `sgd`: Stochastic Gradient Descent
- `adam`: Adam Optimizer

## Usage Example

```javascript
const nn = new carbono();

// Add layers
nn.layer(2, 4, "tanh");    // Input layer with 2 inputs, 4 hidden neurons
nn.layer(4, 1, "sigmoid"); // Output layer with 1 output

// Training data
const trainSet = [
  {input: [0, 0], output: [0]},
  {input: [0, 1], output: [1]},
  {input: [1, 0], output: [1]},
  {input: [1, 1], output: [0]}
];

// Train the network
await nn.train(trainSet, {
  epochs: 1000,
  learningRate: 0.1
});

// Make predictions
const prediction = nn.predict([0, 1]);
```

## Notes
- The network automatically handles label encoding for classification tasks
- Early stopping is implemented to prevent overfitting
- The network supports both regression and classification tasks
- Model state can be saved and loaded using .uai files
- Supports asynchronous training with progress callbacks
- Includes Xavier weight initialization
- Implements gradient clipping for stability
