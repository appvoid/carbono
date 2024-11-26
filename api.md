# Overview
carbono is a lightweight neural network library for JavaScript that supports both classification and regression tasks with multiple activation functions, optimizers, and loss functions.

## New Instance
```javascript
// Create a new neural network instance
const nn = new carbono(debug = true);
```
- `debug`: Enable/disable console logging (default: true)

**Returns:** `this`
## Core Methods

### Adding Layers
```javascript
nn.layer(inputSize, outputSize, activation)
```
- `inputSize`: Number of input neurons
- `outputSize`: Number of output neurons  
- `activation`: Activation function ("tanh", "sigmoid", "relu", "selu", "softmax")

**Returns:** `this`
### Training
```javascript
await nn.train(trainSet, options)
```
**Parameters:**
- `trainSet`: Array of `{input: [], output: []}` objects
- `options`: Training configuration object:
```javascript
{
  epochs: 200,              // Training iterations
  learningRate: 0.212,      // Learning rate
  printEveryEpochs: 100,    // Logging frequency
  earlyStopThreshold: 1e-6, // Early stopping condition
  testSet: null,            // Validation data
  callback: null,           // Epoch callback function
  optimizer: "sgd",         // "sgd" or "adam"
  lossFunction: "mse"       // "mse" or "cross-entropy"
}
```

**Returns:** Training summary object:
```javascript
{
  parameters: number,      // Total model parameters
  training: {
    loss: number,         // Final training loss
    testloss: number,     // Final test loss 
    time: number,         // Training duration (ms)
    epochs: number,       // Total epochs
    learningRate: number  // Learning rate used
  }
}
```

### Prediction
```javascript
nn.predict(input, tags = true)
```
- `input`: Input array
- `tags`: Return labeled predictions (default: true)

Returns:
- Classification with `tags=true`: Array of `{tag: string, probability: number}`
- Otherwise: Raw output array

### Model Persistence
```javascript
// Save model
await nn.save(name = "model") 

// Load model
await nn.load(callback)
```

### Model Metadata
```javascript
nn.info({
  name: "My Model",
  author: "John Doe",
  license: "MIT",
  note: "Description",
  date: "2023-01-01"
})
```

## Features

### Activation Functions
- `tanh`: Hyperbolic tangent
- `sigmoid`: Sigmoid function  
- `relu`: Rectified Linear Unit
- `selu`: Scaled Exponential Linear Unit
- `softmax`: Softmax (for classification)

### Loss Functions
- `mse`: Mean Squared Error
- `cross-entropy`: Cross Entropy Loss

### Optimizers
- `sgd`: Stochastic Gradient Descent
- `adam`: Adam Optimizer

## Example Usage

```javascript
// Create network
const nn = new carbono();

// Add layers
nn.layer(2, 4, "tanh");
nn.layer(4, 1, "sigmoid");

// Training data
const data = [
  {input: [0,0], output: [0]},
  {input: [0,1], output: [1]},
  {input: [1,0], output: [1]},
  {input: [1,1], output: [0]}
];

// Train
await nn.train(data, {
  epochs: 1000,
  learningRate: 0.1
});

// Predict
const prediction = nn.predict([0,1]);

// Save model
await nn.save("xor_model");
```

## Key Features
- Automatic label encoding for classification
- Early stopping to prevent overfitting
- Xavier weight initialization
- Gradient clipping
- Model saving/loading (.uai format)
- Asynchronous training with callbacks
- Comprehensive training metrics
- Support for both regression and classification

This documentation reflects the actual implementation while maintaining clarity and usability for developers.
