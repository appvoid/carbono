## Overview

carbono is a lightweight neural network library for JavaScript that supports both classification and regression tasks with multiple activation functions, optimizers, and loss functions.

## Core API

### Creating a Network

```javascript

const nn = new carbono(debug = true);

```

- debug: Enable/disable console logging (default: true)

### Adding Layers

```javascript

nn.layer(inputSize, outputSize, activation)

```

- inputSize: Number of input neurons

- outputSize: Number of output neurons

- activation: Activation function ("tanh", "sigmoid", "relu", "selu", "softmax")

Returns: this

### Training

```javascript

await nn.train(trainSet, options)

```

Parameters:

- trainSet: Array of {input: [], output: []} objects

- options: Training configuration object:

```javascript

{

  epochs: 10,               // Training iterations (default: 10)

  learningRate: 0.212,      // Learning rate (default: 0.212)

  printEveryEpochs: 1,      // Logging frequency (default: 1)

  earlyStopThreshold: 1e-6, // Early stopping condition

  testSet: null,            // Validation data

  callback: null,           // Epoch callback function  

  optimizer: "sgd",         // "sgd" or "adam"

  lossFunction: "mse"       // "mse" or "cross-entropy"

}

```

### Prediction

```javascript

nn.predict(input, tags = true)

```

- input: Input array

- tags: Return labeled predictions (default: true)

### Model Persistence

```javascript

// Save model (supports .json and binary .uai format)

await nn.save(name = "model", useBinary = false)

// Load model (supports .json and binary .uai format) 

await nn.load(callback, useBinary = false)

```

- useBinary: Use binary .uai format instead of JSON (default: false)

### Model Metadata

```javascript

nn.info({

  name: string,     // Model name

  author: string,   // Author name

  license: string,  // License type

  note: string,     // Description

  date: string      // Creation date

})

```

## Supported Functions

### Activation Functions

- tanh: Hyperbolic tangent

- sigmoid: Sigmoid function

- relu: Rectified Linear Unit 

- selu: Scaled Exponential Linear Unit

- softmax: Softmax (for classification)

### Loss Functions

- mse: Mean Squared Error

- cross-entropy: Cross Entropy Loss

### Optimizers

- sgd: Stochastic Gradient Descent

- adam: Adam Optimizer

## Example Usage

```javascript

// Create network

const nn = new carbono();

// Add layers

nn.layer(2, 4, "tanh")

  .layer(4, 1, "sigmoid");

// Training data

const data = [

  {input: [0,0], output: [0]},

  {input: [0,1], output: [1]},

  {input: [1,0], output: [1]}, 

  {input: [1,1], output: [0]}

];

// Train model

await nn.train(data, {

  epochs: 1000,

  learningRate: 0.1

});

// Make prediction

const prediction = nn.predict([0,1]);

// Save model

await nn.save("xor_model");

```

Additional updates from previous version:

- Updated default epochs to 10 and printEveryEpochs to 1

- Clarified return types and default values

- Corrected default learning rate to 0.212
