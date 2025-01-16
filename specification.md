# Unified AI (UAI) Specification v1.1

## Overview
The Unified AI (UAI) specification defines a standard for implementing and managing feed-forward neural networks in web browsers. The reference implementation `carbono` demonstrates this specification in practice.

## Core Definition
`carbono` is a lightweight, UAI-compatible feed-forward neural network implementation that provides a unified interface for training, inference, and model management. The UAI specification aims to establish a common web standard for managing feed-forward neural networks in browsers.

## Key Components

### 1. Neural Network Architecture
```javascript
class carbono {
  constructor(debug = true) {
    this.layers = [];     // Network layers configuration
    this.weights = [];    // Layer weights matrices
    this.biases = [];     // Layer biases vectors
    this.details = {};    // Model metadata and training details
  }
}
```

#### Required Properties
- `layers`: Array of layer configurations, each containing:
  - `inputSize`: Number of input nodes
  - `outputSize`: Number of output nodes
  - `activation`: Activation function identifier
- `weights`: Array of weight matrices for each layer
- `biases`: Array of bias vectors for each layer
- `details`: Object containing model metadata and training information

### 2. Core Operations

#### Layer Management
- Dynamic layer addition via `layer(inputSize, outputSize, activation)` method
- Automatic layer size validation between consecutive layers
- Xavier weight initialization for optimal training
- Support for various activation functions

#### Training Pipeline
- Forward propagation with cached layer inputs and outputs
- Backward propagation with comprehensive error computation
- Support for multiple optimization methods
- Early stopping capability
- Optional test set evaluation
- Training progress callbacks

#### Optimization Methods
- Stochastic Gradient Descent (SGD)
- Adam optimizer with momentum and adaptive learning rates
  - β₁ = 0.9
  - β₂ = 0.999
  - ε = 1e-8

### 3. Supported Features

#### Activation Functions
All activation functions must implement both forward pass and derivative:
- `tanh`: Hyperbolic tangent
- `sigmoid`: Logistic function
- `relu`: Rectified Linear Unit
- `selu`: Scaled Exponential Linear Unit
- `softmax`: Softmax function (for classification)

#### Loss Functions
Each loss function must provide both loss calculation and derivative:
- `mse`: Mean Squared Error
- `cross-entropy`: Cross-Entropy Loss

#### Model Persistence

##### Save Format Options
1. Binary Format (.uai)
- Header structure (Uint32Array):
  - Metadata length
  - Metadata padding
  - Weight buffer length
  - Bias buffer length
- Metadata section (JSON)
- Weight buffer (Float32Array)
- Bias buffer (Float32Array)

2. JSON Format (.json)
- Complete model state in JSON format
- Includes weights, biases, layer configurations, and metadata

##### Model Metadata
```javascript
{
  parameters: number,           // Total number of parameters
  training: {
    loss: number,              // Final training loss
    testLoss: number,          // Final test loss (if applicable)
    time: number,              // Training duration in ms
    epochs: number,            // Number of epochs trained
    learningRate: number       // Learning rate used
  },
  info: {                      // Optional model information
    name: string,
    author: string,
    license: string,
    note: string,
    date: string
  }
}
```

## Implementation Requirements

### Browser Compatibility
- Zero external dependencies
- Pure JavaScript implementation
- Web-first architecture
- Asynchronous operations support
- File system interaction via browser APIs

### Memory Management
- Efficient matrix operations
- Automatic cleanup of optimization states
- Browser-friendly resource usage
- Proper TypedArray usage for binary operations

### Error Handling
- Layer size validation
- Numerical stability checks
- Graceful fallbacks
- Comprehensive error messages

### Binary Format (.uai) Specification
The .uai format is a binary file format optimized for neural network storage:

1. Header Section (16 bytes)
```
[0-3]   Uint32: Metadata length
[4-7]   Uint32: Metadata padding
[8-11]  Uint32: Weight buffer length
[12-15] Uint32: Bias buffer length
```

2. Metadata Section
- JSON-encoded string containing:
  ```javascript
  {
    layers: LayerConfig[],
    details: ModelMetadata,
    layerInfo: {
      weightShapes: number[][],
      biasShapes: number[]
    },
    tags?: string[]  // Optional for classification
  }
  ```
- Padded to 4-byte alignment

3. Data Section
- Weight values (Float32Array)
- Bias values (Float32Array)

## Version Information
- Specification Version: 1.1
- Implementation Status: Production Ready
- Last Updated: January 15, 2025
- Reference Implementation: carbono.js

This specification defines the complete requirements for implementing UAI-compatible neural networks, with a focus on browser-based deployment and standardization.
