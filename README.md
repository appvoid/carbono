# carbono

<img src="https://raw.githubusercontent.com/appvoid/carbono/main/dot.png" width="256px" height="256px"/>


carbono is an open-source JavaScript micro-library that provides a simple and easy-to-use interface for building and training Feed-Forward neural networks all in just ~500 lines of code. Designed with usability in mind, the library is perfect for those who are new to machine learning and want to get started without the steep learning curve.

### Key Features

- Lightweight: This is just JavaScript, so it's perfect to test on raspberry pi / low-powered devices. If it has a browser, you can run it. If not, you can run it on Node.js.

- Activation Functions: The library supports various activation functions such as Tanh, Sigmoid, ReLU, and SELU, allowing for versatility in building neural network models.

- Xavier/Glorot Initialization: It uses Xavier/Glorot initialization for weight parameters to ensure proper scaling, which helps speed up training.

- Easy-to-Use Interface: With a simple and intuitive API, users can easily define the structure of the neural network, add layers, and set training parameters.

- Training and Prediction: It supports training the neural network with a dataset and making predictions with the trained model.

- Batch Training: Supports training in batches for improved performance and stability.

- Early Stopping: Implements early stopping to prevent overfitting.

- Model Evaluation: Supports evaluation on a test set during training.

- Save and Load: Provides functionality to save the trained model as a JSON file and load it back for future use, making it convenient for users to resume work.

- Detailed Training Summary: Generates a comprehensive summary of the training process, including loss, number of parameters, and training time.

The Carbono Neural Network library is a great starting point for those who want to explore machine learning and neural networks without being overwhelmed by complexity. With its straightforward API and user-friendly design, this library empowers users to build and experiment with neural networks in a flexible and accessible manner.

### Quick Usage
```javascript
const nn = new carbono()
const dataset = [{input: [0], output: [1]}, {input: [1], output: [0]}]
nn.train(dataset)
console.log(nn.predict([0])) // should return a number near to [1]
```

## Class: `carbono`

### Constructor

```javascript
const nn = new carbono(debug = true);
```

- `debug` (optional): If set to `true`, the library will log debug information. Default is `true`.

### Method: `layer`

```javascript
nn.layer(inputSize, outputSize, activation = 'tanh');
```

- `inputSize`: Number of input neurons for the layer.
- `outputSize`: Number of output neurons for the layer.
- `activation` (optional): Activation function for the layer. Supported activation functions include `'tanh'`, `'sigmoid'`, `'relu'`, and `'selu'`. Default is `'tanh'`.

### Method: `train`

```javascript
nn.train(trainSet, options);
```

- `trainSet`: An array of objects with `input` and `output` properties representing the training data.
- `options` (optional): An object containing training parameters:
  - `epochs` (default: 200): Number of training epochs.
  - `learningRate` (default: 0.212): Learning rate for gradient descent.
  - `batchSize` (default: 16): Size of each training batch.
  - `printEveryEpochs` (default: 100): Prints training error every specified number of epochs.
  - `earlyStopThreshold` (default: 1e-6): Threshold for early stopping.
  - `testSet` (default: null): Optional test set for evaluation during training.

### Method: `predict`

```javascript
const result = nn.predict(input);
```

- `input`: An array representing the input data for prediction.

### Method: `save`

```javascript
nn.save(name = 'model');
```

- `name` (optional): The name of the saved model file. Default is `'model'`.

### Method: `load`

```javascript
nn.load(callback);
```

- `callback`: A callback function to be executed after the model is loaded.

## Example Usage

```javascript
// Create a neural network
const nn = new carbono();

// Add layers to the neural network
nn.layer(2, 4, 'relu');
nn.layer(4, 1, 'sigmoid');

// Prepare training data
const trainingData = [
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] },
  { input: [1, 0], output: [1] },
  { input: [1, 1], output: [0] }
];

// Train the neural network
nn.train(trainingData, {
  epochs: 1000,
  learningRate: 0.1,
  batchSize: 4,
  printEveryEpochs: 100
});

// Make predictions
console.log(nn.predict([0, 0])); // Expected output close to [0]
console.log(nn.predict([0, 1])); // Expected output close to [1]
console.log(nn.predict([1, 0])); // Expected output close to [1]
console.log(nn.predict([1, 1])); // Expected output close to [0]

// Save the model
nn.save('xor_model');

// Load the model
nn.load(() => {
  console.log('Model loaded successfully!');
  // You can now use the loaded model for predictions
});
```

Note: Ensure that you're running this in an environment where file operations are supported (e.g., a web browser) for the save and load functionality to work properly.
