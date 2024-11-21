Here's an updated API guide that more accurately reflects the `carbono` class implementation:

## Class: `carbono`

### Constructor
```javascript
const nn = new carbono(debug = true);
```
- `debug` (optional): Enables or disables debug messages. Default is `true`.

### Method: `layer(inputSize, outputSize, activation = 'tanh')`
Adds a new layer to the neural network with specified configuration.
- `inputSize`: Number of input neurons
- `outputSize`: Number of output neurons
- `activation` (optional): Activation function for the layer
  - Supported functions: `'tanh'` (default), `'sigmoid'`, `'relu'`, `'selu'`, `'softmax'`

### Method: `train(trainSet, options = {})`
Trains the neural network with the provided dataset.

#### Training Options:
- `epochs` (default: 200): Number of training iterations
- `learningRate` (default: 0.212): Learning rate for weight updates
- `printEveryEpochs` (default: 100): Frequency of logging training progress
- `earlyStopThreshold` (default: 1e-6): Threshold for stopping training
- `testSet` (default: null): Optional validation dataset
- `callback` (default: null): Optional function called after each epoch
- `optimizer` (default: 'sgd'): Optimization method ('sgd' or 'adam')
- `lossFunction` (default: 'mse'): Loss calculation method ('mse' or 'cross-entropy')

#### Training Set Format:
```javascript
const trainSet = [
  { input: [input1, input2, ...], output: [output1, output2, ...] },
  // or for classification
  { input: [input1, input2, ...], output: 'class_label' }
];
```

### Method: `predict(input, tags = true)`
Generates predictions for the given input.
- `input`: Input data array
- `tags` (optional): 
  - `true` (default): Returns labeled probabilities for classification
  - `false`: Returns raw prediction values

### Method: `save(name = 'model')`
Saves the trained model to a JSON file.
- `name` (optional): Filename for the saved model

### Method: `load(callback)`
Loads a previously saved model from a JSON file.
- `callback`: Function to execute after model is loaded

### Key Features:
- Supports multiple activation functions
- Flexible layer configuration
- Supports both regression and classification
- Built-in optimizers (SGD and Adam)
- Multiple loss functions
- Model save and load capabilities

### Example Usage:
```javascript
// Create a neural network
const nn = new carbono();

// Define network architecture
nn.layer(2, 4, 'tanh');   // Input layer: 2 neurons, hidden layer: 4 neurons
nn.layer(4, 1, 'sigmoid'); // Output layer: 1 neuron

// Train the network
await nn.train(trainingData, {
  epochs: 500,
  learningRate: 0.1,
  optimizer: 'adam'
});

// Make predictions
const prediction = nn.predict([0.5, 0.3]);
```
