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

- `trainSet`: An array of objects with `input` and `output` properties representing the training data. `output` accepts a number and a sequence of numbers (or a label when using softmax).
- `options` (optional): An object containing training parameters:
  - `epochs` (default: 200): Number of training epochs.
  - `learningRate` (default: 0.212): Learning rate for gradient descent.
  - `batchSize` (default: 16): Size of each training batch.
  - `printEveryEpochs` (default: 100): Prints training error every specified number of epochs.
  - `earlyStopThreshold` (default: 1e-6): Threshold for early stopping.
  - `testSet` (default: null): Optional test set for evaluation during training.

### Method: `predict`

```javascript
const result = nn.predict(input,tags=true);
```

- `input`: An array representing the input data for prediction.
- `tags`: Determines if a model (trained on labels) should format with labels or just return raw predictions.

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
