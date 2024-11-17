# carbono

<img src="https://raw.githubusercontent.com/appvoid/carbono/main/dot.png" width="256px" height="256px"/>

carbono is a micro-library that provides a simple and easy-to-use interface for building and training Feed-Forward neural networks all in just ~500 lines of code.

#### Features

- Activations: Tanh, Sigmoid, ReLU, SELU and Softmax.

- Xavier/Glorot: It uses this initialization for weight parameters to ensure proper scaling, which helps speed up training.

- Save and Load: Provides functionality to save the trained model as a JSON file and load it back for future use, making it convenient for users to resume work or just share it with people.

- Summary: Generates a comprehensive summary of the training process, including loss, number of parameters, and training time.

This tool is a great starting point for those who want to explore machine learning and neural networks without being overwhelmed by complexity. With its straightforward API and user-friendly design, carbono empowers anyone to build and experiment with neural networks in a flexible and accessible manner.

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

## Basic example

```javascript
// Create a neural network and add layers to it
const nn = new carbono();
nn.layer(2, 4, 'relu');
nn.layer(4, 1, 'sigmoid');

// Prepare training data
const dataset = [
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] },
  { input: [1, 0], output: [1] },
  { input: [1, 1], output: [0] }
];

// Train the neural network
nn.train(dataset, { epochs: 100, learningRate: 0.1, printEveryEpochs: 10 });

// Make predictions
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

## Classification example

```javascript
// Create a new neural network instance
const nn = new carbono();
nn.layer(4, 6, 'tanh');
nn.layer(6, 4, 'softmax'); 

// Prepare training data
const trainData = [
    {
        input: [0.8, 0.2, 0.2, 0.1],
        output: 'cat'
    },
    {
        input: [0.9, 0.3, 0.4, 0.2],
        output: 'dog'
    },
    {
        input: [1.0, 0.5, 0.6, 0.3],
        output: 'wolf'
    },
    {
        input: [0.4, 0.2, 0.2, 0.1],
        output: 'bird'
    }
];

// Train the network
nn.train(trainData, {
    epochs: 50,
    learningRate: 0.1,
}).then(()=>{
  const testInput = [0.9, 0.3, 0.4, 0.3];
  const prediction = nn.predict(testInput);
  console.log(prediction); // Will return a nicely formatted objects array like: [{'label':'class1','probability': 0.91283},...] 
})
```

## Advanced example

```javascript
function emojiToBinary(emoji) {
  return Array.from(emoji)
    .map(char => char.codePointAt(0).toString(2).padStart(16, '0'))
    .join('');
}

const emojis = [
  '😀', '😊', '😂', '😅', '🤣', '😇', '😉', '😍', '😘', '😜', 
  '😎', '🤩', '🥳', '🤔', '😑', '😒', '🙄', '😏', '😓', '😭', 
  '😡', '🤬', '🥺', '😱', '😴', '😷', '🤒', '🤢', '🤮', '😵', 
  '🤯', '🤠', '🤑', '😈', '👿', '🤡', '👻', '💀',
];

const labels = [
  'smile', 'joy', 'laugh', 'nervous', 'rofl', 'angel', 'wink', 
  'love', 'kiss', 'playful', 'cool', 'starstruck', 'celebrate', 
  'thinking', 'blank', 'annoyed', 'eyeroll', 'smirk', 'sweat', 
  'cry', 'angry', 'rage', 'pleading', 'shock', 'sleepy', 'mask', 
  'sick', 'nauseous', 'vomit', 'dizzy', 'exploding_head', 'cowboy', 
  'lust', 'devil', 'evil', 'clown', 'ghost', 'skull'
];


const emojiClasses = emojis.map((emoji, index) => ({
  emoji,
  binary: emojiToBinary(emoji),
  label: labels[index]
}));

const trainSet = emojiClasses.map(item => ({
  input: item.binary.split('').map(bit => parseInt(bit)),
  output: Array(labels.length).fill(0).map((_, i) => labels[i] === item.label ? 1 : 0)
}));

const nn = new carbono(true);

// Input layer
const binaryLength = emojiToBinary(emojis[0]).length;
nn.layer(binaryLength, 10, "relu"); // 16 bits for each emoji, 10 neurons in the hidden layer

// Output layer with softmax activation
nn.layer(10, labels.length, "softmax"); // output classes

nn.train(trainSet, {
  epochs: 100,
  learningRate: 0.1,
  batchSize: 8,
  printEveryEpochs: 25,
  earlyStopThreshold: 1e-5,
}).then((summary) => {

  const newEmoji = '😎';
  const newInput = emojiToBinary(newEmoji).split('').map(bit => parseInt(bit));
  const prediction = nn.predict(newInput);

  console.log("Prediction:", prediction);

  // Convert softmax output to class label
  const predictedClass = prediction.indexOf(Math.max(...prediction));
  const predictedLabel = labels[predictedClass];
  console.log("Predicted Label:", predictedLabel);
  console.log("Predicted Emoji:", emojis[labels.indexOf(predictedLabel)]);
});
```
