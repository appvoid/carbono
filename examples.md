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
