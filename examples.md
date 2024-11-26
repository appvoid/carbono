### Introduction

carbono is built in such a way that it abstracts away managing feed-forward networks, in fact, you might say it's a web specification for machine learning given how easy it is to make it work with any [technology](https://github.com/appvoid/carbono/blob/main/carbono.ipynb).

These are some basic examples you can try. Keep in mind that some might not work due to the fast-paced working manner i'm approaching on this project, I will try to update this as much as possible though.

## Simple example

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
    epochs: 200,
    learningRate: 0.1,
}).then(()=>{
  const testInput = [0.9, 0.3, 0.4, 0.3]; // Let's try with the "dog" inputs
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

## Evaluation for best model
```javascript
// Softmax/Categorical Cross-Entropy Friendly Dataset (Iris Dataset)
const irisTrainSet = [
  { input: [.1, .5, .4, 0.2], output: [1, 0, 0] }, // Setosa
  { input: [.9, .0, .4, 0.2], output: [1, 0, 0] }, // Setosa
  { input: [.2, .4, .4, .3], output: [0, 0, 1] }, // Virginica
  { input: [.9, .0, .1, .8], output: [0, 0, 1] }, // Virginica
  { input: [.4, .9, .7, 0.4], output: [1, 0, 0] }, // Setosa
  { input: [.0, .2, .7, .4], output: [0, 1, 0] }, // Versicolor
  { input: [.4, .2, .5, .5], output: [0, 1, 0] }, // Versicolor
  { input: [.3, .3, .0, .5], output: [0, 0, 1] }, // Virginica
  { input: [.8, .7, .1, .9], output: [0, 0, 1] }, // Virginica
  { input: [.7, .8, .1, .3], output: [0, 1, 0] }, // Versicolor
];

const irisTestSet = [
  { input: [.1, .5, .4, 0.2], output: [1, 0, 0] }, // Setosa
  { input: [.2, .4, .4, .3], output: [0, 0, 1] }, // Virginica
  { input: [.4, .9, .7, 0.4], output: [1, 0, 0] }, // Setosa
  { input: [.0, .2, .7, .4], output: [0, 1, 0] }, // Versicolor
];

// Simpler Dataset with More Inputs (Regression Task)
const simpleTrainSet = [
  { input: [0.1, 0.2, 0.3, 0.4, 0.5], output: [0.1] },
  { input: [0.2, 0.3, 0.4, 0.5, 0.6], output: [0.2] },
  { input: [0.3, 0.4, 0.5, 0.6, 0.7], output: [0.3] },
  { input: [0.4, 0.5, 0.6, 0.7, 0.8], output: [0.4] },
  { input: [0.5, 0.6, 0.7, 0.8, 0.9], output: [0.5] },
  { input: [0.6, 0.7, 0.8, 0.9, 1.0], output: [0.6] },
  { input: [0.7, 0.8, 0.9, 1.0, 0.1], output: [0.7] },
  { input: [0.8, 0.9, 1.0, 0.1, 0.2], output: [0.8] },
  { input: [0.9, 1.0, 0.1, 0.2, 0.3], output: [0.9] },
  { input: [1.0, 0.1, 0.2, 0.3, 0.4], output: [1.0] },
];

const simpleTestSet = [
  { input: [0.1, 0.2, 0.3, 0.4, 0.5], output: [0.1] },
  { input: [0.2, 0.3, 0.4, 0.5, 0.6], output: [0.2] },
  { input: [0.3, 0.4, 0.5, 0.6, 0.7], output: [0.3] },
  { input: [0.4, 0.5, 0.6, 0.7, 0.8], output: [0.4] },
];

// Function to train a model with given parameters
async function trainModel(trainSet, testSet, optimizer, lossFunction, activation) {
  const model = new carbono(true);

  // Define layers based on activation function and loss function compatibility
  if (lossFunction === "cross-entropy") {
    if (activation !== "softmax" && activation !== "sigmoid") {
      console.warn(
        `⚠️ For cross-entropy loss, it's recommended to use 'softmax' (multi-class) or 'sigmoid' (binary) activation. Using 'softmax' by default.`
      );
      activation = "softmax";
    }
    if (trainSet[0].output.length > 1) {
      model.layer(trainSet[0].input.length, Math.ceil((trainSet[0].input.length + trainSet[0].output.length) / 2), "tanh");
      model.layer(Math.ceil((trainSet[0].input.length + trainSet[0].output.length) / 2), trainSet[0].output.length, activation);
    } else {
      // Binary classification or regression with sigmoid
      model.layer(trainSet[0].input.length, Math.ceil((trainSet[0].input.length + trainSet[0].output.length) / 2), "tanh");
      model.layer(Math.ceil((trainSet[0].input.length + trainSet[0].output.length) / 2), trainSet[0].output.length, activation);
    }
  } else {
    // For MSE or other loss functions, use appropriate activation
    model.layer(trainSet[0].input.length, 3, "tanh");
    model.layer(3, trainSet[0].output.length, "tanh");
  }

  // Train the model
  const options = {
    epochs: 100,
    learningRate: 0.1,
    printEveryEpochs: 100,
    earlyStopThreshold: lossFunction === "cross-entropy" ? 1e-4 : 1e-6,
    testSet,
    optimizer,
    lossFunction,
  };

  // Log the summary
  console.log(`\n🔍 Training Model with Optimizer: ${optimizer}, Loss Function: ${lossFunction}, Activation: ${activation}`);
  const summary = await model.train(trainSet, options);

}

// Train models with different combinations
async function trainAllModels() {
  const optimizers = ["sgd", "adam"];
  const lossFunctions = ["mse", "cross-entropy"];
  const activations = ["tanh", "softmax"];

  // Train models on Iris dataset
  console.log("📚 Training models on Iris dataset:");
  for (const optimizer of optimizers) {
    for (const lossFunction of lossFunctions) {
      for (let activation of activations) {
        // Skip incompatible activation and loss function combinations
        if (lossFunction === "cross-entropy" && activation === "tanh") {
          console.warn(
            `⚠️ Skipping combination: Optimizer=${optimizer}, Loss Function=${lossFunction}, Activation=${activation} (Incompatible)`
          );
          continue;
        }
        await trainModel(irisTrainSet, irisTestSet, optimizer, lossFunction, activation);
      }
    }
  }

  // Train models on simpler dataset
  console.log("\n📚 Training models on simpler dataset:");
  for (const optimizer of optimizers) {
    for (const lossFunction of lossFunctions) {
      for (let activation of activations) {
        // Skip cross-entropy loss for regression tasks
        if (lossFunction === "cross-entropy") {
          console.warn(
            `⚠️ Skipping combination: Optimizer=${optimizer}, Loss Function=${lossFunction}, Activation=${activation} (Not suitable for regression)`
          );
          continue;
        }
        await trainModel(simpleTrainSet, simpleTestSet, optimizer, lossFunction, activation);
      }
    }
  }
}

// Run the training
trainAllModels();
```

## Dummy image recognition

```javascript
// First, let's create a function to load and process images
function loadAndProcessImage(url) {
    // Create a canvas to process the image
    const img = new Image();
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    return new Promise((resolve, reject) => {
        img.crossOrigin = "Anonymous";  // Handle CORS issues
        
        img.onload = () => {
            // Resize image to a standard size (e.g., 64x64)
            canvas.width = 64;
            canvas.height = 64;
            
            // Draw and resize image
            ctx.drawImage(img, 0, 0, 64, 64);
            
            // Get image data and normalize it
            const imageData = ctx.getImageData(0, 0, 64, 64).data;
            
            // Convert to 128-length array
            const processed = new Array(128).fill(0);
            
            // Simple processing: take average of RGB values for each pixel
            for (let i = 0; i < imageData.length; i += 4) {
                const pos = Math.floor(i / 4) % 128;
                const avg = (imageData[i] + imageData[i + 1] + imageData[i + 2]) / 765; // Normalize to 0-1
                processed[pos] = avg;
            }
            
            resolve(processed);
        };
        
        img.onerror = reject;
        img.src = url;
    });
}

// Neural network setup using carbono
const nn = new carbono();
nn.layer(128, 64, 'tanh');  // Input layer
nn.layer(64, 32, 'tanh');   // Hidden layer
nn.layer(32, 2, 'softmax'); // Output layer for 2 categories (cat/dog)

// Array of image URLs
const imageUrls = [
    'https://cdn.pixabay.com/photo/2015/11/17/13/13/bulldog-1047518_1280.jpg',
    'https://cdn.pixabay.com/photo/2018/03/31/06/31/dog-3277416_1280.jpg',
    'https://cdn.pixabay.com/photo/2024/02/17/00/18/cat-8578562_1280.jpg',
    'https://cdn.pixabay.com/photo/2024/01/29/20/40/cat-8540772_1280.jpg'
];

// Labels corresponding to the images (one-hot encoded)
const labels = [
    'dog', // dog
    'dog', // dog
    'cat', // cat
    'cat'  // cat
];

// Process all images and prepare training data
async function prepareTrainingData() {
    const trainData = [];
    
    for (let i = 0; i < imageUrls.length; i++) {
        try {
            const processed = await loadAndProcessImage(imageUrls[i]);
            trainData.push({
                input: processed,
                output: labels[i]
            });
        } catch (error) {
            console.error(`Error processing image ${imageUrls[i]}:`, error);
        }
    }
    
    return trainData;
}

// Train the network with the processed images
async function trainNetwork() {
    try {
        const trainData = await prepareTrainingData();
        
        await nn.train(trainData, {
            epochs: 100,
            learningRate: 0.01,
        });
        
        console.log('Training completed');
        console.log(nn.details);
        
        // Test with a new image
        const testImageUrl = 'https://cdn.pixabay.com/photo/2024/02/17/00/18/cat-8578562_1280.jpg';
        const testProcessed = await loadAndProcessImage(testImageUrl);
        const prediction = nn.predict(testProcessed);
        console.log('Prediction:', prediction);
        
    } catch (error) {
        console.error('Training error:', error);
    }
}

// Function to make a single prediction
async function predictImage(imageUrl) {
    try {
        const processed = await loadAndProcessImage(imageUrl);
        const prediction = nn.predict(processed);
        return prediction;
    } catch (error) {
        console.error('Prediction error:', error);
        return null;
    }
}

// Run the training
trainNetwork();
```
