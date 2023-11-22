# carbono
Carbono is an open-source JavaScript library that provides a simple and easy-to-use interface for building and training Feed-Forward neural networks. Designed with usability in mind, the library is perfect for those who are new to machine learning and want to get started without the steep learning curve.

### Key Features

- Activation Functions: The library supports various activation functions such as Tanh, Sigmoid, ReLU, and SELU, allowing for versatility in building neural network models.

- Glorot Initialization: The library uses Glorot initialization for weight parameters to ensure proper scaling, which helps speed up training.

- Easy-to-Use Interface: With a simple and intuitive API, users can easily define the structure of the neural network, add layers, and set training parameters.

- Training and Prediction: The library supports training the neural network with a dataset and making predictions with the trained model.

- Save and Load: The library provides functionality to save the trained model to a file and load it back for future use, making it convenient for users to resume work.

The Carbono Neural Network library is a great starting point for those who want to explore machine learning and neural networks without being overwhelmed by complexity. With its straightforward API and user-friendly design, this library empowers users to build and experiment with neural networks in a flexible and accessible manner.

### Quick Usage
``` javascript
const model = new carbono()

const dataset = [
  {input: [0], output: [1]},
  {input: [1], output: [0]},
]

model.train(dataset)

console.log(model.predict([0])) // should return a number near to [1]

```

<h2>Class: <code>carbono</code></h2>

<h3>Constructor</h3>

<code>const nn = new carbono(activation = 'tanh', debug = true);</code>

  <ul>
    <li><code>activation</code> (optional): Specifies the activation function to be used in the neural network. Default is <code>'tanh'</code>. Supported activation functions include <code>'tanh'</code>, <code>'sigmoid'</code>, <code>'relu'</code>, and <code>'selu'</code>.</li>
    <li><code>debug</code> (optional): If set to <code>true</code>, the library will log debug information. Default is <code>true</code>.</li>
  </ul>

  <h3>Method: <code>layer</code></h3>

<code>nn.layer(inputSize, outputSize);</code>

  <ul>
    <li><code>inputSize</code>: Number of input neurons for the layer.</li>
    <li><code>outputSize</code>: Number of output neurons for the layer.</li>
  </ul>

  <h3>Method: <code>train</code></h3>

<code>nn.train(dataset, epochs = 200, learningRate = 0.212, batchSize = 16, printEveryEpochs = 100);</code>

  <ul>
    <li><code>dataset</code>: An array of objects with <code>input</code> and <code>output</code> properties representing the training data.</li>
    <li><code>epochs</code> (optional): Number of training epochs. Default is <code>200</code>.</li>
    <li><code>learningRate</code> (optional): Learning rate for gradient descent. Default is <code>0.212</code>.</li>
    <li><code>batchSize</code> (optional): Size of each training batch. Default is <code>16</code>.</li>
    <li><code>printEveryEpochs</code> (optional): Prints training error every specified number of epochs. Default is <code>100</code>.</li>
  </ul>

  <h3>Method: <code>predict</code></h3>

<code>const result = nn.predict(input);</code>

  <ul>
    <li><code>input</code>: An array representing the input data for prediction.</li>
  </ul>

  <h3>Method: <code>save</code></h3>

<code>nn.save(name = 'model');</code>

  <ul>
  <li><code>name</code> (optional): The name of the saved model file. Default is <code>'model'</code>.</li>
  </ul>

  <h3>Method: <code>load</code></h3>

<code>nn.load(callback);</code>

  <ul>
    <li><code>callback</code>: A callback function to be executed after the model is loaded.</li>
  </ul>

  <h2>Example Usage</h2>

``` javascript
// Create a neural network
const nn = new carbono();

// Add layers to the neural network
nn.layer(2, 4);
nn.layer(4, 1);

// Train the neural network
nn.train(trainingData, 500, 0.1, 32);

// Make predictions
const input = [0.5, 0.3];
const output = nn.predict(input);

// Save and load the model
nn.save('my_model');
nn.load(() => console.log('Model loaded successfully.'));
```

  <p>Note: Ensure that the <code>trainingData</code> variable is properly formatted with <code>input</code> and <code>output</code> properties.</p>

