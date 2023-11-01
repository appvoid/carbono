# carbono
Carbono is an open-source JavaScript library that provides a simple and easy-to-use interface for building and training Feed-Forward neural networks. Designed with usability in mind, the library is perfect for those who are new to machine learning and want to get started without the steep learning curve.

### Key Features

- Activation Functions: The library supports various activation functions such as Tanh, Sigmoid, ReLU, and SELU, allowing for versatility in building neural network models.

- Glorot Initialization: The library uses Glorot initialization for weight parameters to ensure proper scaling, which helps speed up training.

- Easy-to-Use Interface: With a simple and intuitive API, users can easily define the structure of the neural network, add layers, and set training parameters.

- Training and Prediction: The library supports training the neural network with a dataset and making predictions with the trained model.

- Save and Load: The library provides functionality to save the trained model to a file and load it back for future use, making it convenient for users to resume work.

The Carbono Neural Network library is a great starting point for those who want to explore machine learning and neural networks without being overwhelmed by complexity. With its straightforward API and user-friendly design, this library empowers users to build and experiment with neural networks in a flexible and accessible manner.

### Usage
``` javascript
const model = new carbono()

const dataset = [
  {input: [0], output: [1},
  {input: [1], output: [0},
]

model.train(dataset)

console.log(model.predict([0])) // should return a number near to [1]

```
