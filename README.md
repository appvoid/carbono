# carbono

carbono is a micro-library that provides a simple and easy-to-use interface for building and training Feed-Forward neural networks all in just ~500 lines of code.

#### Features

- Minimal: Tanh, Sigmoid, ReLU, SELU and Softmax as activations and Xavier/Glorot initialization which helps speed up training.

- Informative: Generates a comprehensive summary of the training process, including loss, number of parameters, and training time.

- Iterative: Saves trained models as a JSON file and load them back. Useful to resume work or just share it with people.

With its straightforward API and user-friendly design, carbono empowers anyone to build and experiment with neural networks in a flexible and accessible manner.

### Quick Usage
```javascript
const nn = new carbono()
const dataset = [{input: [0], output: [1]}, {input: [1], output: [0]}]
nn.train(dataset)
console.log(nn.predict([0])) // should return a number near to [1]
```
