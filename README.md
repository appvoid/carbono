# carbono

carbono is a micro-library that provides a simple and easy-to-use interface for building and training Feed-Forward neural networks all in just ~500 lines of code. If you don't want to code, you can also take a look at the [playground](https://huggingface.co/spaces/appvoid/carbono), try training your first model there. You can also check the [examples](https://github.com/appvoid/carbono/blob/main/examples.md) to get some boilerplate.

#### Features

- Minimal: Tanh, Sigmoid, ReLU, SELU and Softmax as activations and Xavier/Glorot initialization which helps speed up training.

- Informative: Generates a comprehensive summary of the training process, including loss, number of parameters, and training time.

- Iterative: Saves trained models as a JSON file and load them back. Useful to resume work or just share it with people.

### Quick Usage
```html
<script src="carbono.js"></script>
```

```javascript
const nn = new carbono()
const dataset = [{input: [0], output: [1]}, {input: [1], output: [0]}]
nn.train(dataset)
console.log(nn.predict([0])) // should return a number near to [1]
```

With its straightforward [API](https://github.com/appvoid/carbono/blob/main/api.md) and user-friendly design, carbono empowers anyone to build and experiment with neural networks in a flexible and accessible manner.
