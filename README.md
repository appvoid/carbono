# carbono

> changelog
> - v4: optimized save/load handling for new json-based extension name ".uai"
> - v3: adam optmizer, a complete code rewrite for maintainance purposes (official release)
> - v2: softmax, cross-entropy and labels for classifications
> - v1: first release

carbono is a micro-library with a simple api to build and train feed-forward neural networks within 500 lines of code. Take a look at the [playground](https://huggingface.co/spaces/appvoid/carbono) and try training your first model. Also check some [examples](https://github.com/appvoid/carbono/blob/main/examples.md) to get familiar with this library.

#### Features

- Minimal: Tanh, Sigmoid, ReLU, SELU and Softmax as activations and Xavier/Glorot initialization which helps speed up training.

- Complete: Use mse or cross-entropy, change learning rate or use adam to make it converge faster!

- Informative: Generates a comprehensive summary of the training process, including loss, number of parameters, and training time.

- Iterative: Saves trained models as a JSON file and load them back. Useful to resume work or just share it with people.

### Quick Usage
```html
<script src="carbono.js"></script>
```

```javascript
const nn = new carbono()
const dataset = [{input: [0], output: [1]}, {input: [1], output: [0]}]
nn.train(dataset).then(()=>{
  console.log(nn.predict([0])) // should return a number near to [1]  
})
```

With its straightforward [API](https://github.com/appvoid/carbono/blob/main/api.md) and user-friendly design, carbono empowers anyone to build and experiment with neural networks in a flexible and accessible manner.
