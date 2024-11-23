### carbono

> changelog
> - v4: optimized save/load handling for new json-based extension name ".uai"
> - v3: adam optmizer, a complete code rewrite for maintainance purposes (official release)
> - v2: softmax, cross-entropy and labels for classifications
> - v1: first release

a micro-library with a simple api to build and train feed-forward neural networks, all within 500 lines of code. Take a look at the [playground](https://huggingface.co/spaces/appvoid/carbono) and try training your first model. Also check some [examples](https://github.com/appvoid/carbono/blob/main/examples.md) to get familiar with it. Its straightforward [api](https://github.com/appvoid/carbono/blob/main/api.md) lets anyone to prototype with neural networks like never before.

#### Quick Usage
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

> if you need help, you can reach to me as @apppvoidofficial on X, you can also find me on tinyllama discord server sometimes.
