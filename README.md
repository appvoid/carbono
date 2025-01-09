### carbono

> changelog
> - ✦ for more information on the changes, please take a look at the [changelog](https://github.com/appvoid/carbono/blob/main/changelog.md)
> - v7: now default exporting mode is json, with optional binary compression for larger neural networks
> - v6: dramatically optimized file size; for instance, 18k parameters model was reduced from 376kb to 73kb
> - v5: new metadata fields added; removed unnecesary "layers" metadata
> - v4: optimized save/load handling for new json-based extension name ".uai"
> - v3: adam optmizer, a complete code rewrite for maintainance purposes (official release)
> - v2: softmax, cross-entropy and labels for classifications
> - v1: first release

a micro-library with a simple api to build and train feed-forward neural networks, all in less than 1000 lines of code. Take a look at the [playground](https://huggingface.co/spaces/appvoid/carbono) and try training your first model. Also check some [examples](https://github.com/appvoid/carbono/blob/main/examples.md) to get familiar with it. Its straightforward [api](https://github.com/appvoid/carbono/blob/main/api.md) lets anyone to prototype with neural networks like never before.

> using the pytorch version is strongly recommended when training, especially for larger networks, while the lightweight javascript version is better for simpler training, prototyping and inference for your apps.

#### Quick Usage
```html
<script src="carbono.js"></script>
```

```javascript
new carbono().layer(2,1).train([{ input:[1,1], output:[0] }])
```

the purpose of this library is to let anyone quickly play with ideas using neural networks in pytorch and get those cool ideas easily shared across the internet using web technologies and a minimal packaging method.

> if you need help, you can reach to me as @apppvoidofficial on X, you can also find me on tinyllama discord server sometimes.
