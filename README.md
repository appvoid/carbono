### carbono

> latest changes
> - ✦ for more information on the changes, please take a look at the [changelog](https://github.com/appvoid/carbono/blob/main/changelog.md)
> - v7: now default exporting mode is json, with optional binary compression for larger neural networks
> - v6: dramatically optimized file size; for instance, 18k parameters model was reduced from 376kb to 73kb
> - v5: new metadata fields added; removed unnecesary "layers" metadata

A micro-library built for edge devices to train feed-forward neural networks in under 1000 lines of code.

#### Quick Usage
```html
<script src="carbono.js"></script>
```

```javascript
new carbono().layer(2,1).train([{ input:[1,1], output:[0] }])
```

#### for the edge
- default json format: Simple and readable
- binary compression (.uai): optimized for larger networks
- 8-bit quantization: further reduces model size with minimal impact

#### inference on edge devices

```bash
cd server
g++ -std=c++17 src/server.cpp -I./include -pthread -o server
./server xor_model.json 8080

# later...

curl -X POST http://localhost:8080/predict \
     -H "Content-Type: application/json" \
     -d '{"input": [1, 1]}'

# {"output":[0.0009481541869894417]}
```

> if you need help, you can reach to me as @apppvoidofficial on X, you can also find me on tinyllama discord server sometimes.
