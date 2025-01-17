### carbono

> latest changes
> - ✦ for more information on the changes, please take a look at the [changelog](https://github.com/appvoid/carbono/blob/main/changelog.md)
> - v7: now default exporting mode is json, with optional binary compression for larger neural networks (warning: breaking changes for quantization)
> - v6: dramatically optimized file size; for instance, 18k parameters model was reduced from 376kb to 73kb
> - v5: new metadata fields added; removed unnecesary "layers" metadata

a micro-library for neural networks that works everywhere:

#### environments
- browser: fast inference, simple training
- python/pytorch: gpu-accelerated training
- c++ server: optimized edge inference
- python: cpu training without pyTorch dependencies

#### optimization
- compressed models (32kb → 2kb)
- 8-bit quantization for edge devices
- binary format for larger networks

#### quick usage
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
