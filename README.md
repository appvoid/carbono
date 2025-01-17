### carbono

a micro-library for feedforward neural networks that works everywhere:

> latest changes
> - ✦ for more information on the changes, please take a look at the [changelog](https://github.com/appvoid/carbono/blob/main/changelog.md)
> - v7: now default exporting mode is json, with optional binary compression for larger neural networks (warning: breaking changes due to quantization, only js version)
> - v6: dramatically optimized file size; for instance, 18k parameters model was reduced from 376kb to 73kb
> - v5: new metadata fields added; removed unnecesary "layers" metadata

#### Browser & Edge Ready
- Tiny models (2kb quantized)
- Fast browser inference
- Edge-optimized C++ server
- Simple JavaScript API

#### Build Once, Run Anywhere
```javascript
// Train in browser
new carbono().layer(2,1).train([{ input:[1,1], output:[0] }])

// Or use PyTorch for larger models
// Then run anywhere - browser, edge, server
```

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
