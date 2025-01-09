/* 
 *  this is a complete example of a standard feed-forward neural network and ideally, in the future, this self-contained model is planned to be 
 *  sufficient to handle general use-case scenarios as well as becoming a supportive, general-purpose engine for training as well as 
 *  inference on most popular frameworks' models out there - having said that, be warned, as most of my indie open source projects, 
 *  this is not by any means stable yet, even though i'm expecting the tool to be improved over time, you must use it at your own risk.
 */
class carbono {
  constructor(debug = true) {
    this.layers = [];
    this.weights = [];
    this.biases = [];
    this.details = {};
  }

  // Utility Methods
  #xavier(inputSize, outputSize) {
    return (Math.random() - 0.5) * 2 * Math.sqrt(6 / (inputSize + outputSize));
  }

  #clip(value, min = 1e-15, max = 1 - 1e-15) {
    return Math.max(Math.min(value, max), min);
  }

  #matrixMultiply(a, b) {
    return a.map(row =>
      b[0].map((_, i) =>
        row.reduce((sum, val, j) => sum + val * b[j][i], 0)
      )
    );
  }

  // Activation Functions
  #activationFunctions = {
    tanh: {
      fn: x => Math.tanh(x),
      derivative: x => 1 - Math.pow(Math.tanh(x), 2)
    },
    sigmoid: {
      fn: x => 1 / (1 + Math.exp(-x)),
      derivative: x => {
        const sig = 1 / (1 + Math.exp(-x));
        return sig * (1 - sig);
      }
    },
    relu: {
      fn: x => Math.max(0, x),
      derivative: x => x > 0 ? 1 : 0
    },
    selu: {
      fn: x => {
        const alpha = 1.67326;
        const scale = 1.0507;
        return x > 0 ? scale * x : scale * alpha * (Math.exp(x) - 1);
      },
      derivative: x => {
        const alpha = 1.67326;
        const scale = 1.0507;
        return x > 0 ? scale : scale * alpha * Math.exp(x);
      }
    },
    softmax: {
      fn: x => {
        const expValues = Array.isArray(x) ? x.map(val => Math.exp(val)) : [Math.exp(x)];
        const sumExp = expValues.reduce((a, b) => a + b, 0);
        return expValues.map(exp => exp / sumExp);
      },
      derivative: null
    }
  }

  // Loss Functions
  #lossFunctions = {
    mse: {
      loss: (predicted, actual) =>
        predicted.reduce((sum, pred, i) => sum + Math.pow(pred - actual[i], 2), 0),
      derivative: (predicted, actual, activation) =>
        predicted.map((pred, i) => (pred - actual[i]) *
          (activation === 'softmax' ? 1 : this.#getActivationDerivative(pred, activation)))
    },
    'cross-entropy': {
      loss: (predicted, actual) =>
        -actual.reduce((sum, target, i) =>
          sum + target * Math.log(this.#clip(predicted[i])), 0),
      derivative: (predicted, actual) =>
        predicted.map((pred, i) => pred - actual[i])
    }
  }

  #getActivation(x, activation) {
    return this.#activationFunctions[activation].fn(x);
  }

  #getActivationDerivative(x, activation) {
    return this.#activationFunctions[activation].derivative?.(x) ?? null;
  }

  // Layer Management
  layer(inputSize, outputSize, activation = "tanh") {
    if (this.weights.length > 0) {
      const lastLayerOutputSize = this.layers[this.layers.length - 1].outputSize;
      if (inputSize !== lastLayerOutputSize) {
        throw new Error("Layer input size must match previous layer output size.");
      }
    }

    this.layers.push({
      inputSize,
      outputSize,
      activation
    });

    const weights = Array(outputSize)
      .fill()
      .map(() =>
        Array(inputSize)
        .fill()
        .map(() => this.#xavier(inputSize, outputSize))
      );
    this.weights.push(weights);
    this.biases.push(Array(outputSize)
      .fill(0.01));
    return this
  }

  // Forward Propagation
  #forwardPropagate(input) {
    let current = input;
    const layerInputs = [input];
    const layerRawOutputs = [];

    for (let i = 0; i < this.weights.length; i++) {
      const rawOutput = this.weights[i].map((weight, j) =>
        weight.reduce((sum, w, k) => sum + w * current[k], 0) + this.biases[i][j]
      );

      layerRawOutputs.push(rawOutput);
      const layerActivation = this.layers[i].activation;
      current = layerActivation === 'softmax' ?
        this.#getActivation(rawOutput, 'softmax') :
        rawOutput.map(x => this.#getActivation(x, layerActivation));
      layerInputs.push(current);
    }

    return {
      layerInputs,
      layerRawOutputs
    };
  }

  // Backward Propagation
  #backPropagate(layerInputs, layerRawOutputs, target, lossFunction) {
    const outputLayer = this.layers[this.layers.length - 1];
    const outputErrors = this.#lossFunctions[lossFunction].derivative(
      layerInputs[layerInputs.length - 1], target, outputLayer.activation
    );

    const layerErrors = [outputErrors];

    for (let i = this.weights.length - 2; i >= 0; i--) {
      const errors = Array(this.layers[i].outputSize)
        .fill(0);

      for (let j = 0; j < this.layers[i].outputSize; j++) {
        for (let k = 0; k < this.layers[i + 1].outputSize; k++) {
          errors[j] += layerErrors[0][k] * this.weights[i + 1][k][j];
        }
        const activationDeriv = this.#getActivationDerivative(
          layerRawOutputs[i][j], this.layers[i].activation
        );
        if (activationDeriv !== null) {
          errors[j] *= activationDeriv;
        }
      }

      layerErrors.unshift(errors);
    }

    return layerErrors;
  }

  // Optimization Methods
  #initializeOptimizer() {
    if (!this.weight_m) {
      this.weight_m = this.weights.map(layer =>
        layer.map(row => row.map(() => 0))
      );
      this.weight_v = this.weights.map(layer =>
        layer.map(row => row.map(() => 0))
      );
      this.bias_m = this.biases.map(layer => layer.map(() => 0));
      this.bias_v = this.biases.map(layer => layer.map(() => 0));
    }
  }

  #updateWeights(layerIndex, weightGradients, biasGradients, optimizer, params) {
    if (optimizer === 'adam') {
      this.#adamUpdate(layerIndex, weightGradients, biasGradients, params);
    } else {
      this.#sgdUpdate(layerIndex, weightGradients, biasGradients, params.learningRate);
    }
  }

  #adamUpdate(layerIndex, weightGradients, biasGradients, {
    t,
    learningRate
  }) {
    const beta1 = 0.9;
    const beta2 = 0.999;
    const epsilon = 1e-8;

    for (let j = 0; j < this.weights[layerIndex].length; j++) {
      for (let k = 0; k < this.weights[layerIndex][j].length; k++) {
        const g = weightGradients[j][k];
        this.weight_m[layerIndex][j][k] = beta1 * this.weight_m[layerIndex][j][k] + (1 - beta1) * g;
        this.weight_v[layerIndex][j][k] = beta2 * this.weight_v[layerIndex][j][k] + (1 - beta2) * g * g;

        const m_hat = this.weight_m[layerIndex][j][k] / (1 - Math.pow(beta1, t));
        const v_hat = this.weight_v[layerIndex][j][k] / (1 - Math.pow(beta2, t));

        this.weights[layerIndex][j][k] -= (learningRate * m_hat) / (Math.sqrt(v_hat) + epsilon);
      }

      const g_bias = biasGradients[j];
      this.bias_m[layerIndex][j] = beta1 * this.bias_m[layerIndex][j] + (1 - beta1) * g_bias;
      this.bias_v[layerIndex][j] = beta2 * this.bias_v[layerIndex][j] + (1 - beta2) * g_bias * g_bias;

      const m_hat_bias = this.bias_m[layerIndex][j] / (1 - Math.pow(beta1, t));
      const v_hat_bias = this.bias_v[layerIndex][j] / (1 - Math.pow(beta2, t));

      this.biases[layerIndex][j] -= (learningRate * m_hat_bias) / (Math.sqrt(v_hat_bias) + epsilon);
    }
  }

  #sgdUpdate(layerIndex, weightGradients, biasGradients, learningRate) {
    for (let j = 0; j < this.weights[layerIndex].length; j++) {
      for (let k = 0; k < this.weights[layerIndex][j].length; k++) {
        this.weights[layerIndex][j][k] -= learningRate * weightGradients[j][k];
      }
      this.biases[layerIndex][j] -= learningRate * biasGradients[j];
    }
  }

  // Training
  async train(trainSet, options = {}) {
    // Fallback property addition when training a loaded model
    if (!('debug' in this)) {
      this.debug = true; // or any default value you want to set
    }
    const {
      epochs = 10, learningRate = 0.212, printEveryEpochs = 1, earlyStopThreshold = 1e-6, testSet = null, callback = null, optimizer = "sgd", lossFunction = "mse"
    } = options;

    if (typeof trainSet[0].output === "string" ||
      (Array.isArray(trainSet[0].output) && typeof trainSet[0].output[0] === "string")) {
      trainSet = this.#preprocesstags(trainSet);
    }

    const start = Date.now();
    let t = 0;

    if (optimizer === "adam") {
      this.#initializeOptimizer();
    }

    let lastTrainLoss = 0;
    let lastTestLoss = null;

    for (let epoch = 0; epoch < epochs; epoch++) {
      let trainError = 0;

      for (const data of trainSet) {
        t++;
        const {
          layerInputs,
          layerRawOutputs
        } = this.#forwardPropagate(data.input);
        const layerErrors = this.#backPropagate(layerInputs, layerRawOutputs, data.output, lossFunction);

        for (let i = 0; i < this.weights.length; i++) {
          const weightGradients = this.weights[i].map((_, j) =>
            this.weights[i][j].map((_, k) => layerErrors[i][j] * layerInputs[i][k])
          );
          const biasGradients = layerErrors[i];

          this.#updateWeights(i, weightGradients, biasGradients, optimizer, {
            t,
            learningRate
          });
        }

        trainError += this.#lossFunctions[lossFunction].loss(
          layerInputs[layerInputs.length - 1], data.output
        );
      }

      lastTrainLoss = trainError / trainSet.length;

      if (testSet) {
        lastTestLoss = this.#evaluateTestSet(testSet, lossFunction);
      }

      if ((epoch + 1) % printEveryEpochs === 0 && this.debug) {
        console.log(
          `✨ Epoch ${epoch + 1}, Train Loss: ${lastTrainLoss.toFixed(6)}${
            testSet ? `, Test Loss: ${lastTestLoss.toFixed(6)}` : ""
          }`
        );
      }

      if (callback) {
        await callback(epoch + 1, lastTrainLoss, lastTestLoss);
      }

      await new Promise(resolve => setTimeout(resolve, 0));

      if (lastTrainLoss < earlyStopThreshold) {
        if (this.debug) {
          console.log(
            `🚀 Early stopping at epoch ${epoch + 1} with train loss: ${lastTrainLoss.toFixed(6)}${
              testSet ? ` and test loss: ${lastTestLoss.toFixed(6)}` : ""
            }`
          );
        }
        break;
      }
    }

    // Clean up Adam optimizer variables
    if (optimizer === 'adam') {
      delete this.weight_m;
      delete this.weight_v;
      delete this.bias_m;
      delete this.bias_v;
    }

    // Returns metadata
    const summary = this.#generateTrainingSummary(start, Date.now(), {
      epochs,
      learningRate,
      lastTrainLoss,
      lastTestLoss
    });

    this.details = summary;
    return summary;
  }

  #preprocesstags(trainSet) {
    // Initialize tags property only when needed for classification
    const uniquetags = Array.from(
      new Set(
        trainSet
        .map(item => Array.isArray(item.output) ? item.output : [item.output])
        .flat()
      )
    );

    // Set tags property only when preprocessing tags
    this.tags = uniquetags;

    if (this.layers.length === 0) {
      const numInputs = trainSet[0].input.length;
      const numClasses = uniquetags.length;
      this.layer(numInputs, Math.ceil((numInputs + numClasses) / 2), "tanh");
      this.layer(Math.ceil((numInputs + numClasses) / 2), numClasses, "softmax");
    }

    return trainSet.map(item => ({
      input: item.input,
      output: uniquetags.map(tag =>
        (Array.isArray(item.output) ? item.output : [item.output])
        .includes(tag) ? 1 : 0
      )
    }));
  }

  #evaluateTestSet(testSet, lossFunction) {
    return testSet.reduce((error, data) => {
      const prediction = this.predict(data.input, false);
      return error + this.#lossFunctions[lossFunction].loss(prediction, data.output);
    }, 0) / testSet.length;
  }

  #generateTrainingSummary(start, end, {
    epochs,
    learningRate,
    lastTrainLoss,
    lastTestLoss
  }) {
    const totalParams = this.weights.reduce((sum, layer, i) =>
      sum + layer.flat()
      .length + this.biases[i].length, 0
    );

    return {
      parameters: totalParams,
      training: {
        loss: lastTrainLoss,
        testloss: lastTestLoss,
        time: end - start,
        epochs,
        learningRate,
      },
    };
  }

  predict(input, tags = true) {
    const {
      layerInputs,
      layerRawOutputs
    } = this.#forwardPropagate(input);
    const output = layerInputs[layerInputs.length - 1];

    if (this.tags &&
      this.layers[this.layers.length - 1].activation === "softmax" &&
      tags) {
      return output
        .map((prob, idx) => ({
          tag: this.tags[idx],
          probability: prob,
        }))
        .sort((a, b) => b.probability - a.probability);
    }

    return output;
  }

async save(name = "model", useBinary = false) {
  try {
    console.log("Starting save process...");
    console.log("Model state:", {
      weightLayers: this.weights?.length,
      biasLayers: this.biases?.length,
      modelLayers: this.layers?.length
    });

    // Validate weights and biases
    if (!this.weights?.length || !this.biases?.length) {
      throw new Error("Weights or biases are empty. Cannot save model.");
    }

    // Prepare metadata
    if (!this.details.info) {
      this.details.info = {
        name: name,
        author: '',
        license: 'MIT',
        note: '',
        date: new Date().toISOString()
      };
    }

    // Prepare metadata for weights/biases structure
    const layerInfo = {
      weightShapes: this.weights.map(layer => [layer.length, layer[0].length]),
      biasShapes: this.biases.map(layer => layer.length)
    };

    // Create metadata object
    const metadata = {
      layers: this.layers,
      details: this.details,
      layerInfo: layerInfo,
      ...(this.tags && { tags: this.tags })
    };

    let fileBlob;
    if (useBinary) {
      console.log("Using binary compression mode");

      // Calculate total buffer size needed
      const totalWeights = this.weights.reduce((sum, layer) => 
        sum + layer.length * layer[0].length, 0);
      const totalBiases = this.biases.reduce((sum, layer) => 
        sum + layer.length, 0);

      // Create buffers
      const weightBuffer = new Float32Array(totalWeights);
      const biasBuffer = new Float32Array(totalBiases);

      // Flatten weights with proper indexing
      let weightIndex = 0;
      for (const layer of this.weights) {
        for (const row of layer) {
          for (const value of row) {
            weightBuffer[weightIndex++] = value;
          }
        }
      }

      // Flatten biases with proper indexing
      let biasIndex = 0;
      for (const layer of this.biases) {
        for (const value of layer) {
          biasBuffer[biasIndex++] = value;
        }
      }

      // Convert metadata to string and create header
      const metadataString = JSON.stringify(metadata);
      const metadataBytes = new TextEncoder().encode(metadataString);
      
      // Calculate padding for alignment
      const metadataPadding = (4 - (metadataBytes.length % 4)) % 4;

      // Create header with sizes and padding info
      const header = new Uint32Array([
        metadataBytes.length,
        metadataPadding,
        weightBuffer.length,
        biasBuffer.length
      ]);

      console.log("Header values:", {
        metadataLength: header[0],
        padding: header[1],
        weightLength: header[2],
        biasLength: header[3]
      });

      // Calculate total size with padding
      const totalSize = header.byteLength + 
                       metadataBytes.length +
                       metadataPadding +
                       weightBuffer.byteLength + 
                       biasBuffer.byteLength;

      // Combine all buffers
      const combinedBuffer = new Uint8Array(totalSize);
      
      let offset = 0;
      
      // Write header
      combinedBuffer.set(new Uint8Array(header.buffer), offset);
      offset += header.byteLength;
      
      // Write metadata
      combinedBuffer.set(metadataBytes, offset);
      offset += metadataBytes.length;
      
      // Add padding
      offset += metadataPadding;
      
      // Write weights
      combinedBuffer.set(new Uint8Array(weightBuffer.buffer), offset);
      offset += weightBuffer.byteLength;
      
      // Write biases
      combinedBuffer.set(new Uint8Array(biasBuffer.buffer), offset);

      fileBlob = new Blob([combinedBuffer], { type: "application/octet-stream" });
      
      console.log("Binary blob created:", {
        size: fileBlob.size,
        expectedSize: totalSize
      });

    } else {
      console.log("Using JSON mode");
      metadata.weights = this.weights;
      metadata.biases = this.biases;
      fileBlob = new Blob([JSON.stringify(metadata)], { type: "application/json" });
    }

    const extension = useBinary ? '.uai' : '.json';
    const downloadUrl = URL.createObjectURL(fileBlob);
    
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = `${name}${extension}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(downloadUrl);

    console.log("Save process completed successfully");
    return true;
  } catch (error) {
    console.error("Save process failed:", error);
    throw error;
  }
}

async load(callback, useBinary = false) {
  try {
    console.log("Starting load process...");

    const input = document.createElement('input');
    input.type = 'file';
    input.accept = useBinary ? '.uai' : '.json';

    const [file] = await new Promise(resolve => {
      input.onchange = e => resolve(e.target.files);
      document.body.appendChild(input);
      input.click();
      document.body.removeChild(input);
    });

    if (!file) throw new Error("No file selected");

    console.log("File selected:", {
      name: file.name,
      size: file.size,
      type: file.type
    });

    const arrayBuffer = await file.arrayBuffer();
    
    let metadata;
    if (useBinary) {
      // Read header (4 values)
      const headerView = new Uint32Array(arrayBuffer, 0, 4);
      const [metadataLength, metadataPadding, weightLength, biasLength] = headerView;

      console.log("Header values:", {
        metadataLength,
        padding: metadataPadding,
        weightLength,
        biasLength,
        headerSize: headerView.byteLength
      });

      // Calculate aligned offsets
      const metadataOffset = headerView.byteLength;
      const weightOffset = metadataOffset + metadataLength + metadataPadding;
      const biasOffset = weightOffset + (weightLength * 4);

      console.log("Buffer offsets:", {
        metadataOffset,
        weightOffset,
        biasOffset
      });

      // Validate buffer size
      const expectedSize = biasOffset + (biasLength * 4);
      if (arrayBuffer.byteLength !== expectedSize) {
        throw new Error(`Invalid buffer size: expected ${expectedSize}, got ${arrayBuffer.byteLength}`);
      }

      // Read metadata
      const metadataBytes = new Uint8Array(arrayBuffer, metadataOffset, metadataLength);
      metadata = JSON.parse(new TextDecoder().decode(metadataBytes));

      // Read weights
      const weightBuffer = new Float32Array(arrayBuffer, weightOffset, weightLength);
      let weightIndex = 0;
      this.weights = metadata.layerInfo.weightShapes.map(shape => {
        const layerWeights = [];
        for (let i = 0; i < shape[0]; i++) {
          const row = Array.from(weightBuffer.slice(weightIndex, weightIndex + shape[1]));
          layerWeights.push(row);
          weightIndex += shape[1];
        }
        return layerWeights;
      });

      // Read biases
      const biasBuffer = new Float32Array(arrayBuffer, biasOffset, biasLength);
      let biasIndex = 0;
      this.biases = metadata.layerInfo.biasShapes.map(shape => {
        const layerBiases = Array.from(biasBuffer.slice(biasIndex, biasIndex + shape));
        biasIndex += shape;
        return layerBiases;
      });

    } else {
      metadata = JSON.parse(new TextDecoder().decode(arrayBuffer));
      this.weights = metadata.weights;
      this.biases = metadata.biases;
    }

    // Load other metadata
    this.layers = metadata.layers;
    this.details = metadata.details;
    if (metadata.tags) this.tags = metadata.tags;

    console.log("Load process completed successfully");
    callback?.();
    return true;
  } catch (error) {
    console.error("Load process failed:", error);
    throw error;
  }
}
  
  info(infoUpdates) {
    this.details.info = infoUpdates
  }
}
