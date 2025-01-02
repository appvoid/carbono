// This is an experimental version of carbono that supports text, images and audio as inputs through urls

class carbono {
  constructor(debug = true) {
      this.layers = [];
      this.weights = [];
      this.biases = [];
      this.details = {};
      this.debug = debug;
    }

  async preprocessData(input) {
    if (typeof input === 'string' && this.#isUrl(input)) {
      try {
        const response = await fetch(input);
        if (!response.ok) {
          throw new Error(`Failed to fetch data from ${input}: ${response.statusText}`);
        }

        const contentType = response.headers.get('Content-Type');
        const inferredType = this.#inferContentType(input, contentType);

        if (inferredType === 'image') {
          return await this.#preprocessImage(response);
        } else if (inferredType === 'audio') {
          return await this.#preprocessAudio(response);
        } else if (inferredType === 'text') {
          return await this.#preprocessText(response);
        } else {
          throw new Error(`Unsupported data type for ${input}`);
        }
      } catch (error) {
        console.error(`Error preprocessing data from ${input}:`, error);
        throw error;
      }
    } else {
      return input;
    }
  }

  #isUrl(input) {
    try {
      new URL(input);
      return true;
    } catch (_) {
      return false;
    }
  }

  #inferContentType(url, contentType) {
    if (contentType && !contentType.includes('binary/octet-stream')) {
      if (contentType.startsWith('image/')) return 'image';
      if (contentType.startsWith('audio/')) return 'audio';
      if (contentType.startsWith('text/')) return 'text';
    }

    const extension = url.split('.').pop().toLowerCase();
    if (['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'].includes(extension)) {
      return 'image';
    } else if (['wav', 'mp3', 'ogg', 'aac', 'flac'].includes(extension)) {
      return 'audio';
    } else if (['txt', 'csv', 'json', 'html', 'xml'].includes(extension)) {
      return 'text';
    }

    throw new Error(`Unable to infer content type for ${url}`);
  }

  async #preprocessImage(response) {
    try {
      const blob = await response.blob();
      const img = document.createElement('img');
      img.src = URL.createObjectURL(blob);
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = () => reject(new Error('Failed to load image'));
      });

      const canvas = document.createElement('canvas');
      canvas.width = 28;
      canvas.height = 28;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, 28, 28);

      const pixels = ctx.getImageData(0, 0, 28, 28).data;
      const normalizedPixels = Array.from(pixels).map(val => val / 255);
      return normalizedPixels;
    } catch (error) {
      console.error('Error preprocessing image:', error);
      throw error;
    }
  }

  async #preprocessAudio(response) {
    try {
      const arrayBuffer = await response.arrayBuffer();
      const offlineContext = new OfflineAudioContext(2, 2048, 44100);
      const sourceNode = offlineContext.createBufferSource();
      sourceNode.connect(offlineContext.destination);
      const audioBuffer = await offlineContext.decodeAudioData(arrayBuffer);
      const analyser = offlineContext.createAnalyser();
      analyser.fftSize = 2048;
      analyser.minDecibels = -90;
      analyser.maxDecibels = -10;
      analyser.smoothingTimeConstant = 0.8;

      sourceNode.connect(analyser);
      analyser.connect(offlineContext.destination);
      sourceNode.buffer = audioBuffer;
      sourceNode.start();

      await offlineContext.startRendering();
      const frequencyData = new Uint8Array(analyser.frequencyBinCount);
      analyser.getByteFrequencyData(frequencyData);
      const averagedFrequencyData = frequencyData.slice(0);
      const normalizedFrequencyData = Array.from(averagedFrequencyData).map(val => val / 255);

      // Use the first layer's input size for padding/truncating
      const targetSize = this.layers[0]?.inputSize || frequencyData.length;
      return this.#padOrTruncate(normalizedFrequencyData, targetSize);
    } catch (error) {
      console.error('Error preprocessing audio:', error);
      throw error;
    }
  }

  async #preprocessText(response) {
    try {
      const text = await response.text();
      const words = text
        .toLowerCase()
        .replace(/[^\w\s]/g, '')
        .split(/\s+/)
        .filter(word => word.length > 0);

      const wordFrequencies = {};
      words.forEach(word => wordFrequencies[word] = (wordFrequencies[word] || 0) + 1);

      const totalWords = words.length;
      const uniqueWords = Object.keys(wordFrequencies);
      const idf = {};
      uniqueWords.forEach(word => idf[word] = Math.log(totalWords / (wordFrequencies[word] + 1)));

      const tfidfVector = uniqueWords.map(word => {
        const tf = wordFrequencies[word] / totalWords;
        return tf * idf[word];
      });

      // Use the first layer's input size for padding/truncating
      const targetSize = this.layers[0]?.inputSize || tfidfVector.length;
      return this.#padOrTruncate(tfidfVector, targetSize);
    } catch (error) {
      console.error('Error preprocessing text:', error);
      throw error;
    }
  }

  #padOrTruncate(data, targetSize) {
    if (data.length > targetSize) {
      return data.slice(0, targetSize);
    } else if (data.length < targetSize) {
      return data.concat(Array(targetSize - data.length).fill(0));
    }
    return data;
  }
  async trainFromUrls(trainSetUrls, options = {}) {
    try {
      // Convert URLs to preprocessed data
      const trainSet = await Promise.all(trainSetUrls.map(async data => {
        try {
          const input = await this.preprocessData(data.url);
          return { input, output: data.output };
        } catch (error) {
          console.error(`Error preprocessing data for ${data.url}:`, error);
          throw error;
        }
      }));

      // Proceed with training
      return this.train(trainSet, options);
    } catch (error) {
      console.error('Error during training:', error);
      throw error;
    }
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

async train(trainSet, options = {}) {
    if (!('debug' in this)) {
        this.debug = true;
    }
    
    const {
        epochs = 10, 
        learningRate = 0.212, 
        printEveryEpochs = 1, 
        earlyStopThreshold = 1e-6, 
        testSet = null, 
        callback = null, 
        optimizer = "sgd", 
        lossFunction = "mse"
    } = options;

    // Preprocess the training set to handle URLs
    const processedTrainSet = await Promise.all(trainSet.map(async data => {
        try {
            // If the data has a URL, preprocess it
            if (data.url) {
                const processedInput = await this.preprocessData(data.url);
                return { input: processedInput, output: data.output };
            }
            // If data already has numerical input, use it directly
            return { input: data.input, output: data.output };
        } catch (error) {
            console.error(`Error preprocessing data:`, error);
            throw error;
        }
    }));

    // Process string outputs into one-hot encoded format if needed
    if (typeof processedTrainSet[0].output === "string" ||
        (Array.isArray(processedTrainSet[0].output) && 
         typeof processedTrainSet[0].output[0] === "string")) {
        return this.#trainWithPreprocessedData(
            this.#preprocesstags(processedTrainSet),
            options
        );
    }

    return this.#trainWithPreprocessedData(processedTrainSet, options);
}
  
  // Training
  async #trainWithPreprocessedData(trainSet, options) {
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

  async predict(input, tags = true) {
    // Preprocess the input if it's a URL
    const preprocessedInput = await this.preprocessData(input);

    // Forward propagate
    const {
      layerInputs,
      layerRawOutputs
    } = this.#forwardPropagate(preprocessedInput);
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

  async save(name = "model") {
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

    // If no custom name is set, use the save parameter
    if (this.details.info.name === 'Untitled Model') {
      this.details.info.name = name;
    }

    // Flatten and convert weights and biases to Float32Array
    const flattenWeights = this.weights.flatMap(layer => 
      layer.flatMap(row => row.map(val => val))
    );
    const flattenBiases = this.biases.flatMap(layer => layer.map(val => val));

    const weightBuffer = new Float32Array(flattenWeights);
    const biasBuffer = new Float32Array(flattenBiases);

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

    // Convert metadata to string and create binary data
    const metadataString = JSON.stringify(metadata);
    const separator = '\n---BINARY_SEPARATOR---\n';
    
    // Create concatenated binary data
    const binaryData = new Uint8Array([
      ...new TextEncoder().encode(metadataString),
      ...new TextEncoder().encode(separator),
      ...new Uint8Array(weightBuffer.buffer),
      ...new Uint8Array(biasBuffer.buffer)
    ]);

    // Create blob and download
    const fileBlob = new Blob([binaryData], { type: "application/octet-stream" });
    const downloadUrl = URL.createObjectURL(fileBlob);

    try {
      const link = Object.assign(document.createElement('a'), {
        href: downloadUrl,
        download: `${this.details.info.name}.uai`,
        style: 'display: none'
      });

      document.body.appendChild(link);
      link.click();
    } finally {
      URL.revokeObjectURL(downloadUrl);
    }
  }

  async load(callback) {
    const createFileInput = () => Object.assign(document.createElement('input'), {
      type: 'file',
      accept: '.uai',
      style: 'display: none'
    });

    const readFile = file => new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = e => resolve(e.target.result);
      reader.onerror = reject;
      reader.readAsArrayBuffer(file);
    });

    try {
      const input = createFileInput();
      document.body.appendChild(input);

      const [file] = await new Promise(resolve => {
        input.onchange = e => resolve(e.target.files);
        input.click();
      });

      if (!file) return;

      const fileContent = await readFile(file);
      const dataView = new Uint8Array(fileContent);
      
      // Find the separator position
      const separator = '\n---BINARY_SEPARATOR---\n';
      const separatorBytes = new TextEncoder().encode(separator);
      let separatorIndex = -1;
      
      for (let i = 0; i < dataView.length - separatorBytes.length; i++) {
        if (dataView[i] === separatorBytes[0]) {
          let found = true;
          for (let j = 0; j < separatorBytes.length; j++) {
            if (dataView[i + j] !== separatorBytes[j]) {
              found = false;
              break;
            }
          }
          if (found) {
            separatorIndex = i;
            break;
          }
        }
      }

      if (separatorIndex === -1) throw new Error('Invalid file format');

      // Split metadata and binary data
      const metadataBytes = dataView.slice(0, separatorIndex);
      const metadata = JSON.parse(new TextDecoder().decode(metadataBytes));
      
      // Calculate total sizes
      const totalWeights = metadata.layerInfo.weightShapes.reduce((sum, shape) => sum + shape[0] * shape[1], 0);
      const totalBiases = metadata.layerInfo.biasShapes.reduce((a, b) => a + b, 0);
      
      // Extract binary data
      const binaryStart = separatorIndex + separatorBytes.length;
      const weightBuffer = new Float32Array(fileContent, binaryStart, totalWeights);
      const biasBuffer = new Float32Array(fileContent, binaryStart + totalWeights * 4, totalBiases);

      // Reconstruct weights
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

      // Reconstruct biases
      let biasIndex = 0;
      this.biases = metadata.layerInfo.biasShapes.map(shape => {
        const layerBiases = Array.from(biasBuffer.slice(biasIndex, biasIndex + shape));
        biasIndex += shape;
        return layerBiases;
      });

      // Load other metadata
      this.layers = metadata.layers;
      this.details = metadata.details;
      if (metadata.tags) this.tags = metadata.tags;

      this.debug && console.log("✅ Model loaded successfully!");
      callback?.();
    } catch (error) {
      this.debug && console.error("❌ Failed to load model:", error);
    } finally {
      delete this.debug;
      document.querySelector('input[type="file"]')?.remove();
    }
  }

  info(infoUpdates) {
    this.details.info = infoUpdates;
  }
}

// Example usage
const model = new carbono();
// model.load(()=>{
//   model.predict('https://cdn.pixabay.com/photo/2024/01/29/20/40/cat-8540772_1280.jpg').then(prediction => {
//   console.log('Prediction:', prediction);
// });
// })
// Define the model architecture
model.layer(784, 32, 'sigmoid');
model.layer(32, 3, 'softmax');

// Prepare the training data with URLs
const trainSetUrls = [
  { url: 'https://cdn.pixabay.com/photo/2024/01/29/20/40/cat-8540772_1280.jpg', output: 'image'},
  { url: 'https://cdn.jsdelivr.net/gh/lunu-bounir/audio-equalizer/test/left.ogg', output: 'audio'},
      { url: 'https://raw.githubusercontent.com/appvoid/carbono/refs/heads/main/examples.md', output: 'code' },
];

// Train the model
model.train(trainSetUrls, {
  epochs: 10,
  optimizer: 'adam',
  learningRate: 0.1,
  printEveryEpochs: 2,
}).then(summary => {
  model.predict('https://cdn.pixabay.com/photo/2024/01/29/20/40/cat-8540772_1280.jpg').then(prediction => {
  console.log('Prediction:', prediction);
    // console.log(model)
    // model.save()
});
});
