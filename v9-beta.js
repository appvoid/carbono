class carbono {
  // 🏗️ Constructor: Initializes the neural network with default settings
  constructor(debug = true, seed = 1234) {
    this.seed = this.#rng.set(seed); // Sets default seed for easy reproducible results
    this.layers = []; // Stores the layers of the neural network
    this.weights = []; // Stores the weights for each layer
    this.biases = []; // Stores the biases for each layer
    this.details = {}; // Stores metadata about the model (e.g., training details)
  }
  #rng = {
    // Set the random seed
    set(seed) {
      this.seed = seed;
      return seed;
    },
    // Generate seeded random number between 0 and 1
    random() {
      const x = Math.sin(this.seed++) * 10000;
      return x - Math.floor(x);
    }
  };
  #initializeWeights(inputSize, outputSize, activation) {
    switch (activation) {
      case 'relu':
        // He initialization
        // ReLU kills half the signal (negative parts), so multiply by 2
        return this.#rng.random() * Math.sqrt(2 / inputSize);
      case 'tanh':
      case 'sigmoid':
        // Xavier/Glorot with proper variance
        return (this.#rng.random() - 0.5) * Math.sqrt(2 / inputSize);
      case 'softmax':
        // Initialize smaller to start with more uniform probabilities
        return (this.#rng.random() - 0.5) * Math.sqrt(1 / inputSize);
      default:
        throw new Error(`Unknown activation: ${activation}`);
    }
  }
  #calculateGlobalGradientNorm(layerErrors, layerInputs) {
    let sumSquaredGradients = 0;
    for (let i = 0; i < this.weights.length; i++) {
      for (let j = 0; j < this.weights[i].length; j++) {
        for (let k = 0; k < this.weights[i][j].length; k++) {
          const gradient = layerErrors[i][j] * layerInputs[i][k];
          sumSquaredGradients += gradient * gradient;
        }
      }
    }
    return Math.sqrt(sumSquaredGradients);
  }
  // 🛠️ Utility Methods
  // Clip: Ensures values stay within a specified range to avoid numerical instability
  #clip(value, min = 1e-15, max = 1 - 1e-15) {
    return Math.max(Math.min(value, max), min);
  }
  // 🧠 Activation Functions
  // Activation Functions: Define how neurons activate
  #activationFunctions = {
    tanh: {
      fn: x => Math.tanh(x), // Hyperbolic tangent function
      derivative: x => 1 - Math.pow(Math.tanh(x), 2) // Derivative of tanh
    },
    sigmoid: {
      fn: x => 1 / (1 + Math.exp(-x)), // Sigmoid function
      derivative: x => {
        const sig = 1 / (1 + Math.exp(-x));
        return sig * (1 - sig); // Derivative of sigmoid
      }
    },
    relu: {
      fn: x => Math.max(0, x), // Rectified Linear Unit (ReLU)
      derivative: x => x > 0 ? 1 : 0 // Derivative of ReLU
    },
    softmax: {
      fn: x => {
        const expValues = Array.isArray(x) ? x.map(val => Math.exp(val)) : [Math.exp(x)];
        const sumExp = expValues.reduce((a, b) => a + b, 0);
        return expValues.map(exp => exp / sumExp); // Softmax function (used for classification)
      },
      derivative: null // Softmax derivative is handled differently
    }
  }
  // 📉 Loss Functions
  // Loss Functions: Measure how well the model is performing (e.g., Mean Squared Error, Cross-Entropy)
  #lossFunctions = {
    mse: {
      loss: (predicted, actual) => predicted.reduce((sum, pred, i) => sum + Math.pow(pred - actual[i], 2), 0), // Mean Squared Error
      derivative: (predicted, actual, activation) => predicted.map((pred, i) => (pred - actual[i]) * (activation === 'softmax' ? 1 : this.#getActivationDerivative(pred, activation))) // Derivative of MSE
    },
    'cross-entropy': {
      loss: (predicted, actual) => -actual.reduce((sum, target, i) => sum + target * Math.log(this.#clip(predicted[i])), 0), // Cross-Entropy Loss
      derivative: (predicted, actual) => predicted.map((pred, i) => pred - actual[i]) // Derivative of Cross-Entropy
    }
  }
  // 🔧 Helper Methods for Activation Functions
  // Get Activation: Applies the activation function to a value
  #getActivation(x, activation) {
    return this.#activationFunctions[activation].fn(x);
  }
  // Get Activation Derivative: Returns the derivative of the activation function
  #getActivationDerivative(x, activation) {
    return this.#activationFunctions[activation].derivative?.(x) ?? null;
  }
  // 🧱 Layer Management
  // Drop Layer: Removes a layer at the specified index
  drop(layerIndex) {
    if (layerIndex < 0 || layerIndex >= this.layers.length) {
      throw new Error("Invalid layer index");
    }
    // Remove the layer, weights, and biases
    this.layers.splice(layerIndex, 1);
    this.weights.splice(layerIndex, 1);
    this.biases.splice(layerIndex, 1);
    // If we removed a middle layer, reconnect the adjacent layers
    if (layerIndex < this.layers.length) {
      // Update input size of the next layer
      const prevLayerOutputSize = layerIndex > 0 ? this.layers[layerIndex - 1].outputSize : this.layers[0].inputSize;
      const nextLayer = this.layers[layerIndex];
      nextLayer.inputSize = prevLayerOutputSize;
      // Reinitialize weights for the connection
      this.weights[layerIndex] = Array(nextLayer.outputSize).fill().map(() => Array(prevLayerOutputSize).fill().map(() => this.#initializeWeights(prevLayerOutputSize, nextLayer.outputSize, nextLayer.activation)));
    }
    return this;
  }
  #createDropoutMask(size, dropoutRate) {
    return Array(size).fill().map(() => this.#rng.random() > dropoutRate ? 1 / (1 - dropoutRate) : 0);
  }
  // Add Layer: Adds a new layer to the neural network
  layer(inputSize, outputSize, activation = "tanh", options = {}) {
    const {
      dropoutRate = 0, l1 = 0, // L1 regularization strength
        l2 = 0 // L2 regularization strength
    } = options;
    if (this.weights.length > 0) {
      const lastLayerOutputSize = this.layers[this.layers.length - 1].outputSize;
      if (inputSize !== lastLayerOutputSize) {
        throw new Error("Layer input size must match previous layer output size.");
      }
    }
    this.layers.push({
      inputSize,
      outputSize,
      activation,
      dropoutRate,
      l1,
      l2
    });
    const weights = Array(outputSize).fill().map(() => Array(inputSize).fill().map(() => this.#initializeWeights(inputSize, outputSize, activation)));
    this.weights.push(weights);
    this.biases.push(Array(outputSize).fill(0.01));
    return this;
  }
  #calculateRegularizationLoss() {
    return this.layers.reduce((totalLoss, layer, layerIndex) => {
      if (!layer.l1 && !layer.l2) return totalLoss;
      let layerLoss = 0;
      const weights = this.weights[layerIndex];
      if (layer.l1 > 0) {
        const l1Loss = weights.reduce((sum, neuron) => sum + neuron.reduce((nSum, weight) => nSum + Math.abs(weight), 0), 0);
        layerLoss += layer.l1 * l1Loss;
      }
      if (layer.l2 > 0) {
        const l2Loss = weights.reduce((sum, neuron) => sum + neuron.reduce((nSum, weight) => nSum + weight * weight, 0), 0);
        layerLoss += layer.l2 * l2Loss / 2;
      }
      return totalLoss + layerLoss;
    }, 0);
  }
  // Quantization Methods
  quantize(calibrationData = null) {
    this.quants = [];
    this.layers.forEach((layer, layerIdx) => {
      // Find min/max values for weights and biases
      const weights = this.weights[layerIdx];
      const biases = this.biases[layerIdx];
      let maxWeight = -Infinity;
      let minWeight = Infinity;
      weights.forEach(neuron => {
        neuron.forEach(w => {
          maxWeight = Math.max(maxWeight, w);
          minWeight = Math.min(minWeight, w);
        });
      });
      let maxBias = Math.max(...biases);
      let minBias = Math.min(...biases);
      // Calculate scales
      const weightScale = Math.max(Math.abs(minWeight), Math.abs(maxWeight)) / 127;
      const biasScale = Math.max(Math.abs(minBias), Math.abs(maxBias)) / 127;
      // Quantize weights
      const quantizedWeights = weights.map(neuron => {
        return new Int8Array(neuron.map(w => Math.min(Math.max(Math.round(w / weightScale), -127), 127)));
      });
      // Quantize biases
      const quantizedBiases = new Int8Array(biases.map(b => Math.min(Math.max(Math.round(b / biasScale), -127), 127)));
      // Store quantized values and scales
      this.weights[layerIdx] = quantizedWeights;
      this.biases[layerIdx] = quantizedBiases;
      this.quants[layerIdx] = {
        weightScale,
        biasScale
      };
    });
    this.quantized = true;
    return this;
  }
  // Modified forward propagation
  #forwardPropagate(input, isTraining = true) {
    let current = Array.from(input); // Make a copy of input
    const layerInputs = [current];
    const layerRawOutputs = [];
    const dropoutMasks = [];
    for (let i = 0; i < this.weights.length; i++) {
      const rawOutput = [];
      const weights = this.weights[i];
      const biases = this.biases[i];
      const layer = this.layers[i];
      const quants = this.quantized ? this.quants[i] : null;
      // Apply dropout only during training and if dropoutRate > 0
      if (isTraining && layer.dropoutRate > 0) {
        // Create dropout mask
        const dropoutMask = Array(current.length).fill().map(() => {
          // Ensure we don't drop all neurons
          const shouldDrop = Math.random() <= layer.dropoutRate;
          return shouldDrop ? 0 : 1 / (1 - layer.dropoutRate);
        });
        // Ensure at least one neuron is active
        if (dropoutMask.every(x => x === 0)) {
          const randomIndex = Math.floor(Math.random() * dropoutMask.length);
          dropoutMask[randomIndex] = 1 / (1 - layer.dropoutRate);
        }
        dropoutMasks.push(dropoutMask);
        // Apply mask with numerical stability check
        current = current.map((val, idx) => {
          const result = val * dropoutMask[idx];
          return Number.isFinite(result) ? result : 0;
        });
      } else {
        dropoutMasks.push(null);
      }
      // Compute layer output
      for (let j = 0; j < weights.length; j++) {
        let sum = 0;
        for (let k = 0; k < current.length; k++) {
          const w = weights[j][k];
          const actualWeight = quants ? w * quants.weightScale : w;
          const product = actualWeight * current[k];
          if (Number.isFinite(product)) {
            sum += product;
          }
        }
        const b = biases[j];
        const actualBias = quants ? b * quants.biasScale : b;
        // Add bias with numerical stability check
        if (Number.isFinite(sum) && Number.isFinite(actualBias)) {
          sum += actualBias;
        }
        rawOutput.push(sum);
      }
      layerRawOutputs.push(rawOutput);
      const layerActivation = layer.activation;
      // Apply activation function with numerical stability
      if (layerActivation === 'softmax') {
        current = this.#getActivation(rawOutput.map(x => Number.isFinite(x) ? x : 0), 'softmax');
      } else {
        current = rawOutput.map(x => {
          const activated = this.#getActivation(Number.isFinite(x) ? x : 0, layerActivation);
          return Number.isFinite(activated) ? activated : 0;
        });
      }
      layerInputs.push(current);
    }
    return {
      layerInputs,
      layerRawOutputs,
      dropoutMasks
    };
  }
  // Modified save method
  async save(name = "model", useBinary = false) {
    try {
      if (useBinary) {
        const metadata = {
          layers: this.layers,
          details: this.details,
          quantization: this.quantized ? {
            enabled: true,
            quants: this.quants
          } : null,
          ...(this.tags && {
            tags: this.tags
          })
        };
        const metadataString = JSON.stringify(metadata);
        const metadataBytes = new TextEncoder().encode(metadataString);
        // Ensure 8-byte alignment for Float64Array
        const metadataPadding = (8 - (metadataBytes.length % 8)) % 8;
        let totalWeights = 0;
        let totalBiases = 0;
        this.weights.forEach((layer, i) => {
          layer.forEach(neuron => {
            totalWeights += neuron.length;
          });
          totalBiases += this.biases[i].length;
        });
        const header = new Uint32Array([
          metadataBytes.length, metadataPadding, totalWeights, totalBiases
        ]);
        // Calculate aligned size
        const totalSize = header.byteLength + metadataBytes.length + metadataPadding + (this.quantized ? (totalWeights + totalBiases) : (totalWeights + totalBiases) * 8);
        const buffer = new ArrayBuffer(totalSize);
        const view = new Uint8Array(buffer);
        let offset = 0;
        // Write header
        view.set(new Uint8Array(header.buffer), offset);
        offset += header.byteLength;
        // Write metadata with padding
        view.set(metadataBytes, offset);
        offset += metadataBytes.length + metadataPadding;
        // Ensure offset is 8-byte aligned
        if (offset % 8 !== 0) {
          offset += (8 - (offset % 8));
        }
        // Write weights
        for (let i = 0; i < this.weights.length; i++) {
          for (let j = 0; j < this.weights[i].length; j++) {
            if (this.quantized) {
              view.set(this.weights[i][j], offset);
              offset += this.weights[i][j].length;
            } else {
              const floatView = new Float64Array(buffer, offset, this.weights[i][j].length);
              floatView.set(this.weights[i][j]);
              offset += this.weights[i][j].length * 8;
            }
          }
        }
        // Ensure offset is 8-byte aligned before biases
        if (!this.quantized && offset % 8 !== 0) {
          offset += (8 - (offset % 8));
        }
        // Write biases
        for (let i = 0; i < this.biases.length; i++) {
          if (this.quantized) {
            view.set(this.biases[i], offset);
            offset += this.biases[i].length;
          } else {
            const floatView = new Float64Array(buffer, offset, this.biases[i].length);
            floatView.set(this.biases[i]);
            offset += this.biases[i].length * 8;
          }
        }
        const fileBlob = new Blob([buffer], {
          type: "application/octet-stream"
        });
        const downloadUrl = URL.createObjectURL(fileBlob);
        const link = document.createElement('a');
        link.href = downloadUrl;
        link.download = `${name}.uai`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(downloadUrl);
        return true;
      } else {
        // For standard JSON saving
        const metadata = {
          layers: this.layers,
          details: this.details,
          ...(this.tags && {
            tags: this.tags
          })
        };
        // Add quantization data if model is quantized
        if (this.quantized) {
          metadata.quantization = {
            enabled: true,
            quants: this.quants,
            weights: this.weights.map(layer => layer.map(neuron => Array.from(neuron))),
            biases: this.biases.map(bias => Array.from(bias))
          };
        } else {
          // Add standard weights and biases if not quantized
          metadata.weights = this.weights;
          metadata.biases = this.biases;
        }
        const fileBlob = new Blob([JSON.stringify(metadata)], {
          type: "application/json"
        });
        const downloadUrl = URL.createObjectURL(fileBlob);
        const link = document.createElement('a');
        link.href = downloadUrl;
        link.download = `${name}.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(downloadUrl);
        return true;
      }
    } catch (error) {
      console.error("Save process failed:", error);
      throw error;
    }
  }
  // Modified load method
  async load(callback, useBinary = false) {
    try {
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
      const arrayBuffer = await file.arrayBuffer();
      if (useBinary) {
        const headerView = new Uint32Array(arrayBuffer, 0, 4);
        const [metadataLength, metadataPadding, totalWeights, totalBiases] = headerView;
        const metadataOffset = headerView.byteLength;
        const metadataBytes = new Uint8Array(arrayBuffer, metadataOffset, metadataLength);
        const metadata = JSON.parse(new TextDecoder().decode(metadataBytes));
        let offset = metadataOffset + metadataLength + metadataPadding;
        this.quantized = metadata.quantization?.enabled || false;
        this.quants = metadata.quantization?.quants || null;
        this.layers = metadata.layers;
        this.details = metadata.details;
        if (metadata.tags) this.tags = metadata.tags;
        // Load weights
        this.weights = [];
        metadata.layers.forEach((layer, i) => {
          const layerWeights = [];
          for (let j = 0; j < layer.outputSize; j++) {
            if (this.quantized) {
              const neuron = new Int8Array(arrayBuffer, offset, layer.inputSize);
              layerWeights.push(new Int8Array(neuron));
              offset += layer.inputSize;
            } else {
              const neuron = new Float64Array(arrayBuffer, offset, layer.inputSize);
              layerWeights.push(Array.from(neuron));
              offset += layer.inputSize * 8;
            }
          }
          this.weights.push(layerWeights);
        });
        // Load biases
        this.biases = [];
        metadata.layers.forEach(layer => {
          if (this.quantized) {
            const bias = new Int8Array(arrayBuffer, offset, layer.outputSize);
            this.biases.push(new Int8Array(bias));
            offset += layer.outputSize;
          } else {
            const bias = new Float64Array(arrayBuffer, offset, layer.outputSize);
            this.biases.push(Array.from(bias));
            offset += layer.outputSize * 8;
          }
        });
      } else {
        const metadata = JSON.parse(new TextDecoder().decode(arrayBuffer));
        this.quantized = metadata.quantization?.enabled || false;
        this.quants = metadata.quantization?.quants || null;
        this.layers = metadata.layers;
        this.details = metadata.details;
        if (metadata.tags) this.tags = metadata.tags;
        if (this.quantized) {
          this.weights = metadata.quantization.weights.map(layer => layer.map(neuron => new Int8Array(neuron)));
          this.biases = metadata.quantization.biases.map(bias => new Int8Array(bias));
        } else {
          this.weights = metadata.weights;
          this.biases = metadata.biases;
        }
      }
      callback?.();
      return true;
    } catch (error) {
      console.error("Load process failed:", error);
      throw error;
    }
  }
  // ⬅️ Backward Propagation
  // Back Propagate: Calculates errors and updates weights/biases
  #backPropagate(layerInputs, layerRawOutputs, target, lossFunction, dropoutMasks) {
    const outputLayer = this.layers[this.layers.length - 1];
    const outputErrors = this.#lossFunctions[lossFunction].derivative(layerInputs[layerInputs.length - 1], target, outputLayer.activation);
    const layerErrors = [outputErrors];
    for (let i = this.weights.length - 2; i >= 0; i--) {
      const errors = Array(this.layers[i].outputSize).fill(0);
      for (let j = 0; j < this.layers[i].outputSize; j++) {
        for (let k = 0; k < this.layers[i + 1].outputSize; k++) {
          errors[j] += layerErrors[0][k] * this.weights[i + 1][k][j];
        }
        const activationDeriv = this.#getActivationDerivative(layerRawOutputs[i][j], this.layers[i].activation);
        if (activationDeriv !== null) {
          errors[j] *= activationDeriv;
        }
        // Apply dropout mask if it exists
        if (dropoutMasks[i]) {
          errors[j] *= dropoutMasks[i][j];
        }
      }
      layerErrors.unshift(errors);
    }
    return layerErrors;
  }
  // 🚀 Optimization Methods
  // Initialize Optimizer: Sets up variables for Adam optimizer
  #initializeOptimizer() {
    if (!this.weight_m) {
      this.weight_m = this.weights.map(layer => layer.map(row => row.map(() => 0)));
      this.weight_v = this.weights.map(layer => layer.map(row => row.map(() => 0)));
      this.bias_m = this.biases.map(layer => layer.map(() => 0));
      this.bias_v = this.biases.map(layer => layer.map(() => 0));
    }
  }
  // Update Weights: Applies the chosen optimizer (Adam or SGD) to update weights
  #updateWeights(layerIndex, weightGradients, biasGradients, optimizer, params) {
    const layer = this.layers[layerIndex];
    if (layer.frozen) return;
    const regularizedWeightGradients = weightGradients.map((neuronGradients, i) => neuronGradients.map((gradient, j) => {
      let regularizationGradient = 0;
      if (layer.l1 > 0) {
        regularizationGradient += layer.l1 * Math.sign(this.weights[layerIndex][i][j]);
      }
      if (layer.l2 > 0) {
        regularizationGradient += layer.l2 * this.weights[layerIndex][i][j];
      }
      return gradient + regularizationGradient;
    }));
    if (optimizer === 'adam') {
      this.#adamUpdate(layerIndex, regularizedWeightGradients, biasGradients, params);
    } else {
      this.#sgdUpdate(layerIndex, regularizedWeightGradients, biasGradients, params.learningRate);
    }
  }
  // Adam Update: Updates weights using the Adam optimization algorithm
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
  #initializeMomentum() {
    if (!this.weight_velocity) {
      // Match weights structure
      this.weight_velocity = this.weights.map(layer => layer.map(row => row.map(() => 0)));
      // Match biases structure
      this.bias_velocity = this.biases.map(layer => layer.map(() => 0));
    }
  }
  // SGD Update: Updates weights using Stochastic Gradient Descent (SGD)
  #sgdUpdate(layerIndex, weightGradients, biasGradients, learningRate, momentum = 0.9) {
    for (let j = 0; j < this.weights[layerIndex].length; j++) {
      for (let k = 0; k < this.weights[layerIndex][j].length; k++) {
        // Update velocity first
        this.weight_velocity[layerIndex][j][k] = momentum * this.weight_velocity[layerIndex][j][k] - learningRate * weightGradients[j][k];
        // Update weights using velocity
        this.weights[layerIndex][j][k] += this.weight_velocity[layerIndex][j][k];
      }
      // Same for biases
      this.bias_velocity[layerIndex][j] = momentum * this.bias_velocity[layerIndex][j] - learningRate * biasGradients[j];
      this.biases[layerIndex][j] += this.bias_velocity[layerIndex][j];
    }
  }
  // 🏋️ Training
  async finetune(trainSet, options = {}) {
    const {
      freezeLayers = [], // Array of indices to freeze, empty means auto-freeze all except last two layers
        ...trainOptions // Rest of training options passed to train()
    } = options;
    // If no specific layers provided, freeze all except last two layers
    if (freezeLayers.length === 0) {
      for (let i = 0; i < this.layers.length - 2; i++) {
        this.layers[i].frozen = true;
      }
    } else {
      // Freeze specific layers
      freezeLayers.forEach(index => {
        if (index >= 0 && index < this.layers.length) {
          this.layers[index].frozen = true;
        }
      });
    }
    // Modify the weight update logic in training
    const result = await this.train(trainSet, trainOptions);
    // Unfreeze all layers after training
    this.layers.forEach(layer => layer.frozen = false);
    return result;
  }
  // Train: Trains the model on a dataset
  async train(trainSet, options = {}) {
    if (!('debug' in this)) {
      this.debug = true;
    }
    const {
      epochs = 10, learningRate = 0.212, printEveryEpochs = 1, earlyStopThreshold = 1e-6, testSet = null, callback = null, optimizer = "sgd", lossFunction = "mse", decaySteps = 0, decayRate = 0.5,
    } = options;
    let currentLearningRate = learningRate;
    if (typeof trainSet[0].output === "string" || (Array.isArray(trainSet[0].output) && typeof trainSet[0].output[0] === "string")) {
      trainSet = this.#preprocesstags(trainSet);
    }
    const start = Date.now();
    let t = 0;
    if (optimizer === "adam") {
      this.#initializeOptimizer();
    } else {
      this.#initializeMomentum();
    }
    let lastTrainLoss = 0;
    let lastTestLoss = null;
    for (let epoch = 0; epoch < epochs; epoch++) {
      if (decaySteps > 0 && epoch > 0 && epoch % decaySteps === 0) {
        currentLearningRate *= decayRate;
      }
      let trainError = 0;
      let maxGradientNorm = 0; // Track maximum gradient norm for this epoch
      for (const data of trainSet) {
        t++;
        const {
          layerInputs,
          layerRawOutputs,
          dropoutMasks
        } = this.#forwardPropagate(data.input, true);
        const layerErrors = this.#backPropagate(layerInputs, layerRawOutputs, data.output, lossFunction, dropoutMasks);
        // Get single gradient norm value
        const gradientNorm = this.#calculateGlobalGradientNorm(layerErrors, layerInputs);
        maxGradientNorm = Math.max(maxGradientNorm, gradientNorm);
        for (let i = 0; i < this.weights.length; i++) {
          const weightGradients = this.weights[i].map((_, j) => this.weights[i][j].map((_, k) => {
            let gradient = layerErrors[i][j] * layerInputs[i][k];
            if (dropoutMasks[i]) {
              gradient *= dropoutMasks[i][k];
            }
            return gradient;
          }));
          const biasGradients = layerErrors[i];
          this.#updateWeights(i, weightGradients, biasGradients, optimizer, {
            t,
            learningRate: currentLearningRate
          });
        }
        trainError += this.#lossFunctions[lossFunction].loss(layerInputs[layerInputs.length - 1], data.output);
        // Add regularization loss
        trainError += this.#calculateRegularizationLoss();
      }
      lastTrainLoss = trainError / trainSet.length;
      if (testSet) {
        lastTestLoss = this.#evaluateTestSet(testSet, lossFunction);
      }
      if ((epoch + 1) % printEveryEpochs === 0 && this.debug) {
        console.log(`epoch ${epoch + 1}, loss: ${lastTrainLoss.toFixed(6)}${
          testSet ? `, loss (test): ${lastTestLoss.toFixed(6)}` : ""
        }, lr: ${currentLearningRate.toFixed(6)}, gradients: ${maxGradientNorm.toFixed(6)}`);
        // Optional: Add early stopping for gradient explosion
        if (maxGradientNorm > 49) { // Threshold value you can adjust
          console.log('⚠️ Gradient explosion detected!');
        }
        if (maxGradientNorm < 0.02) { // Threshold value you can adjust
          console.log('⚠️ Gradient vanishing detected!');
        }
      }
      if (callback) {
        await callback(epoch + 1, lastTrainLoss, lastTestLoss);
      }
      await new Promise(resolve => setTimeout(resolve, 0));
      if (lastTrainLoss < earlyStopThreshold) {
        if (this.debug) {
          console.log(`🚀 Early stopping at epoch ${epoch + 1} with train loss: ${lastTrainLoss.toFixed(6)}${
            testSet ? ` and test loss: ${lastTestLoss.toFixed(6)}` : ""
          }`);
        }
        break;
      }
    }
    if (optimizer === 'adam') {
      delete this.weight_m;
      delete this.weight_v;
      delete this.bias_m;
      delete this.bias_v;
    } else {
      delete this.weight_velocity;
      delete this.bias_velocity;
    }
    const summary = this.#generateTrainingSummary(start, Date.now(), {
      epochs,
      learningRate,
      lastTrainLoss,
      lastTestLoss,
      finalLearningRate: currentLearningRate,
    });
    this.details = summary;
    return summary;
  }
  // Preprocess Tags: Converts categorical outputs to one-hot encoded vectors
  #preprocesstags(trainSet) {
    // Initialize tags property only when needed for classification
    const uniquetags = Array.from(new Set(trainSet.map(item => Array.isArray(item.output) ? item.output : [item.output]).flat()));
    // Set tags property only when preprocessing tags
    this.tags = uniquetags;
    // Automatically add layers if none exist
    if (this.layers.length === 0) {
      const numInputs = trainSet[0].input.length;
      const numClasses = uniquetags.length;
      this.layer(numInputs, Math.ceil((numInputs + numClasses) / 2), "tanh");
      this.layer(Math.ceil((numInputs + numClasses) / 2), numClasses, "softmax");
    }
    // Convert outputs to one-hot encoded vectors
    return trainSet.map(item => ({
      input: item.input,
      output: uniquetags.map(tag => (Array.isArray(item.output) ? item.output : [item.output]).includes(tag) ? 1 : 0)
    }));
  }
  // Evaluate Test Set: Calculates the loss on the test set
  #evaluateTestSet(testSet, lossFunction) {
    return testSet.reduce((error, data) => {
      const prediction = this.predict(data.input, false);
      return error + this.#lossFunctions[lossFunction].loss(prediction, data.output);
    }, 0) / testSet.length;
  }
  // Generate Training Summary: Creates a summary of the training process
  #generateTrainingSummary(start, end, {
    epochs,
    learningRate,
    lastTrainLoss,
    lastTestLoss
  }) {
    const totalParams = this.weights.reduce((sum, layer, i) => sum + layer.flat().length + this.biases[i].length, 0);
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
  // Predict: Makes predictions using the trained model
  predict(input, tags = true) {
    const {
      layerInputs
    } = this.#forwardPropagate(input, false);
    const output = layerInputs[layerInputs.length - 1];
    if (this.tags && this.layers[this.layers.length - 1].activation === "softmax" && tags) {
      return output.map((prob, idx) => ({
        tag: this.tags[idx],
        probability: prob,
      })).sort((a, b) => b.probability - a.probability);
    }
    return output;
  }
  // ℹ️ Info: Updates model metadata (e.g., author, license, etc.)
  info(infoUpdates) {
    this.details.info = infoUpdates;
  }
}
