/**
 * Class representing a neural network model.
 * Designed to be beginner-friendly with helpful comments and fun emojis.
 */
class carbono {
  /**
   * Constructor for the model.
   * @param {boolean} debug - Enable debug logging.
   * @param {number} seed - Seed for random number generation.
   */
  constructor(debug = true, seed = 1234) {
    this.debug = debug;
    // Set a private seed value using a clean, dedicated field
    this.#seed = seed;
    this.layers = []; // Holds layer definitions 📚
    this.weights = []; // Weights for each layer 🔢
    this.biases = []; // Biases for each layer ➕
    this.details = {}; // Stores training details and summary 📝
    this.quantized = false;
    this.tags = null; // Tag names for classification tasks
  }
  // ==========================
  // PRIVATE PROPERTIES & METHODS
  // ==========================
  // Private seed value for PRNG
  #seed;
  /**
   * A clean PRNG method using a mulberry32-like algorithm.
   * @returns {number} A pseudo-random number between 0 and 1.
   */
  #rand() {
    // Mulberry32 PRNG algorithm
    this.#seed |= 0; // Ensure integer
    this.#seed = (this.#seed + 0x6D2B79F5) | 0;
    let t = Math.imul(this.#seed ^ (this.#seed >>> 15), 1 | this.#seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  /**
   * Initialize a weight for a given layer based on activation.
   * @param {number} inputSize - Number of inputs.
   * @param {number} outputSize - Number of neurons.
   * @param {string} activation - Activation function name.
   * @returns {number} A new weight value.
   */
  #initW(inputSize, outputSize, activation) {
    switch (activation) {
      case 'relu':
        return this.#rand() * Math.sqrt(2 / inputSize);
      case 'tanh':
      case 'sigmoid':
        return (this.#rand() - 0.5) * Math.sqrt(2 / inputSize);
      case 'softmax':
        return (this.#rand() - 0.5) * Math.sqrt(1 / inputSize);
      default:
        throw new Error(`Unknown activation: ${activation}`);
    }
  }
  /**
   * Calculate the global gradient norm from layer errors.
   * @param {Array} layerErrors - Errors per layer.
   * @param {Array} layerInputs - Inputs per layer.
   * @returns {number} Global gradient norm.
   */
  #gradNorm(layerErrors, layerInputs) {
    let sumSq = 0;
    for (let i = 0; i < this.weights.length; i++) {
      for (let j = 0; j < this.weights[i].length; j++) {
        for (let k = 0; k < this.weights[i][j].length; k++) {
          const grad = layerErrors[i][j] * layerInputs[i][k];
          sumSq += grad * grad;
        }
      }
    }
    return Math.sqrt(sumSq);
  }
  /**
   * Clip value to avoid numerical issues.
   * @param {number} value - The value to clip.
   * @param {number} min - Minimum allowed value.
   * @param {number} max - Maximum allowed value.
   * @returns {number} Clipped value.
   */
  #clip(value, min = 1e-15, max = 1 - 1e-15) {
    return Math.max(Math.min(value, max), min);
  }
  /**
   * Activation functions and their derivatives.
   */
  #actFns = {
    tanh: {
      fn: x => Math.tanh(x),
      deriv: x => 1 - Math.pow(Math.tanh(x), 2)
    },
    sigmoid: {
      fn: x => 1 / (1 + Math.exp(-x)),
      deriv: x => {
        const sig = 1 / (1 + Math.exp(-x));
        return sig * (1 - sig);
      }
    },
    relu: {
      fn: x => Math.max(0, x),
      deriv: x => (x > 0 ? 1 : 0)
    },
    softmax: {
      fn: x => {
        const arr = Array.isArray(x) ? x : [x];
        const expArr = arr.map(val => Math.exp(val));
        const sumExp = expArr.reduce((a, b) => a + b, 0);
        return expArr.map(exp => exp / sumExp);
      },
      deriv: null
    }
  };
  /**
   * Loss functions and their derivatives.
   */
  #lossFns = {
    mse: {
      loss: (pred, act) => pred.reduce((sum, p, i) => sum + Math.pow(p - act[i], 2), 0),
      deriv: (pred, act, actFn) => pred.map((p, i) => (p - act[i]) * (actFn === 'softmax' ? 1 : this.#getActDeriv(p, actFn)))
    },
    'cross-entropy': {
      loss: (pred, act) => -act.reduce((sum, target, i) => sum + target * Math.log(this.#clip(pred[i])), 0),
      deriv: (pred, act) => pred.map((p, i) => p - act[i])
    }
  };
  /**
   * Get activation function output.
   * @param {number|Array} x - Input value(s).
   * @param {string} name - Activation function name.
   * @returns {number|Array} Activated output.
   */
  #getAct(x, name) {
    return this.#actFns[name].fn(x);
  }
  /**
   * Get activation derivative for a given value.
   * @param {number} x - Input value.
   * @param {string} name - Activation function name.
   * @returns {number|null} Derivative value or null if not defined.
   */
  #getActDeriv(x, name) {
    return this.#actFns[name].deriv ? this.#actFns[name].deriv(x) : null;
  }
  /**
   * Create dropout mask for a given size and dropout rate.
   * @param {number} size - Number of neurons.
   * @param {number} rate - Dropout rate (0 to 1).
   * @returns {Array} Dropout mask.
   */
  #dropMask(size, rate) {
    return Array(size).fill().map(() => (this.#rand() > rate ? 1 / (1 - rate) : 0));
  }
  /**
   * Calculate regularization loss for all layers.
   * @returns {number} Regularization loss.
   */
  #regLoss() {
    return this.layers.reduce((total, layer, idx) => {
      if (!layer.l1 && !layer.l2) return total;
      let layerLoss = 0;
      const w = this.weights[idx];
      if (layer.l1 > 0) {
        const l1 = w.reduce((sum, neuron) => sum + neuron.reduce((s, wt) => s + Math.abs(wt), 0), 0);
        layerLoss += layer.l1 * l1;
      }
      if (layer.l2 > 0) {
        const l2 = w.reduce((sum, neuron) => sum + neuron.reduce((s, wt) => s + wt * wt, 0), 0);
        layerLoss += layer.l2 * l2 / 2;
      }
      return total + layerLoss;
    }, 0);
  }
  /**
   * Shuffle an array using the internal PRNG.
   * @param {Array} arr - Array to shuffle.
   * @returns {Array} Shuffled array.
   */
  #shuffle(arr) {
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(this.#rand() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }
  // ==========================
  // PUBLIC HELPER METHODS
  // ==========================
  /**
   * Normalize an array of numbers using min-max scaling.
   * 📊 Useful for data pre-processing!
   * @param {number[]} data - Array of numeric values.
   * @param {number} min - Desired minimum value after scaling.
   * @param {number} max - Desired maximum value after scaling.
   * @returns {number[]} Normalized data array.
   */
  normalize(data, min = 0, max = 1) {
    const dMin = Math.min(...data);
    const dMax = Math.max(...data);
    return data.map(x => ((x - dMin) / (dMax - dMin)) * (max - min) + min);
  }
  /**
   * Split data into training and validation sets automatically.
   * 🤖 If more than 10 samples are provided, the data is shuffled and split.
   * @param {Array} data - Array of data samples (objects with input/output).
   * @param {number} ratio - Fraction for the validation set (default 0.2).
   * @returns {Object} { train: [...], valid: [...] }
   */
  split(data, ratio = 0.2) {
    if (data.length <= 10) {
      if (this.debug) console.log("ℹ️ Less than or equal to 10 samples; no split performed.");
      return {
        train: data,
        valid: []
      };
    }
    const shuffled = this.#shuffle(data);
    const validCount = Math.floor(shuffled.length * ratio);
    const valid = shuffled.slice(0, validCount);
    const train = shuffled.slice(validCount);
    if (this.debug) console.log(`🔀 Data split: ${train.length} train samples, ${valid.length} validation samples.`);
    return {
      train,
      valid
    };
  }
  // ==========================
  // MODEL BUILDING METHODS
  // ==========================
  /**
   * Add a new layer to the network.
   * 🧱 Each layer is defined by input/output sizes and an activation.
   * @param {number} inSize - Number of inputs.
   * @param {number} outSize - Number of neurons.
   * @param {string} act - Activation function (default "tanh").
   * @param {Object} opts - Additional options (dropout, l1, l2).
   * @returns {carbono} The model instance.
   */
  layer(inSize, outSize, act = "tanh", opts = {}) {
    const {
      dropoutRate = 0, l1 = 0, l2 = 0
    } = opts;
    // Ensure input size matches previous layer's output
    if (this.weights.length > 0) {
      const lastOut = this.layers[this.layers.length - 1].outSize;
      if (inSize !== lastOut) throw new Error("Layer input size must match previous layer output size.");
    }
    this.layers.push({
      inSize,
      outSize,
      act,
      dropoutRate,
      l1,
      l2
    });
    // Create weights matrix and biases vector
    const w = Array(outSize).fill().map(() => Array(inSize).fill().map(() => this.#initW(inSize, outSize, act)));
    this.weights.push(w);
    this.biases.push(Array(outSize).fill(0.01));
    return this;
  }
  /**
   * Save specific layers to .uai files
   * @param {string} name - Base filename
   * @param {Array} layerIndices - Indices of layers to save
   * @returns {Promise<boolean>} Whether saving was successful
   */
  async saveLayers(name = "layer", layerIndices = [-1]) {
    layerIndices = layerIndices.map(idx => idx === -1 ? this.layers.length - 1 : idx);
    try {
      for (const idx of layerIndices) {
        if (idx < 0 || idx >= this.layers.length) {
          throw new Error(`Invalid layer index: ${idx}`);
        }
        const layerMeta = {
          type: "layer",
          index: idx,
          layer: this.layers[idx],
          quantized: this.quantized,
          weights: this.weights[idx],
          biases: this.biases[idx],
          ...(this.quantized ? {
            quants: this.quants
          } : {})
        };
        const metaStr = JSON.stringify(layerMeta);
        const metaBytes = new TextEncoder().encode(metaStr);
        const pad = (8 - (metaBytes.length % 8)) % 8;
        const totalW = this.weights[idx].reduce((sum, neuron) => sum + neuron.length, 0);
        const totalB = this.biases[idx].length;
        const header = new Uint32Array([metaBytes.length, pad, totalW, totalB]);
        const totalSize = header.byteLength + metaBytes.length + pad + (this.quantized ? (totalW + totalB) : (totalW + totalB) * 8);
        const buffer = new ArrayBuffer(totalSize);
        const view = new Uint8Array(buffer);
        let offset = 0;
        view.set(new Uint8Array(header.buffer), offset);
        offset += header.byteLength;
        view.set(metaBytes, offset);
        offset += metaBytes.length + pad;
        // Save weights
        if (this.quantized) {
          for (const neuron of this.weights[idx]) {
            view.set(neuron, offset);
            offset += neuron.length;
          }
        } else {
          const floatView = new Float64Array(buffer, offset);
          let floatOffset = 0;
          for (const neuron of this.weights[idx]) {
            floatView.set(neuron, floatOffset);
            floatOffset += neuron.length;
          }
          offset += totalW * 8;
        }
        // Save biases
        if (this.quantized) {
          view.set(this.biases[idx], offset);
        } else {
          const floatView = new Float64Array(buffer, offset);
          floatView.set(this.biases[idx]);
        }
        const blob = new Blob([buffer], {
          type: "application/octet-stream"
        });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `${name}_layer${idx}.uai`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
      }
      if (this.debug) console.log("💾 Layers saved successfully");
      return true;
    } catch (err) {
      console.error("Layer save failed:", err);
      throw err;
    }
  }
  /**
   * Load a layer from .uai file
   * @param {number} targetIndex - Index where to load the layer
   * @returns {Promise<boolean>} Whether loading was successful
   */
  async loadLayer() {
    try {
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = '.uai';
      const [file] = await new Promise(resolve => {
        input.onchange = e => resolve(e.target.files);
        document.body.appendChild(input);
        input.click();
        document.body.removeChild(input);
      });
      if (!file) throw new Error("No file selected");
      const buf = await file.arrayBuffer();
      const header = new Uint32Array(buf, 0, 4);
      const [metaLen, pad, totalW, totalB] = header;
      const metaOffset = header.byteLength;
      const metaBytes = new Uint8Array(buf, metaOffset, metaLen);
      const meta = JSON.parse(new TextDecoder().decode(metaBytes));
      if (meta.type !== "layer") {
        throw new Error("Invalid file type - expected layer file");
      }
      const targetIndex = meta.index;
      if (targetIndex >= this.layers.length) {
        throw new Error("Layer index out of bounds");
      }
      // Verify layer compatibility
      if (meta.layer.inSize !== this.layers[targetIndex].inSize || meta.layer.outSize !== this.layers[targetIndex].outSize) {
        throw new Error("Layer dimensions mismatch");
      }
      let offset = metaOffset + metaLen + pad;
      // Load weights
      if (meta.quantized) {
        this.weights[targetIndex] = [];
        for (let j = 0; j < meta.layer.outSize; j++) {
          const neuron = new Int8Array(buf, offset, meta.layer.inSize);
          this.weights[targetIndex].push(new Int8Array(neuron));
          offset += meta.layer.inSize;
        }
      } else {
        this.weights[targetIndex] = [];
        for (let j = 0; j < meta.layer.outSize; j++) {
          const neuron = new Float64Array(buf, offset, meta.layer.inSize);
          this.weights[targetIndex].push(Array.from(neuron));
          offset += meta.layer.inSize * 8;
        }
      }
      // Load biases
      if (meta.quantized) {
        const bias = new Int8Array(buf, offset, meta.layer.outSize);
        this.biases[targetIndex] = new Int8Array(bias);
      } else {
        const bias = new Float64Array(buf, offset, meta.layer.outSize);
        this.biases[targetIndex] = Array.from(bias);
      }
      // Update layer properties
      this.layers[targetIndex] = {
        ...this.layers[targetIndex],
        ...meta.layer
      };
      if (this.debug) console.log(`📂 Layer ${targetIndex} loaded successfully`);
      return true;
    } catch (err) {
      console.error("Layer load failed:", err);
      throw err;
    }
  }
  /**
   * Insert a new layer at a given index.
   * 🔀 Useful for modifying the architecture mid-stream.
   * @param {number} idx - Index at which to insert the new layer.
   * @param {string} act - Activation function (default "tanh").
   * @returns {carbono} The model instance.
   */
  insert(idx, act = "tanh") {
    if (idx < 0 || idx > this.layers.length) throw new Error("Invalid layer index");
    const inSize = idx === 0 ? this.layers[0].inSize : this.layers[idx - 1].outSize;
    const outSize = idx === this.layers.length ? this.layers[this.layers.length - 1].outSize : this.layers[idx].inSize;
    const newLayer = {
      inSize,
      outSize,
      act,
      dropoutRate: 0,
      l1: 0,
      l2: 0
    };
    this.layers.splice(idx, 0, newLayer);
    const w = Array(outSize).fill().map(() => Array(inSize).fill().map(() => this.#initW(inSize, outSize, act)));
    this.weights.splice(idx, 0, w);
    this.biases.splice(idx, 0, Array(outSize).fill(0.01));
    return this;
  }
  /**
   * Replace an existing layer with a new one.
   * 🔄 Useful when you need to change dimensions or activation.
   * @param {number} idx - Index of the layer to replace.
   * @param {number} inSize - New input size.
   * @param {number} outSize - New output size.
   * @param {string} act - Activation function (default "tanh").
   * @returns {carbono} The model instance.
   */
  replace(idx, inSize, outSize, act = "tanh") {
    if (idx < 0 || idx >= this.layers.length) throw new Error("Invalid layer index");
    const newLayer = {
      inSize,
      outSize,
      act,
      dropoutRate: 0,
      l1: 0,
      l2: 0
    };
    this.layers[idx] = newLayer;
    const w = Array(outSize).fill().map(() => Array(inSize).fill().map(() => this.#initW(inSize, outSize, act)));
    this.weights[idx] = w;
    this.biases[idx] = Array(outSize).fill(0.01);
    // Adjust previous and next layers if necessary
    if (idx > 0) {
      this.layers[idx - 1].outSize = inSize;
      this.weights[idx - 1] = Array(inSize).fill().map(() => Array(this.layers[idx - 1].inSize).fill().map(() => this.#initW(this.layers[idx - 1].inSize, inSize, this.layers[idx].act)));
      this.biases[idx - 1] = Array(inSize).fill(0.01);
    }
    if (idx < this.layers.length - 1) {
      this.layers[idx + 1].inSize = outSize;
      this.weights[idx + 1] = Array(this.layers[idx + 1].outSize).fill().map(() => Array(outSize).fill().map(() => this.#initW(outSize, this.layers[idx + 1].outSize, this.layers[idx + 1].act)));
    }
    return this;
  }
  /**
   * Remove a layer at a given index.
   * 🗑️ Use with caution!
   * @param {number} idx - Index of the layer to remove.
   * @returns {carbono} The model instance.
   */
  drop(idx) {
    if (idx < 0 || idx >= this.layers.length) throw new Error("Invalid layer index");
    this.layers.splice(idx, 1);
    this.weights.splice(idx, 1);
    this.biases.splice(idx, 1);
    // Update next layer's input size if needed
    if (idx < this.layers.length) {
      const prevOut = idx > 0 ? this.layers[idx - 1].outSize : this.layers[0].inSize;
      const next = this.layers[idx];
      next.inSize = prevOut;
      this.weights[idx] = Array(next.outSize).fill().map(() => Array(prevOut).fill().map(() => this.#initW(prevOut, next.outSize, next.act)));
    }
    return this;
  }
  // ==========================
  // FORWARD & BACK PROP
  // ==========================
  /**
   * Forward propagate input through the network.
   * ⚡ Handles dropout and activation functions.
   * @param {Array} input - Input data array.
   * @param {boolean} isTrain - Whether in training mode.
   * @returns {Object} Contains layer inputs, raw outputs, and dropout masks.
   */
  #forward(input, isTrain = true) {
    let curr = Array.from(input);
    const ins = [curr]; // Store inputs for each layer
    const raws = []; // Raw outputs (before activation)
    const masks = []; // Dropout masks per layer
    for (let i = 0; i < this.weights.length; i++) {
      const raw = [];
      const w = this.weights[i];
      const b = this.biases[i];
      const layer = this.layers[i];
      // Apply dropout if training and dropoutRate > 0
      if (isTrain && layer.dropoutRate > 0) {
        const mask = this.#dropMask(curr.length, layer.dropoutRate);
        masks.push(mask);
        curr = curr.map((v, idx) => v * mask[idx]);
      } else {
        masks.push(null);
      }
      // Compute raw output for each neuron
      for (let j = 0; j < w.length; j++) {
        let sum = 0;
        for (let k = 0; k < curr.length; k++) {
          sum += curr[k] * w[j][k];
        }
        // Add bias (handle quantization if needed)
        sum += b[j];
        raw.push(sum);
      }
      raws.push(raw);
      // Activation step
      curr = layer.act === 'softmax' ? this.#getAct(raw.map(x => Number.isFinite(x) ? x : 0), 'softmax') : raw.map(x => {
        const actVal = this.#getAct(Number.isFinite(x) ? x : 0, layer.act);
        return Number.isFinite(actVal) ? actVal : 0;
      });
      ins.push(curr);
    }
    return {
      ins,
      raws,
      masks
    };
  }
  /**
   * Backward propagate errors and compute gradients.
   * 🔙 Uses the chain rule to calculate weight updates.
   * @param {Array} ins - Inputs collected during forward pass.
   * @param {Array} raws - Raw outputs (pre-activation).
   * @param {Array} target - Target output array.
   * @param {string} lossName - Loss function name.
   * @param {Array} masks - Dropout masks used during forward pass.
   * @returns {Array} Layer errors (gradients).
   */
  #back(ins, raws, target, lossName, masks) {
    const outLayer = this.layers[this.layers.length - 1];
    // Compute error at output using loss derivative
    const outErrors = this.#lossFns[lossName].deriv(ins[ins.length - 1], target, outLayer.act);
    const errs = [outErrors];
    // Backpropagate through hidden layers
    for (let i = this.weights.length - 2; i >= 0; i--) {
      const err = Array(this.layers[i].outSize).fill(0);
      for (let j = 0; j < this.layers[i].outSize; j++) {
        for (let k = 0; k < this.layers[i + 1].outSize; k++) {
          err[j] += errs[0][k] * this.weights[i + 1][k][j];
        }
        const deriv = this.#getActDeriv(raws[i][j], this.layers[i].act);
        if (deriv !== null) err[j] *= deriv;
        if (masks[i]) err[j] *= masks[i][j];
      }
      errs.unshift(err);
    }
    return errs;
  }
  // ==========================
  // WEIGHT UPDATE METHODS
  // ==========================
  /**
   * Update weights and biases for a given layer.
   * ✏️ Handles regularization and delegates to the optimizer.
   * @param {number} idx - Layer index.
   * @param {Array} wGrads - Weight gradients.
   * @param {Array} bGrads - Bias gradients.
   * @param {string} optim - Optimizer ("adam" or "sgd").
   * @param {Object} params - Optimizer parameters.
   */
  #updW(idx, wGrads, bGrads, optim, params) {
    if (this.layers[idx].frozen) return;
    const layer = this.layers[idx];
    // Apply L1/L2 regularization if set
    if (layer.l1 > 0 || layer.l2 > 0) {
      wGrads = wGrads.map((neuron, j) => neuron.map((grad, k) => {
        let reg = 0;
        if (layer.l1 > 0) reg += layer.l1 * Math.sign(this.weights[idx][j][k]);
        if (layer.l2 > 0) reg += layer.l2 * this.weights[idx][j][k];
        return grad + reg;
      }));
    }
    if (optim === "adam") {
      this.#adam(idx, wGrads, bGrads, params);
    } else {
      this.#sgd(idx, wGrads, bGrads, params.learningRate);
    }
  }
  /**
   * Initialize optimizer state for Adam if needed.
   */
  #initOpt() {
    if (!this.weight_m) {
      this.weight_m = this.weights.map(layer => layer.map(row => row.map(() => 0)));
      this.weight_v = this.weights.map(layer => layer.map(row => row.map(() => 0)));
      this.bias_m = this.biases.map(layer => layer.map(() => 0));
      this.bias_v = this.biases.map(layer => layer.map(() => 0));
    }
  }
  /**
   * Adam optimizer update.
   * ⚙️ Adaptive Moment Estimation.
   * @param {number} idx - Layer index.
   * @param {Array} wGrads - Weight gradients.
   * @param {Array} bGrads - Bias gradients.
   * @param {Object} params - Contains timestep (t) and learningRate.
   */
  #adam(idx, wGrads, bGrads, {
    t,
    learningRate
  }) {
    const beta1 = 0.9,
      beta2 = 0.999,
      epsilon = 1e-8;
    for (let j = 0; j < this.weights[idx].length; j++) {
      for (let k = 0; k < this.weights[idx][j].length; k++) {
        const g = wGrads[j][k];
        this.weight_m[idx][j][k] = beta1 * this.weight_m[idx][j][k] + (1 - beta1) * g;
        this.weight_v[idx][j][k] = beta2 * this.weight_v[idx][j][k] + (1 - beta2) * g * g;
        const m_hat = this.weight_m[idx][j][k] / (1 - Math.pow(beta1, t));
        const v_hat = this.weight_v[idx][j][k] / (1 - Math.pow(beta2, t));
        this.weights[idx][j][k] -= (learningRate * m_hat) / (Math.sqrt(v_hat) + epsilon);
      }
      const g_bias = bGrads[j];
      this.bias_m[idx][j] = beta1 * this.bias_m[idx][j] + (1 - beta1) * g_bias;
      this.bias_v[idx][j] = beta2 * this.bias_v[idx][j] + (1 - beta2) * g_bias * g_bias;
      const m_hat_bias = this.bias_m[idx][j] / (1 - Math.pow(beta1, t));
      const v_hat_bias = this.bias_v[idx][j] / (1 - Math.pow(beta2, t));
      this.biases[idx][j] -= (learningRate * m_hat_bias) / (Math.sqrt(v_hat_bias) + epsilon);
    }
  }
  /**
   * SGD optimizer update.
   * 🚀 (alias: sgd) Standard gradient descent with momentum.
   * @param {number} idx - Layer index.
   * @param {Array} wGrads - Weight gradients.
   * @param {Array} bGrads - Bias gradients.
   * @param {number} learningRate - Learning rate.
   * @param {number} momentum - Momentum factor (default 0.9).
   */
  #sgd(idx, wGrads, bGrads, learningRate, momentum = 0.9) {
    for (let j = 0; j < this.weights[idx].length; j++) {
      for (let k = 0; k < this.weights[idx][j].length; k++) {
        this.weight_velocity[idx][j][k] = momentum * this.weight_velocity[idx][j][k] - learningRate * wGrads[j][k];
        this.weights[idx][j][k] += this.weight_velocity[idx][j][k];
      }
      this.bias_velocity[idx][j] = momentum * this.bias_velocity[idx][j] - learningRate * bGrads[j];
      this.biases[idx][j] += this.bias_velocity[idx][j];
    }
  }
  /**
   * Initialize momentum variables for SGD if needed.
   */
  #initMom() {
    if (!this.weight_velocity) {
      this.weight_velocity = this.weights.map(layer => layer.map(row => row.map(() => 0)));
      this.bias_velocity = this.biases.map(layer => layer.map(() => 0));
    }
  }
  // ==========================
  // TRAIN & PREDICT METHODS
  // ==========================
  /**
   * Train the model on the provided dataset.
   * 🔥 If the dataset is a URL, it fetches the JSON data.
   * If the dataset contains string outputs, it will auto-preprocess tags.
   * Now prints info for every epoch if debug mode is enabled.
   * @param {Array|string} trainSet - Training dataset or URL to fetch it.
   * @param {Object} opts - Training options.
   * @returns {Object} Training summary.
   */
  async train(trainSet, opts = {}) {
    // If trainSet is a URL, fetch data from network
    if (typeof trainSet === "string" && trainSet.startsWith("http")) {
      try {
        const res = await fetch(trainSet);
        if (!res.ok) throw new Error(`Failed to fetch data from ${trainSet}`);
        trainSet = await res.json();
      } catch (err) {
        console.error("Error fetching training data:", err);
        throw err;
      }
    }
    if (this.debug) {
      const frozenLayers = this.layers.map((layer, idx) => ({
        idx,
        frozen: layer.frozen
      })).filter(l => l.frozen);
      if (frozenLayers.length > 0) {
        console.log("🧊 Training with frozen layers:", frozenLayers.map(l => l.idx).join(", "));
      }
    }
    // Auto-preprocess tags if needed
    if (typeof trainSet[0].output === "string" || (Array.isArray(trainSet[0].output) && typeof trainSet[0].output[0] === "string")) {
      trainSet = this.#prepTags(trainSet);
    }
    // Automatically split dataset if more than 10 samples and no testSet provided
    let testSet = opts.testSet;
    if (!testSet && trainSet.length > 10) {
      const splitData = this.split(trainSet, 0.2);
      trainSet = splitData.train;
      testSet = splitData.valid;
    }
    // Extract training options with defaults
    const {
      epochs = 10,
        learningRate = 0.212,
        optim = "sgd",
        loss = "mse",
        decaySteps = 0,
        decayRate = 0.5,
        callback = null,
        every = 10,
        printEveryEpochs = 1, // Default to printing every epoch
        earlyStop = 1e-10
    } = opts;
    let curLR = learningRate;
    const startTime = Date.now();
    let t = 0;
    if (optim === "adam") {
      this.#initOpt();
    } else {
      this.#initMom();
    }
    let lastLoss = 0,
      testLoss = null,
      maxGrad = 0;
    // Training loop over epochs
    for (let epoch = 0; epoch < epochs; epoch++) {
      if (decaySteps > 0 && epoch > 0 && epoch % decaySteps === 0) {
        curLR *= decayRate;
      }
      let epochError = 0;
      maxGrad = 0;
      for (const sample of trainSet) {
        t++;
        // Forward pass
        const {
          ins,
          raws,
          masks
        } = this.#forward(sample.input, true);
        // Backward pass to compute gradients
        const errors = this.#back(ins, raws, sample.output, loss, masks);
        const gradNorm = this.#gradNorm(errors, ins);
        maxGrad = Math.max(maxGrad, gradNorm);
        // Update each layer's weights and biases
        for (let i = 0; i < this.weights.length; i++) {
          const wGrads = this.weights[i].map((_, j) => this.weights[i][j].map((_, k) => {
            let grad = errors[i][j] * ins[i][k];
            if (masks[i]) grad *= masks[i][k];
            return grad;
          }));
          const bGrads = errors[i];
          this.#updW(i, wGrads, bGrads, optim, {
            t,
            learningRate: curLR
          });
        }
        // Compute loss (plus regularization)
        epochError += this.#lossFns[loss].loss(ins[ins.length - 1], sample.output) + this.#regLoss();
      }
      lastLoss = epochError / trainSet.length;
      if (testSet && testSet.length > 0) {
        testLoss = this.#evalTest(testSet, loss);
      }
      // Print training progress at specified intervals
      if (epoch % printEveryEpochs === 0 || epoch === epochs - 1) {
        console.log('\n=== Training Progress ===');
        console.log(`Epoch: ${epoch+1}/${epochs}`);
        console.log(`Training Loss: ${lastLoss.toFixed(6)}`);
        if (testLoss !== null) {
          console.log(`Validation Loss: ${testLoss.toFixed(6)}`);
        }
        console.log(`Learning Rate: ${curLR.toFixed(6)}`);
        console.log(`Max Gradient: ${maxGrad.toFixed(6)}`);
        // Additional warnings for gradient issues
        if (maxGrad > 49) console.log('⚠️ Warning: Gradient explosion detected!');
        if (maxGrad < 0.01) console.log('⚠️ Warning: Gradient vanishing detected!');
        console.log('========================\n');
      }
      if (callback) await callback(epoch + 1, lastLoss, testLoss);
      await new Promise(res => setTimeout(res, 0));
      // Early stopping check
      if (lastLoss < earlyStop) {
        console.log(`🎯 Early stopping achieved at epoch ${epoch + 1} with loss: ${lastLoss.toFixed(6)}`);
        break;
      }
    }
    // Clean up optimizer state
    if (optim === 'adam') {
      delete this.weight_m;
      delete this.weight_v;
      delete this.bias_m;
      delete this.bias_v;
    } else {
      delete this.weight_velocity;
      delete this.bias_velocity;
    }
    const summary = this.#trainSum(startTime, Date.now(), {
      epochs,
      learningRate,
      lastLoss,
      testLoss
    });
    this.details = summary;
    return summary;
  }
  /**
   * Finetune specific layers while freezing others.
   * @param {Array} dataset - Training dataset
   * @param {Array} layers - Indices of layers to train (-1 for last layer only)
   * @param {Object} opts - Training options
   * @returns {Object} Training summary
   */
  async finetune(dataset, layers = [-1], opts = {}) {
    // Convert -1 to last layer index
    layers = layers.map(idx => idx === -1 ? this.layers.length - 1 : idx);
    // Validate layer indices
    layers.forEach(idx => {
      if (idx < 0 || idx >= this.layers.length) {
        throw new Error(`Invalid layer index: ${idx}`);
      }
    });
    // Store original frozen states
    const originalFrozenStates = this.layers.map(layer => layer.frozen || false);
    // Freeze all layers except those being finetuned
    this.layers.forEach((layer, idx) => {
      layer.frozen = !layers.includes(idx);
    });
    // Train with frozen layers
    const summary = await this.train(dataset, opts);
    // Restore original frozen states
    this.layers.forEach((layer, idx) => {
      layer.frozen = originalFrozenStates[idx];
    });
    return summary;
  }
  /**
   * Evaluate the test set performance.
   * @param {Array} testSet - Test dataset.
   * @param {string} loss - Loss function name.
   * @returns {number} Average loss over test set.
   */
  #evalTest(testSet, loss) {
    return testSet.reduce((sum, sample) => {
      const pred = this.predict(sample.input, false);
      return sum + this.#lossFns[loss].loss(pred, sample.output);
    }, 0) / testSet.length;
  }
  /**
   * Generate a training summary.
   * 📝 Includes total parameters, training time, epochs, etc.
   * @param {number} start - Start timestamp.
   * @param {number} end - End timestamp.
   * @param {Object} opts - Training options summary.
   * @returns {Object} Training summary.
   */
  #trainSum(start, end, {
    epochs,
    learningRate,
    lastLoss,
    testLoss
  }) {
    const totalParams = this.weights.reduce((sum, layer, i) => sum + layer.flat().length + this.biases[i].length, 0);
    return {
      parameters: totalParams,
      training: {
        loss: lastLoss,
        testloss: testLoss,
        time: end - start,
        epochs,
        learningRate
      }
    };
  }
  /**
   * Preprocess training set tags for classification tasks.
   * Converts string outputs into one-hot arrays.
   * @param {Array} trainSet - Training dataset.
   * @returns {Array} Processed training set.
   */
  #prepTags(trainSet) {
    const unique = Array.from(new Set(trainSet.map(item => Array.isArray(item.output) ? item.output : [item.output]).flat()));
    this.tags = unique;
    if (this.layers.length === 0) {
      const numInputs = trainSet[0].input.length;
      const numClasses = unique.length;
      // Create two layers by default: one hidden and one output (softmax)
      this.layer(numInputs, Math.ceil((numInputs + numClasses) / 2), "tanh");
      this.layer(Math.ceil((numInputs + numClasses) / 2), numClasses, "softmax");
    }
    return trainSet.map(item => ({
      input: item.input,
      output: unique.map(tag => (Array.isArray(item.output) ? item.output : [item.output]).includes(tag) ? 1 : 0)
    }));
  }
  /**
   * Predict output for a given input.
   * 🤖 Returns probabilities for each tag if available.
   * @param {Array} input - Input data array.
   * @param {boolean} asTags - Whether to return tagged output.
   * @returns {Array} Prediction output.
   */
  predict(input, asTags = true) {
    const {
      ins
    } = this.#forward(input, false);
    const out = ins[ins.length - 1];
    if (this.tags && this.layers[this.layers.length - 1].act === "softmax" && asTags) {
      return out.map((prob, i) => ({
        tag: this.tags[i],
        probability: prob
      })).sort((a, b) => b.probability - a.probability);
    }
    return out;
  }
  /**
   * Attach additional info to the training details.
   * ℹ️ Can be used to store custom metadata.
   * @param {Object} info - Info object.
   */
  info(info) {
    this.details.info = info;
  }
  // ==========================
  // SAVE & LOAD METHODS
  // ==========================
  /**
   * Save the model to file.
   * 💾 Supports "localStorage", "uai" (binary), or "json" formats.
   * @param {string} name - Filename prefix.
   * @param {string} type - Save type ("localStorage", "uai", or "json").
   * @returns {Promise<boolean>} Whether saving was successful.
   */
  async save(name = "model", type = "json") {
    try {
      if (type === "localStorage") {
        const meta = {
          layers: this.layers,
          details: this.details,
          quantized: this.quantized,
          weights: this.weights,
          biases: this.biases,
          tags: this.tags || null,
          ...(this.quantized ? {
            quants: this.quants
          } : {})
        };
        localStorage.setItem(name, JSON.stringify(meta));
        if (this.debug) console.log("💾 Model saved to localStorage under key:", name);
        return true;
      } else if (type === "uai") {
        // Binary format saving (advanced)
        const meta = {
          layers: this.layers,
          details: this.details,
          quantization: this.quantized ? {
            enabled: true,
            quants: this.quants
          } : null,
          tags: this.tags || null
        };
        const metaStr = JSON.stringify(meta);
        const metaBytes = new TextEncoder().encode(metaStr);
        const pad = (8 - (metaBytes.length % 8)) % 8;
        let totalW = 0,
          totalB = 0;
        this.weights.forEach((layer, i) => {
          layer.forEach(neuron => totalW += neuron.length);
          totalB += this.biases[i].length;
        });
        const header = new Uint32Array([metaBytes.length, pad, totalW, totalB]);
        const totalSize = header.byteLength + metaBytes.length + pad + (this.quantized ? (totalW + totalB) : (totalW + totalB) * 8);
        const buffer = new ArrayBuffer(totalSize);
        const view = new Uint8Array(buffer);
        let offset = 0;
        view.set(new Uint8Array(header.buffer), offset);
        offset += header.byteLength;
        view.set(metaBytes, offset);
        offset += metaBytes.length + pad;
        if (offset % 8 !== 0) offset += (8 - (offset % 8));
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
        if (!this.quantized && offset % 8 !== 0) offset += (8 - (offset % 8));
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
        const blob = new Blob([buffer], {
          type: "application/octet-stream"
        });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `${name}.uai`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        if (this.debug) console.log("💾 Model saved in UAI binary format.");
        return true;
      } else {
        // JSON format saving
        const meta = {
          layers: this.layers,
          details: this.details,
          tags: this.tags || null,
          quantized: this.quantized,
          ...(this.quantized ? {
            quantization: {
              enabled: true,
              quants: this.quants,
              weights: this.weights.map(layer => layer.map(neuron => Array.from(neuron))),
              biases: this.biases.map(b => Array.from(b))
            }
          } : {
            weights: this.weights,
            biases: this.biases
          })
        };
        const blob = new Blob([JSON.stringify(meta, null, 2)], {
          type: "application/json"
        });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `${name}.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        if (this.debug) console.log("💾 Model saved in JSON format.");
        return true;
      }
    } catch (err) {
      console.error("Save process failed:", err);
      throw err;
    }
  }
  /**
   * Load a model from file.
   * 📂 Supports both JSON and binary ("uai") formats.
   * @param {Function} callback - Optional callback after loading.
   * @param {boolean} useBin - Whether to use binary loading.
   * @returns {Promise<boolean>} Whether loading was successful.
   */
  async load(callback, useBin = false) {
    try {
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = useBin ? '.uai' : '.json';
      const [file] = await new Promise(resolve => {
        input.onchange = e => resolve(e.target.files);
        document.body.appendChild(input);
        input.click();
        document.body.removeChild(input);
      });
      if (!file) throw new Error("No file selected");
      const buf = await file.arrayBuffer();
      if (useBin) {
        const header = new Uint32Array(buf, 0, 4);
        const [metaLen, pad, totalW, totalB] = header;
        const metaOffset = header.byteLength;
        const metaBytes = new Uint8Array(buf, metaOffset, metaLen);
        const meta = JSON.parse(new TextDecoder().decode(metaBytes));
        let offset = metaOffset + metaLen + pad;
        this.quantized = meta.quantization?.enabled || false;
        this.quants = meta.quantization?.quants || null;
        this.layers = meta.layers;
        this.details = meta.details;
        if (meta.tags) this.tags = meta.tags;
        this.weights = [];
        meta.layers.forEach((layer, i) => {
          const layerW = [];
          for (let j = 0; j < layer.outSize; j++) {
            if (this.quantized) {
              const neuron = new Int8Array(buf, offset, layer.inSize);
              layerW.push(new Int8Array(neuron));
              offset += layer.inSize;
            } else {
              const neuron = new Float64Array(buf, offset, layer.inSize);
              layerW.push(Array.from(neuron));
              offset += layer.inSize * 8;
            }
          }
          this.weights.push(layerW);
        });
        this.biases = [];
        meta.layers.forEach(layer => {
          if (this.quantized) {
            const bias = new Int8Array(buf, offset, layer.outSize);
            this.biases.push(new Int8Array(bias));
            offset += layer.outSize;
          } else {
            const bias = new Float64Array(buf, offset, layer.outSize);
            this.biases.push(Array.from(bias));
            offset += layer.outSize * 8;
          }
        });
      } else {
        const meta = JSON.parse(new TextDecoder().decode(buf));
        this.quantized = meta.quantization?.enabled || false;
        this.quants = meta.quantization?.quants || null;
        this.layers = meta.layers;
        this.details = meta.details;
        if (meta.tags) this.tags = meta.tags;
        if (this.quantized) {
          this.weights = meta.quantization.weights.map(layer => layer.map(neuron => new Int8Array(neuron)));
          this.biases = meta.quantization.biases.map(b => new Int8Array(b));
        } else {
          this.weights = meta.weights;
          this.biases = meta.biases;
        }
      }
      if (callback) callback();
      if (this.debug) console.log("📂 Model loaded successfully.");
      return true;
    } catch (err) {
      console.error("Load process failed:", err);
      throw err;
    }
  }
  /**
   * Reset all weights and biases to initial random values.
   * 🔄 Useful for retraining from scratch.
   * @returns {carbono} The model instance.
   */
  reset() {
    for (let i = 0; i < this.layers.length; i++) {
      const {
        inSize,
        outSize,
        act
      } = this.layers[i];
      this.weights[i] = Array(outSize).fill().map(() => Array(inSize).fill().map(() => this.#initW(inSize, outSize, act)));
      this.biases[i] = Array(outSize).fill(0.01);
    }
    if (this.debug) console.log("🔄 Model weights and biases have been reset.");
    return this;
  }
}
