/* 
 *  this is a complete example of an standard feed-forward neural network and ideally, in the future, this self-contained mdoel is 
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
  async save(name = "model") {
    if (!this.details.info) {
      this.details.info = {
        name: name,
        author: '',
        license: 'MIT',
        note: '',
        date: new Date()
          .toISOString()
      };
    }

    // If no custom name is set, use the save parameter
    if (this.details.info.name === 'Untitled Model') {
      this.details.info.name = name;
    }

    const modelData = {
      weights: this.weights,
      biases: this.biases,
      layers: this.layers,
      details: this.details,
      ...(this.tags && {
        tags: this.tags
      })
    };

    const blob = new Blob([JSON.stringify(modelData)], {
      type: "application/json"
    });
    const downloadUrl = URL.createObjectURL(blob);

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
      reader.readAsText(file);
    });

    try {
      const input = createFileInput();
      document.body.appendChild(input);

      const [file] = await new Promise(resolve => {
        input.onchange = e => resolve(e.target.files);
        input.click();
      });

      if (!file) return;

      const content = await readFile(file);
      const modelData = JSON.parse(content);

      // Load core properties
      this.weights = modelData.weights;
      this.biases = modelData.biases;
      this.layers = modelData.layers;

      // Ensure details and info exist
      this.details = modelData.details || {
        info: {}
      };

      if (modelData.tags)
        this.tags = modelData.tags;

      this.debug && console.log("✅ Model loaded successfully!");
      callback?.();
    } catch (error) {
      this.debug && console.error("❌ Failed to load model:", error);
    } finally {
      delete this.debug
      document.querySelector('input[type="file"]')
        ?.remove();
    }
  }

  info(infoUpdates) {
    this.details.info = infoUpdates
  }
}
