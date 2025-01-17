// appvoid carbono 8 (beta) 
/* 
 *  this is a complete example of a standard feed-forward neural network and ideally, in the future, this self-contained model is planned to be 
 *  sufficient to handle general use-case scenarios as well as becoming a supportive, general-purpose engine for training as well as 
 *  inference on pytorch; this is not by any means stable yet, even though i'm expecting the tool to be improved over time, you must use it 
 *  at your own risk.
 */

// major changes:
// new drop, insert, replace layer methods
// removed unused matrixMultiply method
// new 8-bit quantization feature
// removed selu for simplicity

class carbono {
    // 🏗️ Constructor: Initializes the neural network with default settings
    constructor(debug = true) {
      this.layers = []; // Stores the layers of the neural network
      this.weights = []; // Stores the weights for each layer
      this.biases = []; // Stores the biases for each layer
      this.details = {}; // Stores metadata about the model (e.g., training details)
    }

    // 🛠️ Utility Methods
  
    // Xavier Initialization: Helps initialize weights to improve training
    #xavier(inputSize, outputSize) {
      return (Math.random() - 0.5) * 2 * Math.sqrt(6 / (inputSize + outputSize));
    }
  
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
        loss: (predicted, actual) =>
          predicted.reduce((sum, pred, i) => sum + Math.pow(pred - actual[i], 2), 0), // Mean Squared Error
        derivative: (predicted, actual, activation) =>
          predicted.map((pred, i) => (pred - actual[i]) *
            (activation === 'softmax' ? 1 : this.#getActivationDerivative(pred, activation))) // Derivative of MSE
      },
      'cross-entropy': {
        loss: (predicted, actual) =>
          -actual.reduce((sum, target, i) =>
            sum + target * Math.log(this.#clip(predicted[i])), 0), // Cross-Entropy Loss
        derivative: (predicted, actual) =>
          predicted.map((pred, i) => pred - actual[i]) // Derivative of Cross-Entropy
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
        const prevLayerOutputSize = layerIndex > 0 ? 
          this.layers[layerIndex - 1].outputSize : 
          this.layers[0].inputSize;
        
        const nextLayer = this.layers[layerIndex];
        nextLayer.inputSize = prevLayerOutputSize;
    
        // Reinitialize weights for the connection
        this.weights[layerIndex] = Array(nextLayer.outputSize)
          .fill()
          .map(() =>
            Array(prevLayerOutputSize)
            .fill()
            .map(() => this.#xavier(prevLayerOutputSize, nextLayer.outputSize))
          );
      }
    
      return this;
    }
    
    // Insert Layer: Adds a new layer at the specified index
    insert(layerIndex, activation = "tanh") {
      if (layerIndex < 0 || layerIndex > this.layers.length) {
        throw new Error("Invalid layer index");
      }
    
      // Calculate input and output sizes based on adjacent layers
      const inputSize = layerIndex === 0 ? 
        this.layers[0].inputSize : 
        this.layers[layerIndex - 1].outputSize;
    
      const outputSize = layerIndex === this.layers.length ? 
        this.layers[this.layers.length - 1].outputSize : 
        this.layers[layerIndex].inputSize;
    
      // Create the new layer
      const newLayer = {
        inputSize,
        outputSize,
        activation
      };
    
      // Insert the layer
      this.layers.splice(layerIndex, 0, newLayer);
    
      // Initialize weights for the new layer
      const weights = Array(outputSize)
        .fill()
        .map(() =>
          Array(inputSize)
          .fill()
          .map(() => this.#xavier(inputSize, outputSize))
        );
      this.weights.splice(layerIndex, 0, weights);
    
      // Initialize biases for the new layer
      this.biases.splice(layerIndex, 0, Array(outputSize).fill(0.01));
    
      return this;
    }
  
    // Replace Layer: Replaces a layer at the specified index with a new layer
    replace(layerIndex, inputSize, outputSize, activation = "tanh") {
      if (layerIndex < 0 || layerIndex >= this.layers.length) {
        throw new Error("Invalid layer index");
      }
    
      // Create the new layer configuration
      const newLayer = {
        inputSize,
        outputSize,
        activation
      };
    
      // Replace the layer
      this.layers[layerIndex] = newLayer;
    
      // Initialize new weights for this layer
      const weights = Array(outputSize)
        .fill()
        .map(() =>
          Array(inputSize)
          .fill()
          .map(() => this.#xavier(inputSize, outputSize))
        );
      this.weights[layerIndex] = weights;
    
      // Initialize new biases
      this.biases[layerIndex] = Array(outputSize).fill(0.01);
    
      // Adjust previous layer if it exists
      if (layerIndex > 0) {
        this.layers[layerIndex - 1].outputSize = inputSize;
        
        // Reinitialize weights for the previous layer
        this.weights[layerIndex - 1] = Array(inputSize)
          .fill()
          .map(() =>
            Array(this.layers[layerIndex - 1].inputSize)
            .fill()
            .map(() => this.#xavier(this.layers[layerIndex - 1].inputSize, inputSize))
          );
        
        // Reinitialize biases for the previous layer
        this.biases[layerIndex - 1] = Array(inputSize).fill(0.01);
      }
    
      // Adjust next layer if it exists
      if (layerIndex < this.layers.length - 1) {
        this.layers[layerIndex + 1].inputSize = outputSize;
        
        // Reinitialize weights for the next layer
        this.weights[layerIndex + 1] = Array(this.layers[layerIndex + 1].outputSize)
          .fill()
          .map(() =>
            Array(outputSize)
            .fill()
            .map(() => this.#xavier(outputSize, this.layers[layerIndex + 1].outputSize))
          );
      }
    
      return this;
    }
    
    
    // Add Layer: Adds a new layer to the neural network
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
  
      // Initialize weights using Xavier initialization
      const weights = Array(outputSize)
        .fill()
        .map(() =>
          Array(inputSize)
          .fill()
          .map(() => this.#xavier(inputSize, outputSize))
        );
      this.weights.push(weights);
  
      // Initialize biases with small values
      this.biases.push(Array(outputSize)
        .fill(0.01));
      return this;
    }

  // Add to carbono class:

// Quantization Methods
quantize(calibrationData = null) {
    // Store original weights for backup
    this._originalWeights = structuredClone(this.weights);
    this._originalBiases = structuredClone(this.biases);
    this.quantizationParams = [];

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
            return new Int8Array(neuron.map(w => 
                Math.min(Math.max(Math.round(w / weightScale), -127), 127)
            ));
        });

        // Quantize biases
        const quantizedBiases = new Int8Array(biases.map(b => 
            Math.min(Math.max(Math.round(b / biasScale), -127), 127)
        ));

        // Store quantized values and scales
        this.weights[layerIdx] = quantizedWeights;
        this.biases[layerIdx] = quantizedBiases;
        this.quantizationParams[layerIdx] = {
            weightScale,
            biasScale
        };
    });

    this._isQuantized = true;
    return this;
}

// Modified forward propagation
#forwardPropagate(input) {
    let current = input;
    const layerInputs = [input];
    const layerRawOutputs = [];

    for (let i = 0; i < this.weights.length; i++) {
        const rawOutput = [];
        const weights = this.weights[i];
        const biases = this.biases[i];
        const params = this._isQuantized ? this.quantizationParams[i] : null;

        for (let j = 0; j < weights.length; j++) {
            let sum = 0;
            for (let k = 0; k < current.length; k++) {
                const w = weights[j][k];
                const actualWeight = params ? w * params.weightScale : w;
                sum += actualWeight * current[k];
            }
            const b = biases[j];
            const actualBias = params ? b * params.biasScale : b;
            rawOutput.push(sum + actualBias);
        }

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

// Modified summary method
summary() {
    let output = "\n";
    
    output += this.layers.map((layer, i) => 
        `[${layer.inputSize}] ─ ${layer.activation} → ${i === this.layers.length - 1 ? `[${layer.outputSize}]` : ''}`
    ).join("");
    
    const params = this.weights.reduce((sum, layer, i) => {
        const weightCount = layer.reduce((layerSum, neuron) => {
            return layerSum + (neuron instanceof Int8Array ? neuron.length : neuron.length);
        }, 0);
        
        const biasCount = this.biases[i] instanceof Int8Array ? 
            this.biases[i].length : 
            this.biases[i].length;
            
        return sum + weightCount + biasCount;
    }, 0);
    
    output += `\nParameters: ${params}\n`;
    
    console.log(output);
    return this;
}

// Modified save method
async save(name = "model", useBinary = false) {
    try {
        if (useBinary) {
            const metadata = {
                layers: this.layers,
                details: this.details,
                isQuantized: this._isQuantized,
                quantizationParams: this.quantizationParams,
                ...(this.tags && { tags: this.tags })
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
                metadataBytes.length,
                metadataPadding,
                totalWeights,
                totalBiases
            ]);

            // Calculate aligned size
            const totalSize = header.byteLength + 
                            metadataBytes.length +
                            metadataPadding +
                            (this._isQuantized ? 
                                (totalWeights + totalBiases) : 
                                (totalWeights + totalBiases) * 8);

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
                    if (this._isQuantized) {
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
            if (!this._isQuantized && offset % 8 !== 0) {
                offset += (8 - (offset % 8));
            }

            // Write biases
            for (let i = 0; i < this.biases.length; i++) {
                if (this._isQuantized) {
                    view.set(this.biases[i], offset);
                    offset += this.biases[i].length;
                } else {
                    const floatView = new Float64Array(buffer, offset, this.biases[i].length);
                    floatView.set(this.biases[i]);
                    offset += this.biases[i].length * 8;
                }
            }

            const fileBlob = new Blob([buffer], { type: "application/octet-stream" });
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
            // JSON save implementation (unchanged)
            const metadata = {
                layers: this.layers,
                details: this.details,
                isQuantized: this._isQuantized,
                quantizationParams: this.quantizationParams,
                weights: this._isQuantized ? 
                    this.weights.map(layer => layer.map(neuron => Array.from(neuron))) :
                    this.weights,
                biases: this._isQuantized ? 
                    this.biases.map(bias => Array.from(bias)) :
                    this.biases
            };

            const fileBlob = new Blob([JSON.stringify(metadata)], { type: "application/json" });
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

            this._isQuantized = metadata.isQuantized;
            this.quantizationParams = metadata.quantizationParams;
            this.layers = metadata.layers;
            this.details = metadata.details;
            if (metadata.tags) this.tags = metadata.tags;

            // Load weights
            this.weights = [];
            metadata.layers.forEach((layer, i) => {
                const layerWeights = [];
                for (let j = 0; j < layer.outputSize; j++) {
                    if (this._isQuantized) {
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
                if (this._isQuantized) {
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
            
            this._isQuantized = metadata.isQuantized;
            this.quantizationParams = metadata.quantizationParams;
            this.layers = metadata.layers;
            this.details = metadata.details;
            if (metadata.tags) this.tags = metadata.tags;

            if (this._isQuantized) {
                this.weights = metadata.weights.map(layer =>
                    layer.map(neuron => new Int8Array(neuron))
                );
                this.biases = metadata.biases.map(bias =>
                    new Int8Array(bias)
                );
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
  
    // 🚀 Optimization Methods
  
    // Initialize Optimizer: Sets up variables for Adam optimizer
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
  
    // Update Weights: Applies the chosen optimizer (Adam or SGD) to update weights
    #updateWeights(layerIndex, weightGradients, biasGradients, optimizer, params) {
      if (optimizer === 'adam') {
        this.#adamUpdate(layerIndex, weightGradients, biasGradients, params);
      } else {
        this.#sgdUpdate(layerIndex, weightGradients, biasGradients, params.learningRate);
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
  
    // SGD Update: Updates weights using Stochastic Gradient Descent (SGD)
    #sgdUpdate(layerIndex, weightGradients, biasGradients, learningRate) {
      for (let j = 0; j < this.weights[layerIndex].length; j++) {
        for (let k = 0; k < this.weights[layerIndex][j].length; k++) {
          this.weights[layerIndex][j][k] -= learningRate * weightGradients[j][k];
        }
        this.biases[layerIndex][j] -= learningRate * biasGradients[j];
      }
    }
  
    // 🏋️ Training
  
    // Train: Trains the model on a dataset
    async train(trainSet, options = {}) {
      // Fallback property addition when training a loaded model
      if (!('debug' in this)) {
        this.debug = true; // or any default value you want to set
      }
      const {
        epochs = 10, learningRate = 0.212, printEveryEpochs = 1, earlyStopThreshold = 1e-6, testSet = null, callback = null, optimizer = "sgd", lossFunction = "mse"
      } = options;
  
      // Preprocess tags if the output is categorical (e.g., strings)
      if (typeof trainSet[0].output === "string" ||
        (Array.isArray(trainSet[0].output) && typeof trainSet[0].output[0] === "string")) {
        trainSet = this.#preprocesstags(trainSet);
      }
  
      const start = Date.now();
      let t = 0;
  
      // Initialize Adam optimizer if selected
      if (optimizer === "adam") {
        this.#initializeOptimizer();
      }
  
      let lastTrainLoss = 0;
      let lastTestLoss = null;
  
      // Training loop
      for (let epoch = 0; epoch < epochs; epoch++) {
        let trainError = 0;
  
        // Iterate over each data point in the training set
        for (const data of trainSet) {
          t++;
          const {
            layerInputs,
            layerRawOutputs
          } = this.#forwardPropagate(data.input);
          const layerErrors = this.#backPropagate(layerInputs, layerRawOutputs, data.output, lossFunction);
  
          // Update weights and biases for each layer
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
  
          // Accumulate training error
          trainError += this.#lossFunctions[lossFunction].loss(
            layerInputs[layerInputs.length - 1], data.output
          );
        }
  
        // Calculate average training loss
        lastTrainLoss = trainError / trainSet.length;
  
        // Evaluate on test set if provided
        if (testSet) {
          lastTestLoss = this.#evaluateTestSet(testSet, lossFunction);
        }
  
        // Print progress every specified number of epochs
        if ((epoch + 1) % printEveryEpochs === 0 && this.debug) {
          console.log(
            `✨ Epoch ${epoch + 1}, Train Loss: ${lastTrainLoss.toFixed(6)}${
              testSet ? `, Test Loss: ${lastTestLoss.toFixed(6)}` : ""
            }`
          );
        }
  
        // Callback function for custom actions after each epoch
        if (callback) {
          await callback(epoch + 1, lastTrainLoss, lastTestLoss);
        }
  
        // Allow the event loop to process (useful for async operations)
        await new Promise(resolve => setTimeout(resolve, 0));
  
        // Early stopping if training loss is below the threshold
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
  
      // Generate training summary
      const summary = this.#generateTrainingSummary(start, Date.now(), {
        epochs,
        learningRate,
        lastTrainLoss,
        lastTestLoss
      });
  
      this.details = summary;
      return summary;
    }
  
    // Preprocess Tags: Converts categorical outputs to one-hot encoded vectors
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
        output: uniquetags.map(tag =>
          (Array.isArray(item.output) ? item.output : [item.output])
          .includes(tag) ? 1 : 0
        )
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
  
    // Predict: Makes predictions using the trained model
    predict(input, tags = true) {
      const {
        layerInputs,
        layerRawOutputs
      } = this.#forwardPropagate(input);
      const output = layerInputs[layerInputs.length - 1];
  
      // If the output is categorical (e.g., softmax), return probabilities with tags
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


    
    // ℹ️ Info: Updates model metadata (e.g., author, license, etc.)
    info(infoUpdates) {
      this.details.info = infoUpdates;
    }
  }
