// Neural network --------------------------------------------------------------------------
class carbono {
  constructor(activation = 'tanh', debug = true) {
    this.activation = activation;
    this.layers = [];
    this.weights = [];
    this.biases = [];
    this.details = {};
    this.debug = debug;
  }

  layer(inputSize, outputSize) {
    this.layers.push({ inputSize, outputSize });

    if (this.weights.length > 0) {
      const lastLayerOutputSize =
        this.layers[this.layers.length - 2].outputSize;
      if (inputSize !== lastLayerOutputSize) {
        throw new Error(
          'Input size of the new layer must match the output size of the previous layer.'
        );
      }
    }

    const weights = [];
    for (let i = 0; i < outputSize; i++) {
      const row = [];
      for (let j = 0; j < inputSize; j++) {
        row.push(
          (Math.random() - 0.5) * 2 * Math.sqrt(6 / (inputSize + outputSize))
        ); // Glorot initialization
      }
      weights.push(row);
    }

    this.weights.push(weights);

    const biases = [];
    for (let i = 0; i < outputSize; i++) {
      biases.push(0.01); // Small positive bias
    }

    this.biases.push(biases);
  }

  activationFunction(x) {
    switch (this.activation) {
      case 'tanh':
        return Math.tanh(x);
      case 'sigmoid':
        const sigmoid = 1 / (1 + Math.exp(-x));
        return sigmoid;
      case 'relu':
        return Math.max(0, x);
      case 'selu':
        const alpha = 1.67326;
        const scale = 1.0507;
        return x > 0 ? scale * x : scale * alpha * (Math.exp(x) - 1);
      default:
        throw new Error('Unsupported activation function.');
    }
  }

  activationDerivative(x) {
    switch (this.activation) {
      case 'tanh':
        return 1 - Math.pow(Math.tanh(x), 2);
      case 'sigmoid':
        const sigmoid = 1 / (1 + Math.exp(-x));
        return sigmoid * (1 - sigmoid);
      case 'relu':
        return x > 0 ? 1 : 0;
      case 'selu':
        const alpha = 1.67326;
        const scale = 1.0507;
        return x > 0 ? scale : scale * alpha * Math.exp(x);
      default:
        throw new Error('Unsupported activation function.');
    }
  }

  train(
    dataset,
    epochs = 200,
    learningRate = 0.212,
    batchSize = 16,
    printEveryEpochs = 100
  ) {
    // Timer start
    const start = Date.now();

    // To avoid loop errors
    if (batchSize < 1) batchSize = 2;

    // Automatically initialize layers if not already set
    if (this.layers.length === 0) {
      const numInputs = dataset[0].input.length;
      this.layer(numInputs, numInputs); // hidden layer with 2 units
      this.layer(numInputs, 1); // output layer with 1 unit
    }

    let lastEpochLoss = 0;

    for (let epoch = 0; epoch < epochs; epoch++) {
      let epochError = 0;

      for (let b = 0; b < dataset.length; b += batchSize) {
        const batch = dataset.slice(b, b + batchSize);
        let batchError = 0;

        for (const data of batch) {
          const layerInputs = [data.input];
          for (let i = 0; i < this.weights.length; i++) {
            const inputs = layerInputs[i];
            const weights = this.weights[i];
            const biases = this.biases[i];

            const outputs = [];
            for (let j = 0; j < weights.length; j++) {
              const weight = weights[j];
              let sum = biases[j];
              for (let k = 0; k < inputs.length; k++) {
                sum += inputs[k] * weight[k];
              }
              outputs.push(this.activationFunction(sum));
            }
            layerInputs.push(outputs);
          }

          const outputLayerIndex = this.weights.length - 1;
          const outputLayerInputs = layerInputs[layerInputs.length - 1];
          const outputErrors = [];
          for (let i = 0; i < outputLayerInputs.length; i++) {
            const error = data.output[i] - outputLayerInputs[i];
            outputErrors.push(error);
          }

          let layerErrors = [outputErrors];
          for (let i = this.weights.length - 2; i >= 0; i--) {
            const nextLayerWeights = this.weights[i + 1];
            const nextLayerErrors = layerErrors[0];
            const currentLayerInputs = layerInputs[i + 1];

            const errors = [];
            for (let j = 0; j < this.layers[i].outputSize; j++) {
              let error = 0;
              for (let k = 0; k < this.layers[i + 1].outputSize; k++) {
                error += nextLayerErrors[k] * nextLayerWeights[k][j];
              }
              errors.push(
                error * this.activationDerivative(currentLayerInputs[j])
              );
            }
            layerErrors.unshift(errors);
          }

          for (let i = 0; i < this.weights.length; i++) {
            const inputs = layerInputs[i];
            const outputs = layerInputs[i + 1];
            const errors = layerErrors[i];
            const weights = this.weights[i];
            const biases = this.biases[i];

            for (let j = 0; j < weights.length; j++) {
              const weight = weights[j];
              for (let k = 0; k < inputs.length; k++) {
                weight[k] += learningRate * errors[j] * inputs[k];
              }
              biases[j] += learningRate * errors[j];
            }
          }

          batchError += Math.abs(outputErrors[0]); // Assuming binary output
        }

        // Update learning rate
        // learningRate = Math.max(0.01, learningRate * 0.999);
        epochError += batchError;
        lastEpochLoss = batchError;
      }

      if ((epoch + 1) % printEveryEpochs === 0 && this.debug === true) {
        console.log(`Epoch ${epoch + 1}, Error: ${epochError}`);
      }
    }

    // Timer end
    const end = Date.now();

    // Calculate total number of parameters
    let totalParams = 0;
    for (let i = 0; i < this.weights.length; i++) {
      const weightLayer = this.weights[i];
      const biasLayer = this.biases[i];
      totalParams += weightLayer.flat().length + biasLayer.length;
    }

    // Construct the training summary
    const trainingSummary = {
      loss: lastEpochLoss,
      parameters: totalParams,
      training: {
        time: end - start,
        activation: this.activation,
        epochs, // Added number of epochs
        learningRate, // Added learning rate
        batchSize, // Added batch size
      },
    };

    this.details = trainingSummary;
  }
  predict(input) {
    let layerInput = input;
    for (let i = 0; i < this.weights.length; i++) {
      const weights = this.weights[i];
      const biases = this.biases[i];

      const layerOutput = [];
      for (let j = 0; j < weights.length; j++) {
        const weight = weights[j];
        let sum = biases[j];
        for (let k = 0; k < layerInput.length; k++) {
          sum += layerInput[k] * weight[k];
        }
        layerOutput.push(this.activationFunction(sum));
      }
      layerInput = layerOutput;
    }
    return layerInput;
  }

  save(name = 'model') {
    const data = {
      weights: this.weights,
      biases: this.biases,
      details: this.details,
    };
    const blob = new Blob([JSON.stringify(data)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = `${name}.json`;
    a.click();

    URL.revokeObjectURL(url);
  }

  load(callback) {
    const handleListener = event => {
      const file = event.target.files[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = event => {
        const text = event.target.result;
        try {
          const data = JSON.parse(text);
          this.weights = data.weights;
          this.biases = data.biases;
          this.details = data.details;

          // Set the activation function from the original implementation
          if (this.details.activation) {
            this.activation = this.details.activation;
          }

          callback();
          if (this.debug === true) console.log('Model loaded successfully.');
          input.removeEventListener('change', handleListener);
          input.remove();
        } catch (e) {
          input.removeEventListener('change', handleListener);
          input.remove();
          if (this.debug === true) console.error('Failed to load model:', e);
        }
      };
      reader.readAsText(file);
    };

    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.style.opacity = '0';
    document.body.append(input);
    input.addEventListener('change', handleListener.bind(this));
    input.click();
  }
}
