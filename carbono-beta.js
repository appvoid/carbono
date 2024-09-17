// ✧ carbono: The funiest yet complete neural network js class to learn from
class carbono {
  // ➤ this is the starting point to your new feedforward machine!
  constructor(debug = true) {
    // 🧱 Structural components of our network
    this.layers = [];        // 🥞 Stack of layer configurations
    this.weights = [];       // ⚖️ Connection strengths between neurons
    this.biases = [];        // 🔧 Neuron's individual quirks
    this.activations = [];   // 🚀 Functions that give neurons their personality
    this.details = {};       // 📊 Training results and network info
    this.debug = debug;      // 🐛 To log or not to log, that is the question
    this.dropoutRates = [];  // 🎭 Randomly silencing neurons for better generalization
    // 🏃‍♂️ Adam optimizer components
    this.m = [];             // 🏃‍♂️ Momentum: Keep on rollin'
    this.v = [];             // 🚂 Velocity: Full steam ahead!
    this.beta1 = 0.9;        // 🎚️ Momentum decay rate
    this.beta2 = 0.999;      // 🎛️ Velocity decay rate
    this.epsilon = 1e-8;     // 🦠 Tiny number to prevent division by zero
    // 🎛️ Advanced training options
    this.l2Lambda = 0;                    // 🏋️ L2 regularization strength
    this.lossFunction = 'mse';            // 📉 How we measure our mistakes
    this.learningRateDecay = 1;           // 🐌 Slowing down learning over time
    this.useBatchNorm = false;            // 🧘‍♂️ Keeping inputs zen and balanced
    this.gradientClippingThreshold = Infinity;  // ✂️ Taming wild gradients
  }

  // 🧱 Layer: Adding a new layer to our neural skyscraper
  layer(inputSize, outputSize, activation = 'tanh', dropoutRate = 0) {
    // 🏗️ Construct the layer blueprint
    this.layers.push({ inputSize, outputSize, activation });
    this.dropoutRates.push(dropoutRate);
    // 🧮 Check if the input size matches the previous layer's output
    if (this.weights.length > 0) {
      const lastLayerOutputSize = this.layers[this.layers.length - 2].outputSize;
      if (inputSize !== lastLayerOutputSize) {
        throw new Error('🚨 Oops! Layer sizes don\'t match. Did you forget how to count?');
      }
    }
    
    // 🎲 Initialize weights with Xavier/Glorot initialization
    const weights = Array(outputSize).fill().map(() => 
      Array(inputSize).fill().map(() => (Math.random() - 0.5) * 2 * Math.sqrt(6 / (inputSize + outputSize)))
    );
    this.weights.push(weights);

    // 🏁 Initialize biases with a small positive value
    const biases = Array(outputSize).fill(0.01);
    this.biases.push(biases);

    // 🚀 Store the activation function
    this.activations.push(activation);

    // 🏃‍♂️ Initialize Adam optimizer parameters
    this.m.push(Array(outputSize).fill().map(() => Array(inputSize).fill(0)));
    this.v.push(Array(outputSize).fill().map(() => Array(inputSize).fill(0)));

    // 🧘‍♂️ Initialize batch normalization parameters if enabled
    if (this.useBatchNorm) {
      this.batchNormParams = this.batchNormParams || [];
      this.batchNormParams.push({
        gamma: Array(outputSize).fill(1),     // 🎚️ Scaling parameter
        beta: Array(outputSize).fill(0),      // 🔼 Shifting parameter
        movingMean: Array(outputSize).fill(0),       // 📊 Running mean
        movingVariance: Array(outputSize).fill(1)    // 📏 Running variance
      });
    }
  }

  // 🚀 Activation Function: Giving neurons their unique personalities
  activationFunction(x, activation) {
    switch (activation) {
      case 'tanh':
        return Math.tanh(x);  // 〰️ Squishes input to range [-1, 1]
      case 'sigmoid':
        return 1 / (1 + Math.exp(-x));  // 📈 S-shaped curve, range [0, 1]
      case 'relu':
        return Math.max(0, x);  // 📐 Returns x if positive, else 0
      case 'selu':
        const alpha = 1.67326;
        const scale = 1.0507;
        return x > 0 ? scale * x : scale * alpha * (Math.exp(x) - 1);  // 🚀 Self-normalizing activation
      case 'softmax':
        throw new Error('🎭 Softmax is a diva and needs special treatment!');
      default:
        throw new Error('🤷‍♂️ Unknown activation function. Did you make it up?');
    }
  }

  // 📐 Activation Derivative: Calculating the slope of our activation functions
  activationDerivative(x, activation) {
    switch (activation) {
      case 'tanh':
        return 1 - Math.pow(Math.tanh(x), 2);  // 〰️ Derivative of tanh
      case 'sigmoid':
        const sigmoid = 1 / (1 + Math.exp(-x));
        return sigmoid * (1 - sigmoid);  // 📉 Derivative of sigmoid
      case 'relu':
        return x > 0 ? 1 : 0;  // 📐 1 for positive x, 0 otherwise
      case 'selu':
        const alpha = 1.67326;
        const scale = 1.0507;
        return x > 0 ? scale : scale * alpha * Math.exp(x);  // 🚀 Derivative of SELU
      case 'softmax':
        throw new Error('🎭 Softmax derivative is a drama queen and needs special handling!');
      default:
        throw new Error('🤷‍♂️ Unknown activation function derivative. Are you just making stuff up now?');
    }
  }

  // 🎭 Softmax: The attention seeker of activation functions
  softmax(vector) {
    const max = Math.max(...vector);
    const exps = vector.map(x => Math.exp(x - max));  // 🧮 Subtract max for numerical stability
    const sumExps = exps.reduce((a, b) => a + b, 0);
    return exps.map(exp => exp / sumExps);  // 📊 Normalize to get probabilities
  }

  // 🧘‍♂️ Batch Normalization: Keeping our inputs zen and balanced
  batchNormalization(inputs, layerIndex, isTraining) {
    const { gamma, beta, movingMean, movingVariance } = this.batchNormParams[layerIndex];
    const epsilon = 1e-5;  // 🦠 Tiny number to prevent division by zero

    if (isTraining) {
      // 📊 Calculate mean and variance of the current batch
      const mean = inputs.reduce((sum, x) => sum + x, 0) / inputs.length;
      const variance = inputs.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / inputs.length;

      // 🧮 Normalize inputs
      const normalizedInputs = inputs.map(x => (x - mean) / Math.sqrt(variance + epsilon));
      const scaledAndShiftedInputs = normalizedInputs.map((x, i) => gamma[i] * x + beta[i]);

      // 📈 Update moving averages
      const momentum = 0.9;
      for (let i = 0; i < movingMean.length; i++) {
        movingMean[i] = momentum * movingMean[i] + (1 - momentum) * mean;
        movingVariance[i] = momentum * movingVariance[i] + (1 - momentum) * variance;
      }

      return scaledAndShiftedInputs;
    } else {
      // 🔮 Use moving averages for prediction
      return inputs.map((x, i) => 
        gamma[i] * (x - movingMean[i]) / Math.sqrt(movingVariance[i] + epsilon) + beta[i]
      );
    }
  }

  // 🏋️‍♂️ Train: Time to pump some iron and get our network in shape!
  train(trainSet, options = {}) {
    // 🎛️ Set up training options with default values
    let {
      epochs = 200,
      learningRate = 0.001,
      batchSize = 16,
      printEveryEpochs = 100,
      earlyStopThreshold = 1e-6,
      testSet = null,
      l2Lambda = 0,
      lossFunction = 'mse',
      learningRateDecay = 1,
      useBatchNorm = false,
      gradientClippingThreshold = Infinity
    } = options;

    // 🔧 Update class properties
    this.l2Lambda = l2Lambda;
    this.lossFunction = lossFunction;
    this.learningRateDecay = learningRateDecay;
    this.useBatchNorm = useBatchNorm;
    this.gradientClippingThreshold = gradientClippingThreshold;

    const start = Date.now();  // ⏱️ Start the stopwatch!

    // 🧠 Ensure we have at least one hidden layer
    if (this.layers.length === 0) {
      const numInputs = trainSet[0].input.length;
      this.layer(numInputs, numInputs, 'tanh');
      this.layer(numInputs, 1, this.lossFunction === 'cross-entropy' ? 'sigmoid' : 'tanh');
    }

    let lastTrainLoss = 0;
    let lastTestLoss = null;
    let t = 0;  // 🕰️ Time step for Adam optimizer

    // 🏋️‍♂️ Main training loop
    for (let epoch = 0; epoch < epochs; epoch++) {
      let trainError = 0;
      const currentLearningRate = learningRate * Math.pow(this.learningRateDecay, epoch);

      // 🧺 Process data in batches
      for (let b = 0; b < trainSet.length; b += batchSize) {
        const batch = trainSet.slice(b, b + batchSize);
        let batchError = 0;

        // 📉 Initialize gradients
        const gradients = this.weights.map(layer => 
          layer.map(row => Array(row.length).fill(0))
        );

        // 🔄 Forward and backward pass for each data point in the batch
        for (const data of batch) {
          const layerInputs = [data.input];
          const dropoutMasks = [];
          const preActivations = [];

          // 🏃‍♂️ Forward pass
          for (let i = 0; i < this.weights.length; i++) {
            const inputs = layerInputs[i];
            const weights = this.weights[i];
            const biases = this.biases[i];
            const activation = this.activations[i];
            const dropoutRate = this.dropoutRates[i];

            // 🎭 Apply dropout
            let dropoutMask = null;
            if (dropoutRate > 0) {
              dropoutMask = inputs.map(() => Math.random() > dropoutRate ? 1 / (1 - dropoutRate) : 0);
              dropoutMasks.push(dropoutMask);
            }

            // 🧮 Calculate layer outputs
            const outputs = [];
            const preActivationValues = [];
            for (let j = 0; j < weights.length; j++) {
              const weight = weights[j];
              let sum = biases[j];
              for (let k = 0; k < inputs.length; k++) {
                const input = dropoutMask ? inputs[k] * dropoutMask[k] : inputs[k];
                sum += input * weight[k];
              }
              preActivationValues.push(sum);
              outputs.push(sum);
            }
            preActivations.push(preActivationValues);

            // 🚀 Apply activation function
            if (i === this.weights.length - 1 && activation === 'softmax') {
              layerInputs.push(this.softmax(outputs));
            } else {
              layerInputs.push(outputs.map(output => this.activationFunction(output, activation)));
            }

            // 🧘‍♂️ Apply batch normalization if enabled
            if (this.useBatchNorm && i < this.weights.length - 1) {
              layerInputs[layerInputs.length - 1] = this.batchNormalization(layerInputs[layerInputs.length - 1], i, true);
            }
          }

          // 📉 Calculate output errors
          const outputLayerIndex = this.weights.length - 1;
          const outputLayerInputs = layerInputs[layerInputs.length - 1];
          const outputErrors = [];
          for (let i = 0; i < outputLayerInputs.length; i++) {
            let error;
            if (this.lossFunction === 'mse') {
              error = data.output[i] - outputLayerInputs[i];
            } else if (this.lossFunction === 'cross-entropy') {
              error = data.output[i] - outputLayerInputs[i];
            }
            outputErrors.push(error);
          }

          // 🔙 Backward pass
          let layerErrors = [outputErrors];
          for (let i = this.weights.length - 2; i >= 0; i--) {
            const nextLayerWeights = this.weights[i + 1];
            const nextLayerErrors = layerErrors[0];
            const currentLayerInputs = layerInputs[i + 1];
            const currentActivation = this.activations[i];

            const errors = [];
            for (let j = 0; j < this.layers[i].outputSize; j++) {
              let error = 0;
              for (let k = 0; k < this.layers[i + 1].outputSize; k++) {
                error += nextLayerErrors[k] * nextLayerWeights[k][j];
              }
              if (currentActivation !== 'softmax') {
                error *= this.activationDerivative(preActivations[i][j], currentActivation);
              }
              errors.push(error);
            }
            layerErrors.unshift(errors);
          }

          // 📊 Calculate gradients
          for (let i = 0; i < this.weights.length; i++) {
            const inputs = layerInputs[i];
            const errors = layerErrors[i];
            const dropoutMask = dropoutMasks[i];

            for (let j = 0; j < this.weights[i].length; j++) {
              for (let k = 0; k < inputs.length; k++) {
                const input = dropoutMask ? inputs[k] * dropoutMask[k] : inputs[k];
                gradients[i][j][k] += errors[j] * input;
              }
            }
          }

          // 📉 Calculate batch error
          if (this.lossFunction === 'mse') {
            batchError += outputErrors.reduce((sum, error) => sum + Math.pow(error, 2), 0);
          } else if (this.lossFunction === 'cross-entropy') {
            batchError -= outputLayerInputs.reduce((sum, output, index) => 
              sum + data.output[index] * Math.log(output + 1e-15) + (1 - data.output[index]) * Math.log(1 - output + 1e-15), 0);
          }
        }

        // 🔄 Update weights and biases using Adam optimizer
        t++;
        for (let i = 0; i < this.weights.length; i++) {
          for (let j = 0; j < this.weights[i].length; j++) {
            for (let k = 0; k < this.weights[i][j].length; k++) {
              let gradient = gradients[i][j][k] / batch.length + this.l2Lambda * this.weights[i][j][k];

              // ✂️ Apply gradient clipping
              if (this.gradientClippingThreshold < Infinity) {
                gradient = Math.max(Math.min(gradient, this.gradientClippingThreshold), -this.gradientClippingThreshold);
              }

              // 🏃‍♂️ Update moment estimates
              this.m[i][j][k] = this.beta1 * this.m[i][j][k] + (1 - this.beta1) * gradient;
              this.v[i][j][k] = this.beta2 * this.v[i][j][k] + (1 - this.beta2) * gradient * gradient;

              // 🎯 Compute bias-corrected moment estimates
              const mHat = this.m[i][j][k] / (1 - Math.pow(this.beta1, t));
              const vHat = this.v[i][j][k] / (1 - Math.pow(this.beta2, t));

              // 🔧 Update weights
              this.weights[i][j][k] += currentLearningRate * mHat / (Math.sqrt(vHat) + this.epsilon);
            }
            // 🔧 Update biases
            this.biases[i][j] += currentLearningRate * gradients[i][j].reduce((sum, grad) => sum + grad, 0) / batch.length;
          }
        }
        trainError += batchError;
      }

      lastTrainLoss = trainError / trainSet.length;

      // 🧪 Evaluate on test set if provided
      if (testSet) {
        let testError = 0;
        for (const data of testSet) {
          const prediction = this.predict(data.input);
          if (this.lossFunction === 'mse') {
            testError += prediction.reduce((sum, output, index) => 
              sum + Math.pow(data.output[index] - output, 2), 0);
          } else if (this.lossFunction === 'cross-entropy') {
            testError -= prediction.reduce((sum, output, index) => 
              sum + data.output[index] * Math.log(output + 1e-15) + (1 - data.output[index]) * Math.log(1 - output + 1e-15), 0);
          }
        }
        lastTestLoss = testError / testSet.length;
      }

      // 📢 Print progress
      if ((epoch + 1) % printEveryEpochs === 0 && this.debug === true) {
        console.log(`🏋️‍♂️ Epoch ${epoch + 1}, Train Loss: ${lastTrainLoss.toFixed(6)}${testSet ? `, Test Loss: ${lastTestLoss.toFixed(6)}` : ''}`);
      }

      // 🛑 Early stopping
      if (lastTrainLoss < earlyStopThreshold) {
        console.log(`🎉 Woohoo! We've reached our goal at epoch ${epoch + 1} with train loss: ${lastTrainLoss.toFixed(6)}${testSet ? ` and test loss: ${lastTestLoss.toFixed(6)}` : ''}`);
        break;
      }
    }

    const end = Date.now();

    // 📊 Calculate total parameters
    let totalParams = 0;
    for (let i = 0; i < this.weights.length; i++) {
      const weightLayer = this.weights[i];
      const biasLayer = this.biases[i];
      totalParams += weightLayer.flat().length + biasLayer.length;
    }
    // 📝 Prepare training summary
    const trainingSummary = {
      trainLoss: lastTrainLoss,
      testLoss: lastTestLoss,
      parameters: totalParams,
      training: {
        time: end - start,
        epochs,
        learningRate,
        batchSize,
        l2Lambda: this.l2Lambda,
        lossFunction: this.lossFunction,
        learningRateDecay: this.learningRateDecay,
        useBatchNorm: this.useBatchNorm,
        gradientClippingThreshold: this.gradientClippingThreshold
      },
      layers: this.layers.map((layer, index) => ({
        inputSize: layer.inputSize,
        outputSize: layer.outputSize,
        activation: layer.activation,
        dropoutRate: this.dropoutRates[index]
      }))
    };

    this.details = trainingSummary;
  }

  // 🔮 Predict: Let's see what our network thinks!
  predict(input) {
    let layerInput = input;
    for (let i = 0; i < this.weights.length; i++) {
      const weights = this.weights[i];
      const biases = this.biases[i];
      const activation = this.activations[i];

      // 🧮 Calculate layer outputs
      const layerOutput = [];
      for (let j = 0; j < weights.length; j++) {
        const weight = weights[j];
        let sum = biases[j];
        for (let k = 0; k < layerInput.length; k++) {
          sum += layerInput[k] * weight[k];
        }
        layerOutput.push(sum);
      }

      // 🚀 Apply activation function
      if (i === this.weights.length - 1 && activation === 'softmax') {
        layerInput = this.softmax(layerOutput);
      } else {
        layerInput = layerOutput.map(output => this.activationFunction(output, activation));
      }

      // 🧘‍♂️ Apply batch normalization if enabled
      if (this.useBatchNorm && i < this.weights.length - 1) {
        layerInput = this.batchNormalization(layerInput, i, false);
      }
    }
    return layerInput;
  }

  // 💾 Save: Preserve our neural network for future generations!
  save(name = 'model') {
    const data = {
      weights: this.weights,
      biases: this.biases,
      activations: this.activations,
      layers: this.layers,
      details: this.details,
      dropoutRates: this.dropoutRates,
      m: this.m,
      v: this.v,
      beta1: this.beta1,
      beta2: this.beta2,
      epsilon: this.epsilon
    };
    
    const blob = new Blob([JSON.stringify(data)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${name}.json`;
    a.click();

    URL.revokeObjectURL(url);
  }

  // 📂 Load: Bring a saved neural network back to life!
  load(callback) {
    const handleListener = (event) => {
      const file = event.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (event) => {
        const text = event.target.result;
        try {
          const data = JSON.parse(text);
          this.weights = data.weights;
          this.biases = data.biases;
          this.activations = data.activations;
          this.layers = data.layers;
          this.details = data.details;
          callback();
          if (this.debug === true) console.log('🎉 Model loaded successfully! Time to party!');
          input.removeEventListener('change', handleListener);
          input.remove();
        } catch (e) {
          input.removeEventListener('change', handleListener);
          input.remove();
          if (this.debug === true) console.error('😱 Oh no! Failed to load model:', e);
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
