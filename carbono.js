// 🧠 carbono: A Fun and Friendly Neural Network Class 🧠
// This micro-library wraps everything you need to have
// This is the simplest yet functional feedforward mlp in js
class carbono {
  constructor(debug = true) {
    this.layers = []; // 📚 Stores info about each layer
    this.weights = []; // ⚖️ Stores weights for each layer
    this.biases = []; // 🔧 Stores biases for each layer
    this.activations = []; // 🚀 Stores activation functions for each layer
    this.details = {}; // 📊 Stores details about the model
    this.debug = debug; // 🐛 Enables or disables debug messages
  }

  // 🏗️ Add a new layer to the neural network
  layer(inputSize, outputSize, activation = "tanh") {
    // 🧱 Store layer information
    this.layers.push({
      inputSize,
      outputSize,
      activation
    });

    // 🔍 Check if the new layer's input size matches the previous layer's output size
    if (this.weights.length > 0) {
      const lastLayerOutputSize = this.layers[this.layers.length - 2]
        .outputSize;
      if (inputSize !== lastLayerOutputSize) {
        throw new Error(
          "Oops! The input size of the new layer must match the output size of the previous layer."
        );
      }
    }

    // 🎲 Initialize weights using Xavier/Glorot initialization
    const weights = [];
    for (let i = 0; i < outputSize; i++) {
      const row = [];
      for (let j = 0; j < inputSize; j++) {
        row.push(
          (Math.random() - 0.5) * 2 * Math.sqrt(6 / (inputSize + outputSize))
        );
      }
      weights.push(row);
    }
    this.weights.push(weights);

    // 🎚️ Initialize biases with small positive values
    const biases = Array(outputSize).fill(0.01);
    this.biases.push(biases);

    // 🚀 Store the activation function for this layer
    this.activations.push(activation);
  }

  // 🧮 Apply the activation function
  activationFunction(x, activation) {
    switch (activation) {
      case "tanh":
        return Math.tanh(x); // 〰️ Hyperbolic tangent
      case "sigmoid":
        return 1 / (1 + Math.exp(-x)); // 📈 S-shaped curve
      case "relu":
        return Math.max(0, x); // 📐 Rectified Linear Unit
      case "selu":
        const alpha = 1.67326;
        const scale = 1.0507;
        return x > 0 ? scale * x : scale * alpha * (Math.exp(x) - 1); // 🚀 Scaled Exponential Linear Unit
      case "softmax":
        // Handle array input for softmax
        const expValues = Array.isArray(x)
          ? x.map((val) => Math.exp(val))
          : [Math.exp(x)];
        const sumExp = expValues.reduce((a, b) => a + b, 0);
        return expValues.map((exp) => exp / sumExp);
      default:
        throw new Error("Whoops! We don't know that activation function.");
    }
  }

  // 📐 Calculate the derivative of the activation function
  activationDerivative(x, activation) {
    switch (activation) {
      case "tanh":
        return 1 - Math.pow(Math.tanh(x), 2);
      case "sigmoid":
        const sigmoid = 1 / (1 + Math.exp(-x));
        return sigmoid * (1 - sigmoid);
      case "relu":
        return x > 0 ? 1 : 0;
      case "selu":
        const alpha = 1.67326;
        const scale = 1.0507;
        return x > 0 ? scale : scale * alpha * Math.exp(x);
      case "softmax":
        // For softmax, return simplified diagonal of Jacobian
        const softmaxOutput = this.activationFunction(x, "softmax");
        return softmaxOutput.map((s) => s * (1 - s));
      default:
        throw new Error(
          "Oops! We don't know the derivative of that activation function."
        );
    }
  }

  // 🏋️‍♀️ Train the neural network
  async train(trainSet, options = {}) {
    const {
      epochs = 200,
      learningRate = 0.212,
      batchSize = 16,
      printEveryEpochs = 100,
      earlyStopThreshold = 1e-6,
      testSet = null,
      callback = null
    } = options;

    const start = Date.now();
    if (batchSize < 1) batchSize = 2;

    if (this.layers.length === 0) {
      const numInputs = trainSet[0].input.length;
      this.layer(numInputs, numInputs, "tanh");
      this.layer(numInputs, 1, "tanh");
    }

    let lastTrainLoss = 0;
    let lastTestLoss = null;

    for (let epoch = 0; epoch < epochs; epoch++) {
      let trainError = 0;

      for (let b = 0; b < trainSet.length; b += batchSize) {
        const batch = trainSet.slice(b, b + batchSize);
        let batchError = 0;

        for (const data of batch) {
          // Forward pass
          const layerInputs = [data.input];
          for (let i = 0; i < this.weights.length; i++) {
            const inputs = layerInputs[i];
            const weights = this.weights[i];
            const biases = this.biases[i];
            const activation = this.activations[i];
            const rawOutputs = [];

            // Calculate raw values first
            for (let j = 0; j < weights.length; j++) {
              const weight = weights[j];
              let sum = biases[j];
              for (let k = 0; k < inputs.length; k++) {
                sum += inputs[k] * weight[k];
              }
              rawOutputs.push(sum);
            }

            // Apply activation
            const outputs =
              activation === "softmax"
                ? this.activationFunction(rawOutputs, activation)
                : rawOutputs.map((sum) =>
                    this.activationFunction(sum, activation)
                  );

            layerInputs.push(outputs);
          }

          // Backward pass
          const outputLayerIndex = this.weights.length - 1;
          const outputLayerInputs = layerInputs[layerInputs.length - 1];
          const outputErrors = [];

          if (this.activations[outputLayerIndex] === "softmax") {
            // For softmax, calculate cross-entropy error gradient
            for (let i = 0; i < outputLayerInputs.length; i++) {
              const error = data.output[i] - outputLayerInputs[i];
              outputErrors.push(error);
            }
          } else {
            for (let i = 0; i < outputLayerInputs.length; i++) {
              const error = data.output[i] - outputLayerInputs[i];
              outputErrors.push(error);
            }
          }

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
              errors.push(
                error *
                  this.activationDerivative(
                    currentLayerInputs[j],
                    currentActivation
                  )
              );
            }
            layerErrors.unshift(errors);
          }

          // Update weights and biases
          for (let i = 0; i < this.weights.length; i++) {
            const inputs = layerInputs[i];
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

          // Calculate error based on output type
          if (this.activations[outputLayerIndex] === "softmax") {
            // Use cross-entropy error for softmax
            batchError += -outputLayerInputs.reduce(
              (sum, output, i) =>
                sum + data.output[i] * Math.log(output + 1e-15),
              0
            );
          } else {
            batchError += Math.abs(outputErrors[0]);
          }
        }
        trainError += batchError;
      }

      lastTrainLoss = trainError / trainSet.length;

      if (testSet) {
        let testError = 0;
        for (const data of testSet) {
          const prediction = this.predict(data.input);
          if (this.activations[this.activations.length - 1] === "softmax") {
            // Use cross-entropy error for softmax
            testError += -prediction.reduce(
              (sum, output, i) =>
                sum + data.output[i] * Math.log(output + 1e-15),
              0
            );
          } else {
            testError += Math.abs(data.output[0] - prediction[0]);
          }
        }
        lastTestLoss = testError / testSet.length;
      }

      if ((epoch + 1) % printEveryEpochs === 0 && this.debug === true) {
        console.log(
          `Epoch ${epoch + 1}, Train Loss: ${lastTrainLoss.toFixed(6)}${
            testSet ? `, Test Loss: ${lastTestLoss.toFixed(6)}` : ""
          }`
        );
      }

      if (callback) {
        await callback(epoch + 1, lastTrainLoss, lastTestLoss);
      }

      await new Promise((resolve) => setTimeout(resolve, 0));

      if (lastTrainLoss < earlyStopThreshold) {
        console.log(
          `We stopped at epoch ${
            epoch + 1
          } with train loss: ${lastTrainLoss.toFixed(6)}${
            testSet ? ` and test loss: ${lastTestLoss.toFixed(6)}` : ""
          }`
        );
        break;
      }
    }

    const end = Date.now();
    let totalParams = 0;
    for (let i = 0; i < this.weights.length; i++) {
      const weightLayer = this.weights[i];
      const biasLayer = this.biases[i];
      totalParams += weightLayer.flat().length + biasLayer.length;
    }

    const trainingSummary = {
      trainLoss: lastTrainLoss,
      testLoss: lastTestLoss,
      parameters: totalParams,
      training: {
        time: end - start,
        epochs,
        learningRate,
        batchSize
      },
      layers: this.layers.map((layer) => ({
        inputSize: layer.inputSize,
        outputSize: layer.outputSize,
        activation: layer.activation
      }))
    };

    this.details = trainingSummary;
    return trainingSummary;
  }

  predict(input) {
    let layerInput = input;
    const allActivations = [input];
    const allRawValues = [];

    for (let i = 0; i < this.weights.length; i++) {
      const weights = this.weights[i];
      const biases = this.biases[i];
      const activation = this.activations[i];
      const rawValues = [];

      // Calculate raw values first
      for (let j = 0; j < weights.length; j++) {
        const weight = weights[j];
        let sum = biases[j];
        for (let k = 0; k < layerInput.length; k++) {
          sum += layerInput[k] * weight[k];
        }
        rawValues.push(sum);
      }

      // Apply activation function
      const layerOutput =
        activation === "softmax"
          ? this.activationFunction(rawValues, "softmax")
          : rawValues.map((sum) => this.activationFunction(sum, activation));

      allRawValues.push(rawValues);
      allActivations.push(layerOutput);
      layerInput = layerOutput;
    }

    this.lastActivations = allActivations;
    this.lastRawValues = allRawValues;
    return layerInput;
  }

  // 💾 Save the model to a file
  save(name = "model") {
    const data = {
      weights: this.weights,
      biases: this.biases,
      activations: this.activations,
      layers: this.layers,
      details: this.details
    };
    const blob = new Blob([JSON.stringify(data)], {
      type: "application/json"
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${name}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  // 📂 Load a saved model from a file
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
          if (this.debug === true) console.log("Model loaded successfully!");
          input.removeEventListener("change", handleListener);
          input.remove();
        } catch (e) {
          input.removeEventListener("change", handleListener);
          input.remove();
          if (this.debug === true) console.error("Failed to load model:", e);
        }
      };
      reader.readAsText(file);
    };
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".json";
    input.style.opacity = "0";
    document.body.append(input);
    input.addEventListener("change", handleListener.bind(this));
    input.click();
  }
}
