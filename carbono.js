// 🧠 carbono: A Fun and Friendly Neural Network Class 🧠
// This micro-library wraps everything you need to have
// This is the simplest yet functional feedforward MLP in JS
class carbono {
  constructor(debug = true) {
    this.layers = []; // 📚 Stores info about each layer
    this.weights = []; // ⚖️ Stores weights for each layer
    this.biases = []; // 🔧 Stores biases for each layer
    this.activations = []; // 🚀 Stores activation functions for each layer
    this.details = {}; // 📊 Stores details about the model
    this.labels = null; // 🏷️ Store class labels
    this.debug = debug; // 🐛 Enables or disables debug messages

    // 🌟 Added for Adam optimizer support
    this.weight_m = []; // 🌪️ First moment estimates for weights
    this.weight_v = []; // 🌩️ Second moment estimates for weights
    this.bias_m = [];   // 🌪️ First moment estimates for biases
    this.bias_v = [];   // 🌩️ Second moment estimates for biases
  }

  // 🏗️ Add a new layer to the neural network
  layer(inputSize, outputSize, activation = "tanh") {
    // 🧱 Store layer information
    this.layers.push({
      inputSize,
      outputSize,
      activation,
    });

    // 🔍 Check if the new layer's input size matches the previous layer's output size
    if (this.weights.length > 0) {
      const lastLayerOutputSize = this.layers[this.layers.length - 2].outputSize;
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
        // For softmax, derivative is handled during backpropagation
        return null; // Not used directly
      default:
        throw new Error(
          "Oops! We don't know the derivative of that activation function."
        );
    }
  }

  // 🏋️‍♀️ Train the neural network
  async train(trainSet, options = {}) {
    // 📌 Handle string labels and convert them to one-hot encoding
    if (
      typeof trainSet[0].output === "string" ||
      (Array.isArray(trainSet[0].output) &&
        typeof trainSet[0].output[0] === "string")
    ) {
      // 🔍 Extract unique labels
      const uniqueLabels = Array.from(
        new Set(
          trainSet
            .map((item) =>
              Array.isArray(item.output) ? item.output : [item.output]
            )
            .flat()
        )
      );

      // 🏷️ Store labels
      this.labels = uniqueLabels;

      // 🔄 Convert string labels to one-hot encoding
      trainSet = trainSet.map((item) => {
        const output = Array.isArray(item.output) ? item.output : [item.output];
        const oneHot = uniqueLabels.map((label) =>
          output.includes(label) ? 1 : 0
        );
        return {
          input: item.input,
          output: oneHot,
        };
      });

      // 🏗️ Automatically set up layers for classification if none are defined
      if (this.layers.length === 0) {
        const numInputs = trainSet[0].input.length;
        const numClasses = uniqueLabels.length;
        this.layer(
          numInputs,
          Math.ceil((numInputs + numClasses) / 2),
          "tanh"
        );
        this.layer(Math.ceil((numInputs + numClasses) / 2), numClasses, "softmax");
      }
    }

    const {
      epochs = 200, // 📅 Number of training epochs
      learningRate = 0.212, // 📈 Learning rate for weight updates
      printEveryEpochs = 100, // 🖨️ Frequency of logging progress
      earlyStopThreshold = 1e-6, // 🛑 Threshold for early stopping
      testSet = null, // 🧪 Optional test set for evaluating performance
      callback = null, // 🔄 Optional callback function after each epoch
      optimizer = "sgd", // 🏋️‍♂️ Optimizer to use ('sgd' or 'adam')
      lossFunction = "mse", // 🔥 Loss function to use ('mse' or 'cross-entropy')
    } = options;

    const start = Date.now(); // ⏱️ Start time

    // 🏗️ Initialize layers if none are defined
    if (this.layers.length === 0) {
      const numInputs = trainSet[0].input.length;
      this.layer(numInputs, numInputs, "tanh");
      this.layer(numInputs, 1, "tanh");
    }

    // 🌟 Initialize optimizer variables if Adam is selected
    let t = 0; // 🔢 Time step for Adam optimizer
    if (optimizer === "adam") {
      this.initializeAdam();
    }

    // 🔍 Validate activation and loss function compatibility
    if (lossFunction === "cross-entropy") {
      const outputActivation = this.activations[this.activations.length - 1];
      if (outputActivation !== "softmax" && outputActivation !== "sigmoid") {
        throw new Error(
          "Cross-entropy loss requires the output layer activation to be either 'softmax' or 'sigmoid'."
        );
      }
    }

    // 🔍 Validate that 'softmax' is only used in the output layer
    for (let i = 0; i < this.activations.length - 1; i++) {
      if (this.activations[i] === "softmax") {
        throw new Error(
          "Softmax activation should only be used in the output layer."
        );
      }
    }

    let lastTrainLoss = 0;
    let lastTestLoss = null;

    // 🔄 Training loop over epochs
    for (let epoch = 0; epoch < epochs; epoch++) {
      let trainError = 0;

      // 🔄 Iterate over each training example
      for (const data of trainSet) {
        t++; // ⏱️ Increment time step

        // ➡️ Forward pass
        const layerInputs = [data.input];
        const layerRawOutputs = []; // 🧮 Raw outputs before activation
        for (let i = 0; i < this.weights.length; i++) {
          const inputs = layerInputs[i];
          const weights = this.weights[i];
          const biases = this.biases[i];
          const activation = this.activations[i];
          const rawOutputs = [];

          // 🧮 Calculate raw values
          for (let j = 0; j < weights.length; j++) {
            const weight = weights[j];
            let sum = biases[j];
            for (let k = 0; k < inputs.length; k++) {
              sum += inputs[k] * weight[k];
            }
            rawOutputs.push(sum);
          }

          layerRawOutputs.push(rawOutputs);

          // 🎇 Apply activation function
          const outputs =
            activation === "softmax"
              ? this.activationFunction(rawOutputs, activation)
              : rawOutputs.map((sum) =>
                  this.activationFunction(sum, activation)
                );

          layerInputs.push(outputs);
        }

        // ⬅️ Backward pass
        const outputLayerIndex = this.weights.length - 1;
        const outputLayerInputs = layerInputs[layerInputs.length - 1];
        const outputLayerRawOutputs = layerRawOutputs[layerRawOutputs.length - 1];
        const outputErrors = [];

        // 🔥 Compute error based on loss function
        if (lossFunction === "cross-entropy") {
          // 🛑 Ensure compatibility
          const outputActivation = this.activations[outputLayerIndex];
          if (outputActivation === "softmax" || outputActivation === "sigmoid") {
            // 📉 Cross-entropy error gradient
            for (let i = 0; i < outputLayerInputs.length; i++) {
              const error = outputLayerInputs[i] - data.output[i];
              outputErrors.push(error);
            }
          } else {
            throw new Error(
              "Cross-entropy loss requires 'softmax' or 'sigmoid' activation in the output layer."
            );
          }
        } else if (lossFunction === "mse") {
          // 📊 Mean Squared Error gradient
          for (let i = 0; i < outputLayerInputs.length; i++) {
            const error =
              (outputLayerInputs[i] - data.output[i]) *
              this.activationDerivative(
                outputLayerRawOutputs[i],
                this.activations[outputLayerIndex]
              );
            outputErrors.push(error);
          }
        } else {
          throw new Error("Unsupported loss function.");
        }

        let layerErrors = [outputErrors];

        // 🧮 Propagate errors backward through layers
        for (let i = this.weights.length - 2; i >= 0; i--) {
          const nextLayerWeights = this.weights[i + 1];
          const nextLayerErrors = layerErrors[0];
          const currentLayerRawOutputs = layerRawOutputs[i];
          const errors = [];

          for (let j = 0; j < this.layers[i].outputSize; j++) {
            let error = 0;
            for (let k = 0; k < this.layers[i + 1].outputSize; k++) {
              error += nextLayerErrors[k] * nextLayerWeights[k][j];
            }
            const activationDeriv = this.activationDerivative(
              currentLayerRawOutputs[j],
              this.activations[i]
            );
            if (activationDeriv !== null) {
              error *= activationDeriv;
            }
            errors.push(error);
          }
          layerErrors.unshift(errors);
        }

        // 🔄 Update weights and biases
        for (let i = 0; i < this.weights.length; i++) {
          const inputs = layerInputs[i];
          const errors = layerErrors[i];
          const weights = this.weights[i];
          const biases = this.biases[i];

          // 🌟 Added for gradient storage and optimization
          const weightGradients = [];
          const biasGradients = [];

          for (let j = 0; j < weights.length; j++) {
            const weight = weights[j];
            const weightGradient = [];
            for (let k = 0; k < inputs.length; k++) {
              const grad = errors[j] * inputs[k];
              weightGradient.push(grad);
            }
            weightGradients.push(weightGradient);
            biasGradients.push(errors[j]);
          }

          // 🛠️ Apply optimizer updates
          if (optimizer === "adam") {
            this.applyAdamOptimization(
              i,
              weightGradients,
              biasGradients,
              t,
              learningRate
            );
          } else {
            // 🏃‍♂️ Standard SGD update
            for (let j = 0; j < weights.length; j++) {
              const weight = weights[j];
              for (let k = 0; k < inputs.length; k++) {
                weight[k] -= learningRate * weightGradients[j][k];
              }
              biases[j] -= learningRate * biasGradients[j];
            }
          }
        }

        // 🧮 Calculate loss based on loss function
        if (lossFunction === "cross-entropy") {
          // 📉 Cross-entropy loss for multi-class classification
          // Ensure predictions are clipped to avoid log(0)
          const clippedOutputs = outputLayerInputs.map((pred) =>
            Math.max(Math.min(pred, 1 - 1e-15), 1e-15)
          );
          const singleLoss = -data.output.reduce(
            (sum, target, i) => sum + target * Math.log(clippedOutputs[i]),
            0
          );
          trainError += singleLoss;
        } else if (lossFunction === "mse") {
          // 📊 Mean Squared Error
          const mse = outputErrors.reduce((sum, err) => sum + err ** 2, 0);
          trainError += mse;
        }
      }

      // 📉 Calculate average training loss
      if (lossFunction === "cross-entropy") {
        lastTrainLoss = trainError / trainSet.length;
      } else if (lossFunction === "mse") {
        lastTrainLoss = trainError / trainSet.length;
      }

      // 🧪 Evaluate on test set if provided
      if (testSet) {
        let testError = 0;
        for (const data of testSet) {
          const prediction = this.predict(data.input, false);
          if (lossFunction === "cross-entropy") {
            const outputActivation = this.activations[this.activations.length - 1];
            if (outputActivation === "softmax" || outputActivation === "sigmoid") {
              // 📉 Cross-entropy loss for multi-class classification
              const clippedOutputs = prediction.map((pred) =>
                Math.max(Math.min(pred, 1 - 1e-15), 1e-15)
              );
              const singleLoss = -data.output.reduce(
                (sum, target, i) => sum + target * Math.log(clippedOutputs[i]),
                0
              );
              testError += singleLoss;
            } else {
              throw new Error(
                "Cross-entropy loss requires 'softmax' or 'sigmoid' activation in the output layer."
              );
            }
          } else if (lossFunction === "mse") {
            // 📊 Mean Squared Error
            const mse = prediction.reduce(
              (sum, output, i) => sum + (output - data.output[i]) ** 2,
              0
            );
            testError += mse;
          }
        }
        lastTestLoss = testError / testSet.length;
      }

      // 🖨️ Log progress every specified number of epochs
      if ((epoch + 1) % printEveryEpochs === 0 && this.debug === true) {
        console.log(
          `✨ Epoch ${epoch + 1}, Train Loss: ${lastTrainLoss.toFixed(6)}${
            testSet ? `, Test Loss: ${lastTestLoss.toFixed(6)}` : ""
          }`
        );
      }

      // 🔄 Execute callback if provided
      if (callback) {
        await callback(epoch + 1, lastTrainLoss, lastTestLoss);
      }

      // ⏳ Yield control to avoid blocking (for async purposes)
      await new Promise((resolve) => setTimeout(resolve, 0));

      // 🛑 Early stopping if loss is below threshold
      if (lastTrainLoss < earlyStopThreshold) {
        console.log(
          `🚀 We stopped at epoch ${
            epoch + 1
          } with train loss: ${lastTrainLoss.toFixed(6)}${
            testSet ? ` and test loss: ${lastTestLoss.toFixed(6)}` : ""
          }`
        );
        break;
      }
    }

    const end = Date.now(); // ⏱️ End time

    // 📊 Calculate total number of parameters
    let totalParams = 0;
    for (let i = 0; i < this.weights.length; i++) {
      const weightLayer = this.weights[i];
      const biasLayer = this.biases[i];
      totalParams += weightLayer.flat().length + biasLayer.length;
    }

    // 📋 Training summary
    const trainingSummary = {
      trainLoss: lastTrainLoss,
      testLoss: lastTestLoss,
      parameters: totalParams,
      training: {
        time: end - start, // ⏳ Training time in ms
        epochs,
        learningRate,
      },
      layers: this.layers.map((layer) => ({
        inputSize: layer.inputSize,
        outputSize: layer.outputSize,
        activation: layer.activation,
      })),
    };

    this.details = trainingSummary; // 📚 Store training details
    return trainingSummary; // 📤 Return summary
  }

  // 🌟 Initialize Adam optimizer variables
  initializeAdam() {
    this.weight_m = this.weights.map((weightLayer) =>
      weightLayer.map((row) => row.map(() => 0))
    );
    this.weight_v = this.weights.map((weightLayer) =>
      weightLayer.map((row) => row.map(() => 0))
    );
    this.bias_m = this.biases.map((biasLayer) => biasLayer.map(() => 0));
    this.bias_v = this.biases.map((biasLayer) => biasLayer.map(() => 0));
  }

  // 🌟 Apply Adam optimizer updates
  applyAdamOptimization(
    layerIndex,
    weightGradients,
    biasGradients,
    t,
    learningRate
  ) {
    const beta1 = 0.9;
    const beta2 = 0.999;
    const epsilon = 1e-8;

    const weights = this.weights[layerIndex];
    const biases = this.biases[layerIndex];
    const weight_m = this.weight_m[layerIndex];
    const weight_v = this.weight_v[layerIndex];
    const bias_m = this.bias_m[layerIndex];
    const bias_v = this.bias_v[layerIndex];

    for (let j = 0; j < weights.length; j++) {
      // Update weights
      for (let k = 0; k < weights[j].length; k++) {
        // Gradient
        const g = weightGradients[j][k];

        // Update moments
        weight_m[j][k] = beta1 * weight_m[j][k] + (1 - beta1) * g;
        weight_v[j][k] = beta2 * weight_v[j][k] + (1 - beta2) * g * g;

        // Bias correction
        const m_hat = weight_m[j][k] / (1 - Math.pow(beta1, t));
        const v_hat = weight_v[j][k] / (1 - Math.pow(beta2, t));

        // Update weights
        weights[j][k] -= (learningRate * m_hat) / (Math.sqrt(v_hat) + epsilon);
      }

      // Update biases
      const g_bias = biasGradients[j];

      // Update moments
      bias_m[j] = beta1 * bias_m[j] + (1 - beta1) * g_bias;
      bias_v[j] = beta2 * bias_v[j] + (1 - beta2) * g_bias * g_bias;

      // Bias correction
      const m_hat_bias = bias_m[j] / (1 - Math.pow(beta1, t));
      const v_hat_bias = bias_v[j] / (1 - Math.pow(beta2, t));

      // Update biases
      biases[j] -= (learningRate * m_hat_bias) / (Math.sqrt(v_hat_bias) + epsilon);
    }
  }

  // 🔮 Predict the output for a given input, optionally returning labeled probabilities
  predict(input, tags = true) {
    let layerInput = input; // 📝 Initial input
    const allActivations = [input]; // 🌐 Store activations of all layers
    const allRawValues = []; // 🧮 Store raw values before activation

    // ➡️ Forward pass through all layers
    for (let i = 0; i < this.weights.length; i++) {
      const weights = this.weights[i];
      const biases = this.biases[i];
      const activation = this.activations[i];
      const rawValues = [];

      // 🧮 Calculate raw values
      for (let j = 0; j < weights.length; j++) {
        const weight = weights[j];
        let sum = biases[j];
        for (let k = 0; k < layerInput.length; k++) {
          sum += layerInput[k] * weight[k];
        }
        rawValues.push(sum);
      }

      // 🎇 Apply activation function
      const layerOutput =
        activation === "softmax"
          ? this.activationFunction(rawValues, "softmax")
          : rawValues.map((sum) => this.activationFunction(sum, activation));

      allRawValues.push(rawValues);
      allActivations.push(layerOutput);
      layerInput = layerOutput; // 🔄 Set input for next layer
    }

    this.lastActivations = allActivations; // 📖 Store activations
    this.lastRawValues = allRawValues; // 📖 Store raw values

    // 🏷️ If labels are available and last layer is softmax, return labeled probabilities
    if (
      this.labels &&
      this.activations[this.activations.length - 1] === "softmax" &&
      tags === true
    ) {
      return layerInput
        .map((prob, idx) => ({
          label: this.labels[idx],
          probability: prob,
        }))
        .sort((a, b) => b.probability - a.probability); // 📈 Sort by probability
    }

    return layerInput; // 🎯 Return raw output
  }

  // 💾 Save the model to a file
  save(name = "model") {
    const data = {
      weights: this.weights,
      biases: this.biases,
      activations: this.activations,
      layers: this.layers,
      details: this.details,
      labels: this.labels,
    };
    const blob = new Blob([JSON.stringify(data)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${name}.uai`;
    a.click();
    URL.revokeObjectURL(url); // 🚮 Clean up
  }

  // 📂 Load a saved model from a file
  load(callback) {
    const handleListener = (event) => {
      const file = event.target.files[0]; // 📄 Selected file
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (event) => {
        const text = event.target.result;
        try {
          const data = JSON.parse(text); // 📖 Parse JSON
          this.weights = data.weights;
          this.biases = data.biases;
          this.activations = data.activations;
          this.layers = data.layers;
          this.details = data.details;
          this.labels = data.labels;
          callback(); // 🔄 Execute callback
          if (this.debug === true) console.log("✅ Model loaded successfully!");
          input.removeEventListener("change", handleListener);
          input.remove(); // 🗑️ Remove input element
        } catch (e) {
          input.removeEventListener("change", handleListener);
          input.remove();
          if (this.debug === true)
            console.error("❌ Failed to load model:", e);
        }
      };
      reader.readAsText(file); // 📖 Read file content
    };
    const input = document.createElement("input"); // 📥 Create file input
    input.type = "file";
    input.accept = ".uai";
    input.style.opacity = "0"; // 🕶️ Hide input
    document.body.append(input);
    input.addEventListener("change", handleListener.bind(this)); // 📑 Listen for file selection
    input.click(); // 🖱️ Trigger file dialog
  }
}
