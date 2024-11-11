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
      layer(inputSize, outputSize, activation = 'tanh') {
        // 🧱 Store layer information
        this.layers.push({
          inputSize,
          outputSize,
          activation
        });
        // 🔍 Check if the new layer's input size matches the previous layer's output size
        if (this.weights.length > 0) {
          const lastLayerOutputSize = this.layers[this.layers.length - 2].outputSize;
          if (inputSize !== lastLayerOutputSize) {
            throw new Error('Oops! The input size of the new layer must match the output size of the previous layer.');
          }
        }
        // 🎲 Initialize weights using Xavier/Glorot initialization
        const weights = [];
        for (let i = 0; i < outputSize; i++) {
          const row = [];
          for (let j = 0; j < inputSize; j++) {
            row.push((Math.random() - 0.5) * 2 * Math.sqrt(6 / (inputSize + outputSize)));
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
          case 'tanh':
            return Math.tanh(x); // 〰️ Hyperbolic tangent
          case 'sigmoid':
            return 1 / (1 + Math.exp(-x)); // 📈 S-shaped curve
          case 'relu':
            return Math.max(0, x); // 📐 Rectified Linear Unit
          case 'selu':
            const alpha = 1.67326;
            const scale = 1.0507;
            return x > 0 ? scale * x : scale * alpha * (Math.exp(x) - 1); // 🚀 Scaled Exponential Linear Unit
          default:
            throw new Error('Whoops! We don\'t know that activation function.');
        }
      }
      // 📐 Calculate the derivative of the activation function
      activationDerivative(x, activation) {
        switch (activation) {
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
            throw new Error('Oops! We don\'t know the derivative of that activation function.');
        }
      }
      // 🏋️‍♀️ Train the neural network
      async train(trainSet, options = {}) {
        // 🎛️ Set up training options with default values
        const {
          epochs = 200, // 🔄 Number of times to go through the entire dataset
            learningRate = 0.212, // 📏 How big of steps to take when adjusting weights
            batchSize = 16, // 📦 Number of samples to process before updating weights
            printEveryEpochs = 100, // 🖨️ How often to print progress
            earlyStopThreshold = 1e-6, // 🛑 When to stop if the error is small enough
            testSet = null, // 🧪 Optional test set for evaluation
            callback = null // 📡 Callback function for real-time updates
        } = options;
        const start = Date.now(); // ⏱️ Start the timer
        // 🛡️ Make sure batch size is at least 2
        if (batchSize < 1) batchSize = 2;
        // 🏗️ Automatically create layers if none exist
        if (this.layers.length === 0) {
          const numInputs = trainSet[0].input.length;
          this.layer(numInputs, numInputs, 'tanh');
          this.layer(numInputs, 1, 'tanh');
        }
        let lastTrainLoss = 0;
        let lastTestLoss = null;
        // 🔄 Main training loop
        for (let epoch = 0; epoch < epochs; epoch++) {
          let trainError = 0;
          // 📦 Process data in batches
          for (let b = 0; b < trainSet.length; b += batchSize) {
            const batch = trainSet.slice(b, b + batchSize);
            let batchError = 0;
            // 🧠 Forward pass and backward pass for each item in the batch
            for (const data of batch) {
              // 🏃‍♂️ Forward pass
              const layerInputs = [data.input];
              for (let i = 0; i < this.weights.length; i++) {
                const inputs = layerInputs[i];
                const weights = this.weights[i];
                const biases = this.biases[i];
                const activation = this.activations[i];
                const outputs = [];
                for (let j = 0; j < weights.length; j++) {
                  const weight = weights[j];
                  let sum = biases[j];
                  for (let k = 0; k < inputs.length; k++) {
                    sum += inputs[k] * weight[k];
                  }
                  outputs.push(this.activationFunction(sum, activation));
                }
                layerInputs.push(outputs);
              }
              // 🔙 Backward pass
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
                const currentActivation = this.activations[i];
                const errors = [];
                for (let j = 0; j < this.layers[i].outputSize; j++) {
                  let error = 0;
                  for (let k = 0; k < this.layers[i + 1].outputSize; k++) {
                    error += nextLayerErrors[k] * nextLayerWeights[k][j];
                  }
                  errors.push(error * this.activationDerivative(currentLayerInputs[j], currentActivation));
                }
                layerErrors.unshift(errors);
              }
              // 🔧 Update weights and biases
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
              batchError += Math.abs(outputErrors[0]); // Assuming binary output
            }
            trainError += batchError;
          }
          lastTrainLoss = trainError / trainSet.length;
          // 🧪 Evaluate on test set if provided
          if (testSet) {
            let testError = 0;
            for (const data of testSet) {
              const prediction = this.predict(data.input);
              testError += Math.abs(data.output[0] - prediction[0]);
            }
            lastTestLoss = testError / testSet.length;
          }
          // 📢 Print progress if needed
          if ((epoch + 1) % printEveryEpochs === 0 && this.debug === true) {
            console.log(`Epoch ${epoch + 1}, Train Loss: ${lastTrainLoss.toFixed(6)}${testSet ? `, Test Loss: ${lastTestLoss.toFixed(6)}` : ''}`);
          }
          // 📡 Call the callback function with current progress
          if (callback) {
            await callback(epoch + 1, lastTrainLoss, lastTestLoss);
          }
          // Add a small delay to prevent UI freezing
          await new Promise(resolve => setTimeout(resolve, 0));
          // 🛑 Check for early stopping
          if (lastTrainLoss < earlyStopThreshold) {
            console.log(`We stopped at epoch ${epoch + 1} with train loss: ${lastTrainLoss.toFixed(6)}${testSet ? ` and test loss: ${lastTestLoss.toFixed(6)}` : ''}`);
            break;
          }
        }
        const end = Date.now(); // ⏱️ Stop the timer
        // 🧮 Calculate total number of parameters
        let totalParams = 0;
        for (let i = 0; i < this.weights.length; i++) {
          const weightLayer = this.weights[i];
          const biasLayer = this.biases[i];
          totalParams += weightLayer.flat().length + biasLayer.length;
        }
        // 📊 Create a summary of the training
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
          layers: this.layers.map(layer => ({
            inputSize: layer.inputSize,
            outputSize: layer.outputSize,
            activation: layer.activation
          }))
        };
        this.details = trainingSummary;
        return trainingSummary;
      }
      // 🔮 Use the trained network to make predictions
      predict(input) {
        let layerInput = input;
        for (let i = 0; i < this.weights.length; i++) {
          const weights = this.weights[i];
          const biases = this.biases[i];
          const activation = this.activations[i];
          const layerOutput = [];
          for (let j = 0; j < weights.length; j++) {
            const weight = weights[j];
            let sum = biases[j];
            for (let k = 0; k < layerInput.length; k++) {
              sum += layerInput[k] * weight[k];
            }
            layerOutput.push(this.activationFunction(sum, activation));
          }
          layerInput = layerOutput;
        }
        return layerInput;
      }
      // 💾 Save the model to a file
      save(name = 'model') {
        const data = {
          weights: this.weights,
          biases: this.biases,
          activations: this.activations,
          layers: this.layers,
          details: this.details
        };
        const blob = new Blob([JSON.stringify(data)], {
          type: 'application/json'
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
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
              if (this.debug === true) console.log('Model loaded successfully!');
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
