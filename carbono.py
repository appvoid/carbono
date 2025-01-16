import math
import json
import time
import random
import struct
from typing import List, Dict, Union, Optional, Tuple, Any
import base64

class Carbono:
    """Neural network implementation in pure Python without external dependencies."""
    
    def __init__(self, debug: bool = True):
        """Initialize the neural network with default settings."""
        self.layers = []  # Stores the layers of the neural network
        self.weights = []  # Stores the weights for each layer
        self.biases = []  # Stores the biases for each layer
        self.details = {}  # Stores metadata about the model
        self.debug = debug
        
    def _xavier(self, input_size: int, output_size: int) -> float:
        """Xavier initialization for weights."""
        return (random.random() - 0.5) * 2 * math.sqrt(6 / (input_size + output_size))
        
    def _clip(self, value: float, min_val: float = 1e-15, max_val: float = 1 - 1e-15) -> float:
        """Ensure values stay within a specified range."""
        return max(min(value, max_val), min_val)
        
    def _matrix_multiply(self, a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
        """Matrix multiplication implementation."""
        return [[sum(val_a * b[j][i] for j, val_a in enumerate(row_a))
                for i in range(len(b[0]))]
                for row_a in a]

    # Activation functions
    _activation_functions = {
        'tanh': {
            'fn': lambda x: math.tanh(x),
            'derivative': lambda x: 1 - math.tanh(x) ** 2
        },
        'sigmoid': {
            'fn': lambda x: 1 / (1 + math.exp(-x)),
            'derivative': lambda x: (1 / (1 + math.exp(-x))) * (1 - 1 / (1 + math.exp(-x)))
        },
        'relu': {
            'fn': lambda x: max(0, x),
            'derivative': lambda x: 1 if x > 0 else 0
        },
        'selu': {
            'fn': lambda x: 1.0507 * x if x > 0 else 1.0507 * 1.67326 * (math.exp(x) - 1),
            'derivative': lambda x: 1.0507 if x > 0 else 1.0507 * 1.67326 * math.exp(x)
        }
    }

    # Add softmax separately due to its special handling of arrays
    def _softmax(self, x: Union[float, List[float]]) -> List[float]:
        """Softmax activation function."""
        if not isinstance(x, list):
            x = [x]
        exp_values = [math.exp(val) for val in x]
        sum_exp = sum(exp_values)
        return [exp / sum_exp for exp in exp_values]

    # Loss functions
    _loss_functions = {
        'mse': {
            'loss': lambda predicted, actual: sum((pred - act) ** 2 
                                                for pred, act in zip(predicted, actual)),
            'derivative': lambda predicted, actual, activation:
                [(pred - act) * (1 if activation == 'softmax' else 
                 Carbono._activation_functions[activation]['derivative'](pred))
                 for pred, act in zip(predicted, actual)]
        },
        'cross-entropy': {
            # Fixed cross-entropy loss function
            'loss': lambda predicted, actual: -sum(
                target * math.log(max(pred, 1e-15)) 
                for pred, target in zip(predicted, actual)
            ),
            'derivative': lambda predicted, actual, _: [
                pred - act for pred, act in zip(predicted, actual)
            ]
        }
    }
    def save(self, name: str = "model", use_binary: bool = False) -> bool:
        """Save the model to a file with JavaScript-compatible format."""
        try:
            if not self.weights or not self.biases:
                raise ValueError("Weights or biases are empty. Cannot save model.")

            # Prepare metadata
            if not hasattr(self, 'details'):
                self.details = {}
                
            if 'info' not in self.details:
                self.details['info'] = {
                    'name': name,
                    'author': 'Carbono Python',
                    'license': 'MIT',
                    'note': '',
                    'date': time.strftime('%Y-%m-%dT%H:%M:%SZ')
                }

            # Prepare metadata for weights/biases structure - match JS format exactly
            layer_info = {
                'weightShapes': [[len(layer), len(layer[0])] for layer in self.weights],
                'biasShapes': [len(layer) for layer in self.biases]
            }

            # Create metadata object matching JS expectations
            metadata = {
                'layers': self.layers,
                'details': self.details,
                'layerInfo': layer_info
            }
            
            if hasattr(self, 'tags'):
                metadata['tags'] = self.tags

            if use_binary:
                # Flatten weights and biases
                weight_data = [val for layer in self.weights for row in layer for val in row]
                bias_data = [val for layer in self.biases for val in layer]

                # Convert metadata to bytes
                metadata_bytes = json.dumps(metadata).encode('utf-8')
                metadata_padding = (4 - (len(metadata_bytes) % 4)) % 4

                # Create header
                header = struct.pack('IIII',
                    len(metadata_bytes),
                    metadata_padding,
                    len(weight_data),
                    len(bias_data)
                )

                with open(f"{name}.uai", 'wb') as f:
                    # Write header
                    f.write(header)
                    
                    # Write metadata with padding
                    f.write(metadata_bytes)
                    f.write(b'\x00' * metadata_padding)
                    
                    # Write weights and biases as 32-bit floats
                    for value in weight_data:
                        f.write(struct.pack('f', float(value)))
                    for value in bias_data:
                        f.write(struct.pack('f', float(value)))
            else:
                # Use JSON mode
                metadata['weights'] = self.weights
                metadata['biases'] = self.biases
                
                with open(f"{name}.json", 'w') as f:
                    json.dump(metadata, f)

            return True

        except Exception as error:
            print(f"Save process failed: {error}")
            raise

    def load(self, callback: Optional[callable] = None, use_binary: bool = False) -> bool:
        """Load a model from a file."""
        try:
            file_path = input(f"Enter path to model file (*{'.uai' if use_binary else '.json'}): ")

            # Create empty model state
            self.weights = []
            self.biases = []
            self.layers = []
            self.details = {}
            self.tags = None

            with open(file_path, 'rb' if use_binary else 'r') as f:
                if use_binary:
                    # Read header
                    header = struct.unpack('IIII', f.read(16))
                    metadata_length, metadata_padding, weight_length, bias_length = header

                    # Read metadata
                    metadata_bytes = f.read(metadata_length)
                    metadata = json.loads(metadata_bytes.decode('utf-8'))
                    f.read(metadata_padding)

                    # Read weights
                    for shape in metadata['layerInfo']['weightShapes']:
                        layer = []
                        for _ in range(shape[0]):
                            row = []
                            for _ in range(shape[1]):
                                value_bytes = f.read(4)
                                value = struct.unpack('f', value_bytes)[0]
                                row.append(value)
                            layer.append(row)
                        self.weights.append(layer)

                    # Read biases
                    for shape in metadata['layerInfo']['biasShapes']:
                        layer = []
                        for _ in range(shape):
                            value_bytes = f.read(4)
                            value = struct.unpack('f', value_bytes)[0]
                            layer.append(value)
                        self.biases.append(layer)

                else:
                    # Use JSON mode
                    metadata = json.load(f)
                    self.weights = metadata['weights']
                    self.biases = metadata['biases']

                # Load other metadata
                self.layers = metadata['layers']
                self.details = metadata['details']
                if 'tags' in metadata:
                    self.tags = metadata['tags']

            # Model is fully loaded, now call the callback
            if callback and callable(callback):
                callback()

            return True

        except Exception as error:
            print(f"Load process failed: {error}")
            raise

    def _get_activation(self, x: float, activation: str) -> Union[float, List[float]]:
        """Apply the activation function."""
        if activation == 'softmax':
            return self._softmax(x)
        return self._activation_functions[activation]['fn'](x)

    def _get_activation_derivative(self, x: float, activation: str) -> Optional[float]:
        """Get the derivative of the activation function."""
        return self._activation_functions[activation]['derivative'](x) if activation != 'softmax' else None

    def layer(self, input_size: int, output_size: int, activation: str = "tanh") -> 'Carbono':
        """Add a new layer to the neural network."""
        if self.weights and input_size != self.layers[-1]['output_size']:
            raise ValueError("Layer input size must match previous layer output size.")

        self.layers.append({
            'input_size': input_size,
            'output_size': output_size,
            'activation': activation
        })

        # Initialize weights using Xavier initialization
        self.weights.append([[self._xavier(input_size, output_size) 
                            for _ in range(input_size)]
                            for _ in range(output_size)])
        
        # Initialize biases
        self.biases.append([0.01] * output_size)
        return self

    def _forward_propagate(self, input_data: List[float]) -> Dict[str, List]:
        """Forward propagation through the network."""
        current = input_data
        layer_inputs = [input_data]
        layer_raw_outputs = []

        for i, layer in enumerate(self.layers):
            raw_output = [sum(w * current[k] for k, w in enumerate(weight)) + self.biases[i][j]
                         for j, weight in enumerate(self.weights[i])]

            layer_raw_outputs.append(raw_output)
            activation = layer['activation']
            
            if activation == 'softmax':
                current = self._softmax(raw_output)
            else:
                current = [self._get_activation(x, activation) for x in raw_output]
            
            layer_inputs.append(current)

        return {
            'layer_inputs': layer_inputs,
            'layer_raw_outputs': layer_raw_outputs
        }

    def _back_propagate(self, layer_inputs: List[List[float]], 
                       layer_raw_outputs: List[List[float]], 
                       target: List[float], 
                       loss_function: str) -> List[List[float]]:
        """Backward propagation to compute gradients."""
        output_layer = self.layers[-1]
        output_errors = self._loss_functions[loss_function]['derivative'](
            layer_inputs[-1], target, output_layer['activation']
        )

        layer_errors = [output_errors]

        for i in range(len(self.weights) - 2, -1, -1):
            errors = [0] * self.layers[i]['output_size']

            for j in range(self.layers[i]['output_size']):
                for k in range(self.layers[i + 1]['output_size']):
                    errors[j] += layer_errors[0][k] * self.weights[i + 1][k][j]
                
                activation_deriv = self._get_activation_derivative(
                    layer_raw_outputs[i][j], self.layers[i]['activation']
                )
                if activation_deriv is not None:
                    errors[j] *= activation_deriv

            layer_errors.insert(0, errors)

        return layer_errors

    def _initialize_optimizer(self):
        """Initialize Adam optimizer variables."""
        if not hasattr(self, 'weight_m'):
            self.weight_m = [[[0] * len(row) for row in layer] for layer in self.weights]
            self.weight_v = [[[0] * len(row) for row in layer] for layer in self.weights]
            self.bias_m = [[0] * len(layer) for layer in self.biases]
            self.bias_v = [[0] * len(layer) for layer in self.biases]

    def _adam_update(self, layer_index: int, weight_gradients: List[List[float]], 
                    bias_gradients: List[float], t: int, learning_rate: float):
        """Update weights using Adam optimization."""
        beta1, beta2 = 0.9, 0.999
        epsilon = 1e-8

        for j in range(len(self.weights[layer_index])):
            for k in range(len(self.weights[layer_index][j])):
                g = weight_gradients[j][k]
                self.weight_m[layer_index][j][k] = beta1 * self.weight_m[layer_index][j][k] + (1 - beta1) * g
                self.weight_v[layer_index][j][k] = beta2 * self.weight_v[layer_index][j][k] + (1 - beta2) * g * g

                m_hat = self.weight_m[layer_index][j][k] / (1 - beta1 ** t)
                v_hat = self.weight_v[layer_index][j][k] / (1 - beta2 ** t)

                self.weights[layer_index][j][k] -= (learning_rate * m_hat) / (math.sqrt(v_hat) + epsilon)

            g_bias = bias_gradients[j]
            self.bias_m[layer_index][j] = beta1 * self.bias_m[layer_index][j] + (1 - beta1) * g_bias
            self.bias_v[layer_index][j] = beta2 * self.bias_v[layer_index][j] + (1 - beta2) * g_bias * g_bias

            m_hat_bias = self.bias_m[layer_index][j] / (1 - beta1 ** t)
            v_hat_bias = self.bias_v[layer_index][j] / (1 - beta2 ** t)

            self.biases[layer_index][j] -= (learning_rate * m_hat_bias) / (math.sqrt(v_hat_bias) + epsilon)

    def _sgd_update(self, layer_index: int, weight_gradients: List[List[float]], 
                   bias_gradients: List[float], learning_rate: float):
        """Update weights using SGD optimization."""
        for j in range(len(self.weights[layer_index])):
            for k in range(len(self.weights[layer_index][j])):
                self.weights[layer_index][j][k] -= learning_rate * weight_gradients[j][k]
            self.biases[layer_index][j] -= learning_rate * bias_gradients[j]

    def _update_weights(self, layer_index: int, weight_gradients: List[List[float]], 
                       bias_gradients: List[float], optimizer: str, params: Dict):
        """Apply the chosen optimizer to update weights."""
        if optimizer == 'adam':
            self._adam_update(layer_index, weight_gradients, bias_gradients, 
                            params['t'], params['learning_rate'])
        else:
            self._sgd_update(layer_index, weight_gradients, bias_gradients, params['learning_rate'])

    def train(self, train_set: List[Dict], options: Dict = None) -> Dict:
        """Train the model on a dataset."""
        if options is None:
            options = {}

        epochs = options.get('epochs', 10)
        learning_rate = options.get('learning_rate', 0.212)
        print_every_epochs = options.get('print_every_epochs', 1)
        early_stop_threshold = options.get('early_stop_threshold', 1e-6)
        test_set = options.get('test_set')
        callback = options.get('callback')
        optimizer = options.get('optimizer', 'sgd')
        loss_function = options.get('loss_function', 'mse')

        # Rest of the training code remains the same, but remove await
        if isinstance(train_set[0]['output'], str) or \
           (isinstance(train_set[0]['output'], list) and 
            isinstance(train_set[0]['output'][0], str)):
            train_set = self._preprocess_tags(train_set)

        start = time.time()
        t = 0

        if optimizer == 'adam':
            self._initialize_optimizer()

        last_train_loss = 0
        last_test_loss = None

        for epoch in range(epochs):
            train_error = 0

            for data in train_set:
                t += 1
                forward_result = self._forward_propagate(data['input'])
                layer_errors = self._back_propagate(
                    forward_result['layer_inputs'],
                    forward_result['layer_raw_outputs'],
                    data['output'],
                    loss_function
                )

                for i in range(len(self.weights)):
                    weight_gradients = [[layer_errors[i][j] * forward_result['layer_inputs'][i][k]
                                       for k in range(len(self.weights[i][j]))]
                                      for j in range(len(self.weights[i]))]
                    
                    self._update_weights(i, weight_gradients, layer_errors[i], optimizer, {
                        't': t,
                        'learning_rate': learning_rate
                    })

                train_error += self._loss_functions[loss_function]['loss'](
                    forward_result['layer_inputs'][-1], data['output']
                )

            last_train_loss = train_error / len(train_set)

            if test_set:
                last_test_loss = self._evaluate_test_set(test_set, loss_function)

            if (epoch + 1) % print_every_epochs == 0 and self.debug:
                test_loss_str = f", Test Loss: {last_test_loss:.6f}" if test_set else ""
                print(f"✨ Epoch {epoch + 1}, Train Loss: {last_train_loss:.6f}{test_loss_str}")

            if callback:
                callback(epoch + 1, last_train_loss, last_test_loss)

            if last_train_loss < early_stop_threshold:
                if self.debug:
                    test_loss_str = f" and test loss: {last_test_loss:.6f}" if test_set else ""
                    print(f"🚀 Early stopping at epoch {epoch + 1} "
                          f"with train loss: {last_train_loss:.6f}{test_loss_str}")
                break

        if optimizer == 'adam':
            del self.weight_m
            del self.weight_v
            del self.bias_m
            del self.bias_v

        summary = self._generate_training_summary(start, time.time(), {
            'epochs': epochs,
            'learning_rate': learning_rate,
            'last_train_loss': last_train_loss,
            'last_test_loss': last_test_loss
        })

        self.details = summary
        return summary

    def _preprocess_tags(self, train_set: List[Dict]) -> List[Dict]:
        """Convert categorical outputs to one-hot encoded vectors."""
        unique_tags = list(set(
            tag for item in train_set 
            for tag in (item['output'] if isinstance(item['output'], list) else [item['output']])
        ))

        self.tags = unique_tags

        if not self.layers:
            num_inputs = len(train_set[0]['input'])
            num_classes = len(unique_tags)
            self.layer(num_inputs, math.ceil((num_inputs + num_classes) / 2), "tanh")
            self.layer(math.ceil((num_inputs + num_classes) / 2), num_classes, "softmax")
        return [{
            'input': item['input'],
            'output': [1 if tag in (item['output'] if isinstance(item['output'], list) 
                                  else [item['output']]) else 0
                      for tag in unique_tags]
        } for item in train_set]

    def _evaluate_test_set(self, test_set: List[Dict], loss_function: str) -> float:
        """Calculate the loss on the test set."""
        total_error = sum(
            self._loss_functions[loss_function]['loss'](
                self.predict(data['input'], False),
                data['output']
            )
            for data in test_set
        )
        return total_error / len(test_set)

    def _generate_training_summary(self, start: float, end: float, 
                                 training_info: Dict) -> Dict[str, Any]:
        """Create a summary of the training process."""
        total_params = sum(
            len(layer) * len(layer[0]) + len(self.biases[i])
            for i, layer in enumerate(self.weights)
        )

        return {
            'parameters': total_params,
            'training': {
                'loss': training_info['last_train_loss'],
                'testloss': training_info['last_test_loss'],
                'time': end - start,
                'epochs': training_info['epochs'],
                'learning_rate': training_info['learning_rate'],
            },
        }

    def predict(self, input_data: List[float], tags: bool = True) -> Union[List[float], 
                                                                         List[Dict[str, Union[str, float]]]]:
        """Make predictions using the trained model."""
        forward_result = self._forward_propagate(input_data)
        output = forward_result['layer_inputs'][-1]

        if (hasattr(self, 'tags') and 
            self.layers[-1]['activation'] == "softmax" and 
            tags):
            return sorted([{
                'tag': self.tags[idx],
                'probability': prob
            } for idx, prob in enumerate(output)],
                key=lambda x: x['probability'],
                reverse=True)

        return output

    def info(self, info_updates: Dict[str, str]):
        """Update model metadata."""
        self.details['info'] = info_updates
