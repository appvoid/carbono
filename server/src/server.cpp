#include <httplib.h>
#include "json.hpp"
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <map>
#include <iostream>
#include <memory>

using json = nlohmann::json;

class CarbonoInference {
private:
    struct Layer {
        int inputSize;
        int outputSize;
        std::string activation;
    };

    std::vector<Layer> layers;
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> biases;
    std::vector<std::string> tags;
    bool debug;

    // Activation functions
    std::map<std::string, std::function<double(double)>> activationFunctions = {
        {"tanh", [](double x) { return std::tanh(x); }},
        {"sigmoid", [](double x) { return 1.0 / (1.0 + std::exp(-x)); }},
        {"relu", [](double x) { return std::max(0.0, x); }},
        {"selu", [](double x) {
            const double alpha = 1.67326;
            const double scale = 1.0507;
            return x > 0 ? scale * x : scale * alpha * (std::exp(x) - 1);
        }}
    };

    std::vector<double> softmax(const std::vector<double>& x) {
        std::vector<double> result(x.size());
        double sum = 0.0;
        double maxVal = *std::max_element(x.begin(), x.end());
        
        for (const double& val : x) {
            sum += std::exp(val - maxVal);
        }
        
        for (size_t i = 0; i < x.size(); i++) {
            result[i] = std::exp(x[i] - maxVal) / sum;
        }
        return result;
    }

    std::vector<std::vector<double>> matrixMultiply(
        const std::vector<std::vector<double>>& a,
        const std::vector<std::vector<double>>& b
    ) {
        std::vector<std::vector<double>> result(a.size(), std::vector<double>(b[0].size(), 0));
        for (size_t i = 0; i < a.size(); i++) {
            for (size_t j = 0; j < b[0].size(); j++) {
                for (size_t k = 0; k < b.size(); k++) {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        return result;
    }

    struct ForwardResult {
        std::vector<std::vector<double>> layerInputs;
        std::vector<std::vector<double>> layerRawOutputs;
    };

    ForwardResult forwardPropagate(const std::vector<double>& input) {
        ForwardResult result;
        std::vector<double> current = input;
        result.layerInputs.push_back(input);

        for (size_t i = 0; i < weights.size(); i++) {
            std::vector<double> rawOutput(layers[i].outputSize, 0.0);
            
            // Calculate raw output
            for (size_t j = 0; j < layers[i].outputSize; j++) {
                for (size_t k = 0; k < layers[i].inputSize; k++) {
                    rawOutput[j] += weights[i][j][k] * current[k];
                }
                rawOutput[j] += biases[i][j];
            }
            
            result.layerRawOutputs.push_back(rawOutput);

            // Apply activation function
            if (layers[i].activation == "softmax") {
                current = softmax(rawOutput);
            } else {
                current.resize(rawOutput.size());
                for (size_t j = 0; j < rawOutput.size(); j++) {
                    current[j] = activationFunctions[layers[i].activation](rawOutput[j]);
                }
            }
            
            result.layerInputs.push_back(current);
        }

        return result;
    }

    bool loadModelFromJson(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open model file: " + filename);
        }

        json modelJson;
        file >> modelJson;
        
        // Load layers
        for (const auto& layer : modelJson["layers"]) {
            layers.push_back({
                layer["inputSize"],
                layer["outputSize"],
                layer["activation"]
            });
        }

        // Load weights
        weights = modelJson["weights"].get<std::vector<std::vector<std::vector<double>>>>();
        
        // Load biases
        biases = modelJson["biases"].get<std::vector<std::vector<double>>>();
        
        // Load tags if they exist
        if (modelJson.contains("tags")) {
            tags = modelJson["tags"].get<std::vector<std::string>>();
        }

        return true;
    }

    bool loadModelFromBinary(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open model file: " + filename);
        }

        // Read header
        uint32_t metadataLength, metadataPadding, weightLength, biasLength;
        file.read(reinterpret_cast<char*>(&metadataLength), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&metadataPadding), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&weightLength), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&biasLength), sizeof(uint32_t));

        // Read metadata
        std::vector<char> metadataBuffer(metadataLength);
        file.read(metadataBuffer.data(), metadataLength);
        
        // Skip padding
        file.seekg(metadataPadding, std::ios::cur);

        // Parse metadata
        json metadata = json::parse(std::string(metadataBuffer.begin(), metadataBuffer.end()));
        
        // Load layers
        for (const auto& layer : metadata["layers"]) {
            layers.push_back({
                layer["inputSize"],
                layer["outputSize"],
                layer["activation"]
            });
        }

        // Load tags if they exist
        if (metadata.contains("tags")) {
            tags = metadata["tags"].get<std::vector<std::string>>();
        }

        // Read weights
        std::vector<float> weightBuffer(weightLength);
        file.read(reinterpret_cast<char*>(weightBuffer.data()), weightLength * sizeof(float));

        // Read biases
        std::vector<float> biasBuffer(biasLength);
        file.read(reinterpret_cast<char*>(biasBuffer.data()), biasLength * sizeof(float));

        // Reconstruct weights and biases based on layer info
        size_t weightIndex = 0;
        weights.resize(layers.size());
        for (size_t i = 0; i < layers.size(); i++) {
            weights[i].resize(layers[i].outputSize);
            for (size_t j = 0; j < layers[i].outputSize; j++) {
                weights[i][j].resize(layers[i].inputSize);
                for (size_t k = 0; k < layers[i].inputSize; k++) {
                    weights[i][j][k] = weightBuffer[weightIndex++];
                }
            }
        }

        size_t biasIndex = 0;
        biases.resize(layers.size());
        for (size_t i = 0; i < layers.size(); i++) {
            biases[i].resize(layers[i].outputSize);
            for (size_t j = 0; j < layers[i].outputSize; j++) {
                biases[i][j] = biasBuffer[biasIndex++];
            }
        }

        return true;
    }

public:
    CarbonoInference(bool debugMode = false) : debug(debugMode) {}

    bool loadModel(const std::string& filename) {
        std::string extension = filename.substr(filename.find_last_of(".") + 1);
        if (extension == "json") {
            return loadModelFromJson(filename);
        } else if (extension == "uai") {
            return loadModelFromBinary(filename);
        } else {
            throw std::runtime_error("Unsupported file format: " + extension);
        }
    }

    json predict(const std::vector<double>& input) {
        auto result = forwardPropagate(input);
        auto output = result.layerInputs.back();
        
        json response;
        if (!tags.empty() && layers.back().activation == "softmax") {
            json predictions = json::array();
            for (size_t i = 0; i < output.size(); i++) {
                predictions.push_back({
                    {"tag", tags[i]},
                    {"probability", output[i]}
                });
            }
            // Sort by probability
            std::sort(predictions.begin(), predictions.end(),
                [](const json& a, const json& b) {
                    return a["probability"] > b["probability"];
                });
            response["predictions"] = predictions;
        } else {
            response["output"] = output;
        }
        
        return response;
    }
};

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <port>" << std::endl;
        return 1;
    }

    const std::string modelPath = argv[1];
    const int port = std::stoi(argv[2]);

    try {
        // Initialize model
        auto model = std::make_unique<CarbonoInference>(true);
        model->loadModel(modelPath);
        std::cout << "Model loaded successfully" << std::endl;

        // Initialize server
        httplib::Server server;

        // Health check endpoint
        server.Get("/health", [](const httplib::Request&, httplib::Response& res) {
            res.set_content("{\"status\":\"ok\"}", "application/json");
        });

        // Prediction endpoint
        server.Post("/predict", [&model](const httplib::Request& req, httplib::Response& res) {
            try {
                // Parse request
                json requestJson = json::parse(req.body);
                
                // Validate input
                if (!requestJson.contains("input") || !requestJson["input"].is_array()) {
                    res.status = 400;
                    res.set_content("{\"error\":\"Invalid input format\"}", "application/json");
                    return;
                }

                // Convert input to vector
                std::vector<double> input = requestJson["input"].get<std::vector<double>>();
                
                // Get prediction
                json prediction = model->predict(input);
                
                // Send response
                res.set_content(prediction.dump(), "application/json");

            } catch (const std::exception& e) {
                res.status = 500;
                json error = {{"error", e.what()}};
                res.set_content(error.dump(), "application/json");
            }
        });

        std::cout << "Server starting on port " << port << std::endl;
        server.listen("0.0.0.0", port);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}