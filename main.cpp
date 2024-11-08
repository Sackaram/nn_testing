

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>


// TODO
// loss funcions
// regressions vs class
// batch vs sgr
// why should the network even have activation func pointers???
// setting gradiant clipping, no booelan. Looks bad ??

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(-0.05, 0.05);


std::string formatNr(long nr) {
    if (std::to_string(nr).length() < 4) { return std::to_string(nr); }

    std::string newString = std::to_string(nr);
    for (size_t i = newString.length() - 3; i >= 1; i -= 3) {
        newString.insert(i, 1, '.');
    }
    return newString;
}


class NeuralNetwork {

    struct Node {
        std::vector<float> weights;
        float bias;
        float output;
        float delta;
        float (*activFunc)(float);
        float (*activFuncDeriv)(float);

        Node() {}
        Node(float (*activFunc)(float), float (*activFuncDeriv)(float))
            : output(0), delta(0), activFunc(activFunc), activFuncDeriv(activFuncDeriv), bias(dis(gen)) {}

        Node(float (*activFunc)(float), float (*activFuncDeriv)(float), int nrWeights)
            : output(0), delta(0), activFunc(activFunc), activFuncDeriv(activFuncDeriv), bias(dis(gen)) {
            for (size_t i = 0; i < nrWeights; i++) {
                weights.push_back(dis(gen));
            }
        }
        void applyActivation() { float temp = activFunc(output); }
        float setDelta(float error) { return error * activFuncDeriv(output); }
        // operator float() { return output; }
    };

    struct Layer {
        std::vector<Node> nodes;
        Layer(int size) : nodes(size) {}
        void clearInputWeights() {
            for (Node &nod : nodes) {
                nod.weights.clear();
            }
        }
        Node &operator[](int node) { return nodes[node]; }
    };

    struct ActivFunc {

        static float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
        static float sigmoid_derivative(float x) {
            float sig = sigmoid(x);
            return sig * (1.0f - sig);
        }

        static float tanh(float x) { return std::tanh(x); }
        static float tanh_derivative(float x) {
            float tanh_x = tanh(x);
            return 1.0f - tanh(x) * tanh(x);
        }

        static float relu(float x) { return std::max(0.0f, x); }
        static float relu_derivative(float x) { return x > 0.0f ? 1.0 : 0.0f; }

        static float leaky_relu(float x) {
            float alpha = 0.01f;
            return x > 0.0f ? x : alpha * x;
        }

        static float leaky_relu_derivative(float x) {
            float alpha = 0.01f;
            return x > 0.0f ? 1.0f : alpha;
        }

        // static float nothing(float x) { return x; }
    };


    struct LossFunction {
        const float minVal = 1e-15f;                                            // better name?
        float MAE(std::vector<float> &target, std::vector<float> &prediction) { // change name?
            assert(target.size() == prediction.size());
            float sum = 0;
            for (size_t i = 0; i < target.size(); i++) {
                sum += std::abs(target[i] - prediction[i]);
            }
            return sum / target.size();
        }

        float MSE(std::vector<float> &target, std::vector<float> &prediction) {
            assert(target.size() == prediction.size());
            float sum = 0;
            for (size_t i = 0; i < target.size(); i++) {
                float temp = target[i] - prediction[i];
                temp = temp * temp;
                sum += temp;
            }
            return sum / target.size();
        }

        float crossEntropy(std::vector<float> &target, std::vector<float> &prediction) {
            assert(target.size() == prediction.size());
            float sum = 0;
            for (size_t i = 0; i < target.size(); i++) {
                sum += target[i] * std::log(std::max(prediction[i], minVal));
            }
            return -sum;
        }

        float binaryCrossEntropy(std::vector<float> &target, std::vector<float> &prediction) {
            assert(target.size() == prediction.size());
            float sum = 0;
            for (size_t i = 0; i < target.size(); i++) {
                sum +=
                    target[i] * std::log(std::max(prediction[i], minVal)) + (1 - target[i]) * std::log(std::max(1 - prediction[i], minVal));
            }
            return -(sum / target.size());
        }

        float softmaxLoss(const std::vector<float> &target, const std::vector<float> &prediction, size_t nrClasses) {
            assert(target.size() == prediction.size());

            float sum = 0.0;
            for (size_t i = 0; i < nrClasses; i++) {
                float predictedProb = std::max(prediction[i], minVal);
                sum += target[i] * std::log(predictedProb);
            }

            return -sum;
        }
    };


  private:
    float LR;
    float error;
    int iterations;
    bool gradientClipping;
    float gradiantTreshold;
    std::vector<Layer> layers;
    std::vector<float> lastErrors;
    std::vector<float> input;
    std::vector<float> target;
    std::vector<float> cumulativeHistory;
    float cumulativeError;
    float threshold;
    float bestCumulative;
    float (*activFunc)(float);
    float (*activFuncDeriv)(float);
    float (*outputActivFunc)(float);
    float (*outputActivFuncDeriv)(float);
    void (*lossFunction)(void);

  public: // set a defualt lr ??
    NeuralNetwork(std::vector<int> dimentions, float lr, size_t itr = 1000)
        : iterations(itr), gradientClipping(false), LR(lr), lastErrors(200, __FLT_MAX__), gradiantTreshold(0.005),
          bestCumulative(__FLT_MAX__) {

        activFunc = &ActivFunc::leaky_relu;
        activFuncDeriv = &ActivFunc::leaky_relu_derivative;
        outputActivFunc = &ActivFunc::leaky_relu;
        outputActivFuncDeriv = &ActivFunc::leaky_relu_derivative;

        init(dimentions);
    }


    void train(std::vector<float> &input, std::vector<float> &target) {

        assert(input.size() == target.size());
        setNrInputWeights(1);


        size_t printIterations = (iterations + 4) / 5;
        for (size_t i = 0; i < iterations; i++) {
            for (size_t j = 0; j < input.size(); j++) {
                this->input = {input[j]};
                this->target = {target[j]};


                propagate();
                backprop();
                // std::cout << "We are crashing here...." << std::endl;

                // if (i % printIterations == 0) { printInfo(i); }  // <----- this is the crashing part ????
                std::cout << "123..." << std::endl;
                printInfo(i);
            }
            /* calcCumulativeError();
            if (cumulativeError <= threshold) {
                std::cout << "---Stopping,  cumulativeError <= threshold: " << threshold << ". After: " << i << " iterations---"
                          << std::endl;
                return;
            } */
        }
        /*  calcCumulativeError();
         size_t size = lastErrors.size();
         std::cout << "\nThe best cumulative error was: " << bestCumulative << std::endl;
         std::cout << "Meaning, the average error the last: " << size << " propagations was: " << (bestCumulative / size) << std::endl; */
    }

    void train(std::vector<std::vector<float>> &input, std::vector<float> &target) {
        assert(input.size() == target.size());
        setNrInputWeights(input[0].size());

        size_t printIterations = (iterations + 4) / 5;
        for (size_t i = 0; i < iterations; i++) {
            for (size_t j = 0; j < input.size(); j++) {
                this->input = {input[j]};
                this->target = {target[j]};

                propagate();
                backprop();
                if (i % printIterations == 0 || i == iterations - 1) { printInfo(i); }
            }
            calcCumulativeError();
            if (cumulativeError <= threshold) {
                std::cout << "---Stopping,  cumulativeError <= threshold: " << threshold << ". After: " << i << " iterations---"
                          << std::endl;
                return;
            }
            if (std::isnan(getMaxOutput()) || std::isinf(getMaxOutput())) {
                std::cout << "helloeaseadsadsad";
                exit(-1);
            }
        }
        calcCumulativeError();
        size_t size = lastErrors.size();
        std::cout << "\nThe best cumulative error was: " << bestCumulative << std::endl;
        std::cout << "Meaning, the average error the last: " << size << " propagations was: " << (bestCumulative / size) << std::endl;
        std::cout << "Where the largest cumulative error was : " << *std::max_element(lastErrors.begin(), lastErrors.end()) << std::endl;
    }


    void setgradientClipping(std::string input, float value) {
        if (input == "true" || input == "True" || input == "TRUE") {
            gradientClipping = true;
            gradiantTreshold = value;
        } else if (input == "false" || input == "False" || input == "FALSE") {
            gradientClipping = false;
        } else {
            std::cout << "\nerror setting clipping boolean..\n" << std::endl;
            exit(-1);
        }
    }

    void setActivationFunc(std::string name) {
        if (name == "sigmoid") {
            activFunc = &ActivFunc::sigmoid;
            activFuncDeriv = &ActivFunc::sigmoid_derivative;
        } else if (name == "tanh") {
            activFunc = &ActivFunc::tanh;
            activFuncDeriv = &ActivFunc::tanh_derivative;

        } else if (name == "relu") {
            activFunc = &ActivFunc::relu;
            activFuncDeriv = &ActivFunc::relu_derivative;

        } else if (name == "leakyRelu") {
            activFunc = &ActivFunc::leaky_relu;
            activFuncDeriv = &ActivFunc::leaky_relu_derivative;

        } else {
            std::cout << "\nerror setting activation function..\n" << std::endl;
            exit(-1);
        }
        setActivationFunc(activFunc, activFuncDeriv);
    }

    void setActivationFunc(float (*func)(float), float (*funcDeriv)(float)) {
        for (size_t i = 0; i < layers.size() - 1; i++) {
            for (size_t j = 0; j < layers[i].nodes.size(); j++) {
                layers[i][j].activFunc = func;
                layers[i][j].activFuncDeriv = funcDeriv;
            }
        }
    }


    float predict(std::vector<float> &input) {
        this->input = input;
        propagate();
        return getMaxOutput();
    }


    float predict(float input) {
        this->input = {input};
        propagate();
        return getMaxOutput();
    }


    void printWeights() {
        std::cout << "\nWeights, biases, and deltas\n";
        for (size_t i = 0; i < layers.size(); i++) {
            std::cout << "Layer: " << i << " -> ";
            for (size_t j = 0; j < layers[i].nodes.size(); j++) {
                std::cout << "node: " << j << ", weights: ";
                for (size_t k = 0; k < layers[i][j].weights.size(); k++) {
                    std::cout << layers[i][j].weights[k] << " ";
                }
                std::cout << "b: " << layers[i][j].bias << ", d: " << layers[i][j].delta << std::endl;
            }
        }
        std::cout << "\n";
    }


    void printLastErrors() {
        std::cout << "Last " << lastErrors.size() << " errors" << std::endl;
        for (size_t i = 0; i < lastErrors.size(); i++) {
            std::cout << lastErrors[i] << ", ";
        }
        std::cout << "\nCumulativeError: " << cumulativeError << "\n";
    }


    void printLastErrorsSorted() {
        std::vector<float> errors = lastErrors;
        std::cout << "Last " << errors.size() << " errors, sorted" << std::endl;
        std::sort(errors.begin(), errors.end(), std::greater<int>());
        for (size_t i = 0; i < errors.size(); i++) {
            std::cout << errors[i] << ", ";
        }
        std::cout << std::endl;
    }


    void printCumulativeHistory() { // pair it with its index iteration?
        int nrPerRow = 8;
        std::cout << "Cumulative error history :" << std::endl;
        for (size_t i = 0; i < cumulativeHistory.size(); i++) {
            std::cout << cumulativeHistory[i] << ", ";
            if (i % 7 == 0 && i > 0) std::cout << '\n';
        }
        std::cout << std::endl;
    }


    std::vector<std::vector<float>> normalizeInputs(std::vector<std::vector<float>> &inputs) {
        int numValues = inputs[0].size();
        for (int j = 0; j < numValues; ++j) {
            float minVal = inputs[0][j];
            float maxVal = inputs[0][j];

            for (int i = 0; i < inputs.size(); ++i) {
                if (inputs[i][j] < minVal) minVal = inputs[i][j];
                if (inputs[i][j] > maxVal) maxVal = inputs[i][j];
            }

            for (int i = 0; i < inputs.size(); ++i) {
                inputs[i][j] = (inputs[i][j] - minVal) / (maxVal - minVal);
            }
        }
        return inputs;
    }


    std::vector<float> normalizeTarget(std::vector<float> &vect) {
        // any benefit of returning, rather than just void?
        float min_val = *std::min_element(vect.begin(), vect.end());
        float max_val = *std::max_element(vect.begin(), vect.end());

        for (int i = 0; i < vect.size(); ++i) {
            vect[i] = (vect[i] - min_val) / (max_val - min_val);
        }
        return vect;
    }


    void setRegularization() {
        // TODO
    }


  private:
    void init(const std::vector<int> &dimensions) {
        for (size_t i = 0; i < dimensions.size(); i++) {
            Layer layer(dimensions[i]);
            for (size_t j = 0; j < dimensions[i]; j++) {
                if (i == 0) {
                    layer[j] = Node(activFunc, activFuncDeriv);
                    std::cout << "Got here!" << std::endl; // ????
                } else if (i == dimensions.size() - 1) {
                    layer[j] = Node(outputActivFunc, outputActivFuncDeriv);
                    std::cout << "Got here2!" << std::endl; // ????
                } else {
                    layer[j] = Node(activFunc, activFuncDeriv, dimensions[i - 1]);
                    std::cout << "Got here3!" << std::endl; // ????
                }
            }
            layers.push_back(layer);
        }

        /*  for (size_t i = 0; i < lastErrors.size(); i++) {
             lastErrors[i] = __FLT_MAX__;
         } */
    }

    void setNrInputWeights(int nrInputs) {
        layers[0].clearInputWeights();
        for (size_t i = 0; i < layers[0].nodes.size(); i++) {
            for (size_t j = 0; j < nrInputs; j++) {
                layers[0][i].weights.push_back(dis(gen));
            }
        }
    }


    void propagate() {
        for (size_t layer = 0; layer < layers.size(); layer++) {
            for (size_t node = 0; node < layers[layer].nodes.size(); node++) {
                float sum = 0;
                for (size_t k = 0; k < layers[layer][node].weights.size(); k++) {
                    if (layer == 0) {
                        sum += layers[layer][node].weights[k] * input[k];
                    } else {
                        sum += layers[layer][node].weights[k] * layers[layer - 1][k].output;
                    }
                }
                sum += layers[layer][node].bias;
                layers[layer][node].output = sum;
                layers[layer][node].applyActivation();
            }
        }
        calcError();
    }

    void backprop() {

        for (size_t i = 0; i < layers.back().nodes.size(); i++) {
            layers.back().nodes[i].delta = target[i] - layers.back().nodes[i].output;
            if (gradientClipping) { clipGradiant(layers.back().nodes[i].delta); }
        }

        for (int i = layers.size() - 2; i >= 0; i--) {
            for (size_t j = 0; j < layers[i].nodes.size(); j++) {
                float error = 0;
                for (size_t k = 0; k < layers[i + 1].nodes.size(); k++) {
                    std::cout << "j: " << j << " i: " << i << std::endl;
                    error += layers[i + 1][k].weights[j] * layers[i + 1][k].delta;
                }
                layers[i][j].setDelta(error);
                if (gradientClipping) { clipGradiant(layers[i][j].delta); }
            }
        }

        for (size_t i = 0; i < layers.size(); i++) {
            for (size_t j = 0; j < layers[i].nodes.size(); j++) {
                if (i == 0) {
                    for (size_t k = 0; k < input.size(); k++) {
                        layers[i][j].weights[k] += LR * (layers[i][j].delta * input[k]);
                    }
                } else {
                    for (size_t k = 0; k < layers[i - 1].nodes.size(); k++) {
                        layers[i][j].weights[k] += LR * (layers[i][j].delta * layers[i - 1][k].output);
                    }
                }
                layers[i][j].bias += LR * layers[i][j].delta;
            }
        }
    }

    void calcError() {
        error = std::abs(target[0] - getMaxOutput());
        static int index = 0;
        lastErrors[index] = error;
        index = (index + 1) % lastErrors.size();
    }

    void clipGradiant(float &value) {
        if (value > gradiantTreshold) { value = gradiantTreshold; }
        if (value < -gradiantTreshold) { value = -gradiantTreshold; }
    }

    float getMaxOutput() {
        /* float max = -__FLT_MAX__;
        for (size_t i = 0; i < layers[layers.size() - 1].nodes.size(); i++) {
            float val = layers[layers.size() - 1][i].output;
            max = (val > max) ? val : max;
        }
        return max; */
        return layers.back().nodes[0].output;
    }

    int getMaxOutputIndex() {
        int index = -1;
        float val = -__FLT_MAX__;
        for (size_t i = 0; i < layers.back().nodes.size(); i++) {
            if (layers.back().nodes[i].output > val) {
                val = layers.back().nodes[i].output;
                index = i;
            }
        }
        return index;
    }

    void calcCumulativeError() {
        float sum = 0;
        for (size_t i = 0; i < lastErrors.size(); i++) {
            sum += lastErrors[i];
        }
        cumulativeError = sum;
        if (bestCumulative > cumulativeError) {
            bestCumulative = cumulativeError;
            cumulativeHistory.push_back(cumulativeError); /* could auto save model here? */
        }
    }

    void printInfo(int itr) {

        std::cout << "Iteration: " << formatNr(itr) << std::endl;

        std::cout << "Input was: ";
        for (size_t i = 0; i < input.size(); i++) {
            std::cout << input[i] << " ";
        }
        std::cout << "\nTarget was: " << target[0] << std::endl;
        std::cout << "Prediction was: ";
        for (size_t i = 0; i < layers.back().nodes.size(); i++) {
            std::cout << layers.back()[i].output << " ";
        }
        std::cout << "\nError was: " << error << "\n\n";
    }
};


int main() {


    std::vector<float> input1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};       // single input
    std::vector<float> target1 = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20}; // target is input * 2


    NeuralNetwork nn1({1}, 0.1, 100);
    // nn1.printWeights();


    nn1.train(input1, target1);

    std::cout << "Program did not crash" << std::endl;

    return 0;
}
