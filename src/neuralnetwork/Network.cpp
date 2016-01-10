
#include "Network.hpp"
#include "../util/Util.hpp"
#include <cassert>
#include <cmath>


static const float INIT_WEIGHT_RANGE = 0.1f;

struct Network::NetworkImpl {
  unsigned numInputs;
  unsigned numOutputs;
  unsigned numLayers;

  Tensor layerWeights;
  vector<Vector> layerOutputs;
  vector<Vector> layerDeltas;


  NetworkImpl(const vector<unsigned> &layerSizes) {
    assert(layerSizes.size() >= 2);
    this->numLayers = layerSizes.size() - 1;
    this->numInputs = layerSizes[0];
    this->numOutputs = layerSizes[layerSizes.size() - 1];

    for (unsigned i = 0; i < numLayers; i++) {
      layerWeights.AddLayer(createLayer(layerSizes[i], layerSizes[i+1]));
    }
  }

  Vector Process(const Vector &input) {
    assert(input.rows() == numInputs);

    layerOutputs.clear();
    Vector nodeValues = input;

    for (unsigned i = 0; i < layerWeights.NumLayers(); i++) {
      nodeValues = getLayerOutput(nodeValues, layerWeights(i));
      layerOutputs.push_back(nodeValues);
    }

    assert(nodeValues.rows() == numOutputs);
    return nodeValues;
  }

  pair<Tensor, float> ComputeGradient(const vector<TrainingSample> &samples) {
    Tensor netGradient = zeroGradient();
    float error = 0.0f;

    for (const auto& sample : samples) {
      pair<Tensor, float> gradientAndError = computeSampleGradient(sample);
      netGradient += gradientAndError.first;
      error += gradientAndError.second;
    }

    float scaleFactor = 1.0f / samples.size();
    netGradient *= scaleFactor;
    error *= scaleFactor;

    return make_pair(netGradient, error);
  }

  void ApplyUpdate(const Tensor &weightUpdates) {
    layerWeights += weightUpdates;
  }

private:

  Matrix createLayer(unsigned inputSize, unsigned layerSize) {
    assert(inputSize > 0 && layerSize > 0);

    unsigned numRows = layerSize;
    unsigned numCols = inputSize + 1; // +1 accounts for bias input

    Matrix result(numRows, numCols);

    for (unsigned r = 0; r < result.rows(); r++) {
      for (unsigned c = 0; c < result.cols(); c++) {
        result(r, c) = Util::RandInterval(-INIT_WEIGHT_RANGE, INIT_WEIGHT_RANGE);
      }
    }

    return result;
  }

  Vector getLayerOutput(const Vector &prevLayer, const Matrix &layerWeights) {
    Vector inputWithBias = getInputWithBias(prevLayer);

    Vector z = layerWeights * inputWithBias;
    for (unsigned i = 0; i < z.rows(); i++) {
      z(i) = activationFunc(z(i));
    }
    return z;
  }

  float activationFunc(float v) {
    return 1.0f / (1.0f + expf(-v));
  }

  Tensor zeroGradient(void) {
    Tensor result = layerWeights;
    for (unsigned i = 0; i < result.NumLayers(); i++) {
      result(i).setZero();
    }
    return result;
  }

  pair<Tensor, float> computeSampleGradient(const TrainingSample &sample) {
    Vector output = Process(sample.input);

    layerDeltas.resize(numLayers);
    layerDeltas[layerDeltas.size() - 1] = output - sample.expectedOutput; // cross entropy error function.

    for (int i = layerDeltas.size() - 2; i >= 0; i--) {
      Matrix noBiasWeights =
          layerWeights(i+1).bottomRightCorner(layerWeights(i+1).rows(), layerWeights(i+1).cols()-1);
      layerDeltas[i] = noBiasWeights.transpose() * layerDeltas[i+1];

      assert(layerDeltas[i].rows() == layerOutputs[i].rows());
      for (unsigned r = 0; r < layerDeltas[i].rows(); r++) {
        float out = layerOutputs[i](r);
        layerDeltas[i](r) *= out * (1.0f - out);
      }
    }

    Tensor weightGradients;
    for (unsigned i = 0; i < numLayers; i++) {
      auto input = getInputWithBias(i == 0 ? sample.input : layerOutputs[i-1]);
      auto inputsT = input.transpose();
      weightGradients.AddLayer(layerDeltas[i] * inputsT);
    }

    float error = 0.0;
    for (unsigned i = 0; i < output.rows(); i++) {
      error += (output(i) - sample.expectedOutput(i)) * (output(i) - sample.expectedOutput(i));
    }
    return make_pair(weightGradients, error);
  }

  Vector getInputWithBias(const Vector &noBiasInput) {
    Vector bias(1);
    bias(0) = 1.0f;

    Vector inputWithBias(noBiasInput.rows() + bias.rows());
    inputWithBias << bias, noBiasInput;
    return inputWithBias;
  }
};


Network::Network(const vector<unsigned> &layerSizes) : impl(new NetworkImpl(layerSizes)) {}
Network::~Network() = default;

Vector Network::Process(const Vector &input) {
  return impl->Process(input);
}

pair<Tensor, float> Network::ComputeGradient(const vector<TrainingSample> &samples) {
  return impl->ComputeGradient(samples);
}

void Network::ApplyUpdate(const Tensor &weightUpdates) {
  impl->ApplyUpdate(weightUpdates);
}

std::ostream& Network::Output(std::ostream& stream) {
  for (unsigned i = 0; i < impl->layerWeights.NumLayers(); i++) {
    for (unsigned r = 0; r < impl->layerWeights(i).rows(); r++) {
      for (unsigned c = 0; c < impl->layerWeights(i).cols(); c++) {
        stream << impl->layerWeights(i)(r, c) << " ";
      }
      stream << endl;
    }
    stream << endl;
  }
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const TrainingSample& ts) {
  stream << ts.input << " : " << ts.expectedOutput;
  return stream;
}
