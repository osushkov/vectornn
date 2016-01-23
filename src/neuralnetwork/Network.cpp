
#include "Network.hpp"
#include "../util/Util.hpp"
#include "../common/ThreadPool.hpp"
#include <cassert>
#include <cmath>
#include <future>
#include <iostream>


static const float INIT_WEIGHT_RANGE = 0.1f;
// static ThreadPool threadPool(2);

struct NetworkContext {
  vector<Vector> layerOutputs;
  vector<Vector> layerDeltas;
};

struct Network::NetworkImpl {
  unsigned numInputs;
  unsigned numOutputs;
  unsigned numLayers;

  Tensor layerWeights;


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

    NetworkContext ctx;
    return process(input, ctx);
  }

  pair<Tensor, float> ComputeGradient(const TrainingProvider &samplesProvider) {
    auto gradient = make_pair(zeroGradient(), 0.0f);
    Tensor& netGradient{gradient.first};
    float& error{gradient.second};

    NetworkContext ctx;
    for (unsigned i = 0; i < samplesProvider.NumSamples(); i++) {
      pair<Tensor, float> gradientAndError =
          computeSampleGradient(samplesProvider.GetSample(i), ctx);
      netGradient += gradientAndError.first;
      error += gradientAndError.second;
    }

    float scaleFactor = 1.0f / samplesProvider.NumSamples();
    netGradient *= scaleFactor;
    error *= scaleFactor;

    return gradient;
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

  Vector process(const Vector &input, NetworkContext &ctx) {
    assert(input.rows() == numInputs);

    ctx.layerOutputs.clear();
    Vector nodeValues = input;

    for (unsigned i = 0; i < layerWeights.NumLayers(); i++) {
      nodeValues = getLayerOutput(nodeValues, layerWeights(i));
      ctx.layerOutputs.push_back(nodeValues);
    }

    assert(nodeValues.rows() == numOutputs);
    return nodeValues;
  }

  Vector getLayerOutput(const Vector &prevLayer, const Matrix &layerWeights) {
    // Vector inputWithBias = getInputWithBias(prevLayer);

    Vector z = layerWeights.topRightCorner(layerWeights.rows(), layerWeights.cols()-1) * prevLayer; //inputWithBias;
    for (unsigned i = 0; i < layerWeights.rows(); i++) {
      z(i) += layerWeights(i, 0);
    }

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

  pair<Tensor, float> computeGradient(
      const vector<TrainingSample> &samples, unsigned start, unsigned end) {
    Tensor netGradient = zeroGradient();
    float error = 0.0f;

    for (unsigned i = start; i < end; i++) {
      NetworkContext ctx;
      pair<Tensor, float> gradientAndError = computeSampleGradient(samples[i], ctx);
      netGradient += gradientAndError.first;
      error += gradientAndError.second;
    }

    float scaleFactor = 1.0f / (end - start);
    netGradient *= scaleFactor;
    error *= scaleFactor;

    return make_pair(netGradient, error);
  }

  pair<Tensor, float> computeSampleGradient(const TrainingSample &sample, NetworkContext &ctx) {
    Vector output = process(sample.input, ctx);

    ctx.layerDeltas.resize(numLayers);
    ctx.layerDeltas[ctx.layerDeltas.size() - 1] = output - sample.expectedOutput; // cross entropy error function.

    for (int i = ctx.layerDeltas.size() - 2; i >= 0; i--) {
      Matrix noBiasWeights =
          layerWeights(i+1).bottomRightCorner(layerWeights(i+1).rows(), layerWeights(i+1).cols()-1);
      ctx.layerDeltas[i] = noBiasWeights.transpose() * ctx.layerDeltas[i+1];

      assert(ctx.layerDeltas[i].rows() == ctx.layerOutputs[i].rows());
      for (unsigned r = 0; r < ctx.layerDeltas[i].rows(); r++) {
        float out = ctx.layerOutputs[i](r);
        ctx.layerDeltas[i](r) *= out * (1.0f - out);
      }
    }

    Tensor weightGradients;
    for (unsigned i = 0; i < numLayers; i++) {
      auto input = getInputWithBias(i == 0 ? sample.input : ctx.layerOutputs[i-1]);
      auto inputsT = input.transpose();
      weightGradients.AddLayer(ctx.layerDeltas[i] * inputsT);
    }

    float error = 0.0;
    for (unsigned i = 0; i < output.rows(); i++) {
      error += (output(i) - sample.expectedOutput(i)) * (output(i) - sample.expectedOutput(i));
    }
    return make_pair(weightGradients, error);
  }

  Vector getInputWithBias(const Vector &noBiasInput) {
    Vector result(noBiasInput.rows() + 1);
    result(0) = 1.0f;
    result.bottomRightCorner(noBiasInput.rows(), 1) = noBiasInput;
    return result;
  }
};


Network::Network(const vector<unsigned> &layerSizes) : impl(new NetworkImpl(layerSizes)) {}
Network::~Network() = default;

Vector Network::Process(const Vector &input) {
  return impl->Process(input);
}

pair<Tensor, float> Network::ComputeGradient(const TrainingProvider &samplesProvider) {
  return impl->ComputeGradient(samplesProvider);
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
