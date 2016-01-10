
#include "DynamicTrainer.hpp"
#include <cassert>
#include <iostream>
#include <random>


DynamicTrainer::DynamicTrainer(float startLearnRate,
                               float maxLearnRate,
                               float momentumAmount,
                               unsigned stochasticSamples) :
    startLearnRate(startLearnRate),
    maxLearnRate(maxLearnRate),
    momentumAmount(momentumAmount),
    stochasticSamples(stochasticSamples) {

  assert(startLearnRate > 0.0f);
  assert(maxLearnRate > 0.0f);
  assert(momentumAmount >= 0.0f && momentumAmount < 1.0f);
  assert(stochasticSamples > 0);

  random_device rd;
  this->rnd = mt19937(rd());
}

void DynamicTrainer::Train(
    Network &network, vector<TrainingSample> &trainingSamples, unsigned iterations) {

  shuffle(trainingSamples.begin(), trainingSamples.end(), rnd);
  curSamplesIndex = 0;

  curLearnRate = startLearnRate;
  prevSampleError = 0.0f;

  Tensor momentum;
  for (unsigned i = 0; i < iterations; i++) {
    vector<TrainingSample> samples = getStochasticSamples(trainingSamples);
    pair<Tensor, float> gradientError = network.ComputeGradient(samples);

    gradientError.first *= -curLearnRate;

    if (i == 0) {
      momentum = gradientError.first;
    } else {
      momentum = momentum*momentumAmount + gradientError.first*(1.0f - momentumAmount);
    }

    network.ApplyUpdate(momentum);
    updateLearnRate(i, iterations, gradientError.second);
  }
}

void DynamicTrainer::updateLearnRate(unsigned curIter, unsigned iterations, float sampleError) {
  if (curIter > 0) {
    if (sampleError < prevSampleError) {
      curLearnRate *= 1.1f;
      curLearnRate = min<float>(curLearnRate, maxLearnRate);
    } else {
      curLearnRate *= 0.95f;
    }
  }

  prevSampleError = sampleError;
}

vector<TrainingSample> DynamicTrainer::getStochasticSamples(vector<TrainingSample> &allSamples) {
  unsigned numSamples = min<unsigned>(allSamples.size(), stochasticSamples);

  if ((curSamplesIndex + numSamples) > allSamples.size()) {
    shuffle(allSamples.begin(), allSamples.end(), rnd);
    curSamplesIndex = 0;
  }

  vector<TrainingSample> result;
  result.reserve(numSamples);

  for (unsigned i = curSamplesIndex; i < (curSamplesIndex + numSamples); i++) {
    result.push_back(allSamples[i]);
  }
  curSamplesIndex += numSamples;

  return result;
}
