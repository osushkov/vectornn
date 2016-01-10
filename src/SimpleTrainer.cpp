
#include "SimpleTrainer.hpp"
#include <cassert>


SimpleTrainer::SimpleTrainer(float startLearnRate, float endLearnRate, unsigned stochasticSamples) :
    startLearnRate(startLearnRate),
    endLearnRate(endLearnRate),
    stochasticSamples(stochasticSamples) {

  assert(startLearnRate > endLearnRate);
  assert(endLearnRate >= 0.0f);
  assert(stochasticSamples > 0);
}

void SimpleTrainer::Train(
    Network &network, vector<TrainingSample> &trainingSamples, unsigned iterations) {

  random_shuffle(trainingSamples.begin(), trainingSamples.end());
  curSamplesIndex = 0;

  for (unsigned i = 0; i < iterations; i++) {
    float lr = getLearnRate(i, iterations);

    vector<TrainingSample> samples = getStochasticSamples(trainingSamples);
    pair<Tensor, float> gradientError = network.ComputeGradient(samples);
    network.ApplyUpdate(gradientError.first * -lr);
  }
}

float SimpleTrainer::getLearnRate(unsigned curIter, unsigned iterations) {
  return startLearnRate + (endLearnRate - startLearnRate) * curIter / (float) iterations;
}

vector<TrainingSample> SimpleTrainer::getStochasticSamples(vector<TrainingSample> &allSamples) {
  unsigned numSamples = min<unsigned>(allSamples.size(), stochasticSamples);

  if ((curSamplesIndex + numSamples) >= allSamples.size()) {
    random_shuffle(allSamples.begin(), allSamples.end());
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
