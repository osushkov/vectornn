#pragma once

#include "../common/Common.hpp"
#include "../common/Math.hpp"
#include "Tensor.hpp"
#include <vector>

struct TrainingSample {
  Vector input;
  Vector expectedOutput;

  TrainingSample(const Vector &input, const Vector &expectedOutput) :
      input(input), expectedOutput(expectedOutput) {}
};

std::ostream& operator<<(std::ostream& stream, const TrainingSample& ts);


class Network {
public:
  Network(const vector<unsigned> &layerSizes);
  virtual ~Network();

  Vector Process(const Vector &input);
  pair<Tensor, float> ComputeGradient(const vector<TrainingSample> &samples);
  void ApplyUpdate(const Tensor &weightUpdates);

  std::ostream& Output(std::ostream& stream);

private:
  struct NetworkImpl;
  uptr<NetworkImpl> impl;
};
