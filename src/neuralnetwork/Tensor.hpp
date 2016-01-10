#pragma once

#include "../common/Common.hpp"
#include "../common/Math.hpp"


class Tensor {
public:

  unsigned NumLayers(void) const;
  void AddLayer(const Matrix &m);

  Matrix& operator()(unsigned index);
  const Matrix& operator()(unsigned index) const;

  Tensor operator+(const Tensor &t) const;
  Tensor operator-(const Tensor &t) const;
  Tensor operator*(float s) const;
  Tensor operator/(float s) const;

  Tensor& operator+=(const Tensor &t);
  Tensor& operator-=(const Tensor &t);
  Tensor& operator*=(float s);
  Tensor& operator/=(float s);

private:
  vector<Matrix> data;
};
