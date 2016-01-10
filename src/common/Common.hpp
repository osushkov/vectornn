
#pragma once

#include <memory>
#include <algorithm>
#include "Maybe.hpp"


using namespace std;

template<typename T>
using uptr = unique_ptr<T>;

template<typename T>
using sptr = shared_ptr<T>;

template<typename T>
using wptr = weak_ptr<T>;

template<typename T>
inline shared_ptr<T> u2sptr(unique_ptr<T> &rhs) {
  return shared_ptr<T>(move(rhs));
}

template<typename T>
inline shared_ptr<T> u2sptr(unique_ptr<T> &&rhs) {
  return shared_ptr<T>(move(rhs));
}

template<class Container, class Function>
Function for_each(Container &container, Function fn) {
  return for_each(container.begin(), container.end(), fn);
}

template<typename Container, class UnaryPredicate>
Maybe<typename Container::value_type> find_if(Container &container, UnaryPredicate fn) {
  auto it = find_if(container.begin(), container.end(), fn);

  if (it == container.end()) {
    return Maybe<typename Container::value_type>::none;
  } else {
    return Maybe<typename Container::value_type>(*it);
  }
}
