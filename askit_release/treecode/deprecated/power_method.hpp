
#ifndef ASKIT_POWER_METHOD_HPP_
#define ASKIT_POWER_METHOD_HPP_

#include <vector>

namespace askit {

  // Estimates the two-norm of the approximate K computed by the ASKIT algorithm
  // (provided as a template)
template <class TAlg>
double PowerMethod(TAlg& alg, double acc, int max_iterations);

} // namespace


#include "power_method_impl.hpp"


#endif
