#ifndef SCCBF_LEMKE_H_
#define SCCBF_LEMKE_H_

#include <cstdint>

#include "sccbf/data_types.h"

namespace sccbf {

enum class LCPStatus : uint8_t {
  OPTIMAL,
  INFEASIBLE,
  MAX_ITER_REACHED,
};

// Solve: w = Mz + q, w >= 0, z >= 0, <w, z> = 0.
// Uses "Murty, Katta G., and Feng-Tien Yu. Linear complementarity, linear and
// nonlinear programming. Vol. 3. Berlin: Heldermann, 1988".
LCPStatus solve_LCP(const MatrixXd& M, const VectorXd& q, VectorXd& z);

}  // namespace sccbf

#endif  // SCCBF_LEMKE_H_
