#ifndef SCCBF_LEMKE_H_
#define SCCBF_LEMKE_H_

#include <cstdint>

#include "sccbf/data_types.h"

namespace sccbf {

enum class LcpStatus : uint8_t;

struct LcpOptions;

// Solve: w = Mz + q, w >= 0, z >= 0, <w, z> = 0.
// Uses "Murty, Katta G., and Feng-Tien Yu. Linear complementarity, linear and
// nonlinear programming. Vol. 3. Berlin: Heldermann, 1988".
LcpStatus SolveLcp(const MatrixXd& M, const VectorXd& q, VectorXd& z,
                   const LcpOptions& opt);

}  // namespace sccbf

#endif  // SCCBF_LEMKE_H_
