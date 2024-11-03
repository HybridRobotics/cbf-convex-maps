#ifndef SCCBF_COLLISION_COLLISION_INFO_H_
#define SCCBF_COLLISION_COLLISION_INFO_H_

#include <memory>

#include "sccbf/data_types.h"


namespace sccbf {

struct CollisionInfo {
  MatrixXd M;

  CollisionInfo(int nz);

  CollisionInfo(const MatrixXd& M);
};

inline CollisionInfo::CollisionInfo(int nz): M(nz, nz) {}

inline CollisionInfo::CollisionInfo(const MatrixXd& M): M(M) {
  assert(M.rows() == M.cols());
}

} // namespace sccbf

#endif // SCCBF_COLLISION_COLLISION_INFO_H_
