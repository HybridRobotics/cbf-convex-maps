# cbf-convex-maps
This repository provides a method to enforce control barrier function (CBF) constraints between state-dependent convex sets, defined as smooth and strongly convex maps (see the reference below).

We provide the following features:
- Geometry classes to define state-dependent strongly convex sets (smooth and strongly convex maps).
- Ipopt solver interface to compute the minimum distance between strongly convex maps.
- An ODE solver (the KKT solution ODE) that rapidly computes the minimum distance and KKT solutions along state trajectories. The KKT solutions can be used to compute the distance derivative and thus enforce CBF constraints (see the paper).

For a brief overview on how to use the repository, see the [usage file](https://github.com/HybridRobotics/cbf-convex-maps/blob/main/usage.ipynb).
The core algorithm is implemented in the [CollisionPair class](https://github.com/HybridRobotics/cbf-convex-maps/blob/main/src/collision/collision_pair.cc).

---

### Citing
The technical paper corresponding to this repository is in review (second round in SICON24).

An arXiv version of the paper will be uploaded soon.

---

### Requirements

The following C++ libraries are required:
- `Eigen` (>= 3.4.90; install from the [source](https://eigen.tuxfamily.org/index.php?title=Main_Page))
- `Ipopt` (install from [source](https://coin-or.github.io/Ipopt/INSTALL.html))
- `OSQP` (install from [source](https://osqp.org/docs/get_started/sources.html))
- `OSQPEigen` (install from [source](https://github.com/robotology/osqp-eigen))

The following Python libraries are required to generate the plots and visualizations in the paper (optional):
- `numpy`
- `scipy`
- `matplotlib`
- `meshcat-dev` (install from [source](https://github.com/meshcat-dev/meshcat-python))
- `skimage` (for the marching cubes algorithm)
- `polytope` (for polytope computations)

Testing and benchmarks (optional) are done using the GoogleTest and Google Benchmark libraries.
Code formatting is done via `pre-commit` hooks.

### Build from source

1. Clone the repository:
    ```
    git clone https://github.com/HybridRobotics/cbf-convex-maps.git
    ```

2. Build:
    ```
    cd cbf-convex-maps
    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ../
    ```
    To prevent building tests and benchmarks, use the following cmake command:
    ```
    cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF ../
    ```

    Then, build:
    ```
    cmake --build .
    ```

3. To run the KKT ODE example, use:
     ```
     ./apps/sccbf_kkt_ode
     ```
     To run the CBF example, use:
     ```
     ./apps/sccbf_ma_cbf
     ```

4. (optional) To generate the plots, install the required Python libraries. Then, add the `apps/` directory to `PYTHONPATH`:
     ```
     export PYTHONPATH=$PYTHONPATH:<path-to-source-directory>/apps/
     ```
     Then, run the Python files in `apps/plots/` to generate the plots.

---

### Examples

The examples in `apps/` consider two scenarios for a quadrotor system (see the paper for a complete description).


