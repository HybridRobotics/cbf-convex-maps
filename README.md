# cbf-convex-maps
A repository for enforcing control barrier function constraints between strongly convex maps.

We provide:
- Geometry classes to define state-dependent strongly convex maps.
- Solver interface to compute the minimum distance between strongly convex maps.
- An ODE solver that integrates the KKT ODE to quickly compute minimum distance and KKT solutions.

### Citing
The technical paper corresponding to this repository is in review.

### Requirements

- MATLAB (tested on 2023b)
- [Optimization Toolbox](https://www.mathworks.com/products/optimization.html)
- [Multi-Parametric Toolbox](https://www.mpt3.org/)
- [Symbolic Math Toolbox](https://www.mathworks.com/products/symbolic.html) (Optional)
- Stable Log-Sum-Exp function (included in the code) (see P. Blanchard, D. J. Higham, and N. J. Higham, [Computing the Log-Sum-Exp and Softmax Functions.](https://doi.org/10.1093/imanum/draa038)
IMA J. Numer. Anal., Advance access, 2020.)

### Usage

1. Create a geometry class by inheriting from the `AbstractConvexSet` class and implementing the required functions.
2. Create a dynamical system by inheriting from the `DynamicalSystem` class and implementing the dynamics functions.
3. Instantiate a `Robot` object by providing the `DynamicalSystem` and (possibly multiple) `AbstractConvexSet` objects.
4. Solve for the initial minimum distance and KKT solutions using `minimum_distance()`.
5. Subsequent minimum distances and KKT solutions can be obtained using `minimum_distance_step()`.

### Examples

Example use cases are provided in the `examples\` directory.
