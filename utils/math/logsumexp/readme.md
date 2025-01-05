# Log-sum-exp and softmax functions

### About

`logsumexp_a()` and `softmax_a()` evaluate the
log-sum-exp function
```math
lse(x) = \log \sum_{i=1}^n e^{x_i},
```
and the
softmax function
```math
g_j(x) = \frac{e^{x_j}}{e^{lse(x)}},
```
where $x$ is a vector.
Softmax is the derivative of log-sum-exp.

Also provided is `test.m`, which runs some simple tests of the functions.

### Usage

The lines
```
sm = softmax_a(x)
[sm,lse] = softmax_a(x)
lse = logsumexp_a(x)
[lse,sm] = logsumexp_a(x)
```

compute the softmax `sm` and the log-sum-exp `lse` at the vector `x`.
Both functions can compute both quantities because there is a significant
overlap in the computations of the two functions.

### Requirements

The code was developed in MATLAB R2020b and works with versions
back to at least R2016b.

### References

P. Blanchard, D. J. Higham, and N. J. Higham.  
[[https://doi.org/10.1093/imanum/draa038][Computing the Log-Sum-Exp and Softmax Functions]].
IMA J. Numer. Anal., Advance access, 2020.

### License

See `license.txt` for licensing information.
