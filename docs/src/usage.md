```@meta
EditURL = "../../examples/usage.jl"
```

# Usage

Here we demonstrate the basic usage for the [SmoQyKPMCore](https://github.com/SmoQySuite/SmoQyKPMCore.jl) package.
Let us consider a 1D chain tight-binding model
```math
\hat{H} = -t \sum_{i,\sigma} (\hat{c}_{i+1,\sigma}^\dagger \hat{c}_{i,\sigma} + {\rm H.c.})
        = \sum_\sigma \hat{c}_{i,\sigma}^\dagger [H_{i,j}] \hat{c}_{j,\sigma},
```
where ``\hat{c}_{i,\sigma}^\dagger \ (\hat{c}_{i,\sigma})`` creates (annihilates) a spin-``\sigma`` electron on site ``i`` in the lattice,
and ``t`` is the nearest-neighbor hopping amplitude.
With ``H`` the Hamiltonian matrix, the corresponding density matrix is given by ``\rho = f(H)`` where
```math
f(\epsilon) = \frac{1}{1+e^{\beta (\epsilon-\mu)}} = \frac{1}{2} \left[ 1 + \tanh\left(\tfrac{\beta(\epsilon-\mu)}{2}\right) \right]
```
is the Fermi function, with ``\beta = 1/T`` is the inverse temperature and ``\mu`` is the chemical potential.
Here we will applying the kernel polynomial method (KPM) algorithm to approximate the density matrix ``\rho`` by
a Chebyshev polynomial expansion.

Let us begin by importing the packages we need.

````@example usage
using LinearAlgebra
using SmoQyKPMCore
````

Next, let us define the relevant system parameters.

````@example usage
# Nearest-neighbor hopping amplitude.
t = 1.0

# Chemical potential.
μ = 0.0

# System size.
L = 16

# Inverse temperature.
β = 4.0;
nothing #hide
````

Now we need to construct the Hamiltonian matrix ``H``.

````@example usage
H = zeros(L,L)
for i in 1:L
    j = mod1(i+1,L)
    H[j,i], H[i,j] = -t, -t
end
H
````

We also need to define the Fermi function.

````@example usage
# Fermi function.
fermi(ϵ, μ, β) = (1+tanh(β*(ϵ-μ)/2))/2
````

Let us calculate the exact density matrix ``\rho`` so that we may asses the accuracy of the KPM method.

````@example usage
# Diagonalize the Hamiltonian matrix.
ϵ, U = eigen(H)

# Calculate the density matrix.
ρ = U * Diagonal(fermi.(ϵ, μ, β)) * adjoint(U)
````

Now let use the KPM to approximate ``\rho``. We will need to define the order ``M`` of the expansion
and give approximate bounds for the egenspectrum of ``H``, making sure to overestimate the the true
interval spanned by the egienvalues of ``H``. Note that because we are considering the simple
non-interacting model here, the exact eigenspectrum of ``H`` is known and spans
the interval ``\epsilon \in [-2t, 2t].``

````@example usage
# Define eigenspectrum bounds.
bounds = (-3.0t, 3.0t)

# Define order of Chebyshev expansion used in KPM approximation.
M = 10

# Initialize expansion.
kpm_expansion = KPMExpansion(x -> fermi(x, μ, β), bounds, M);
nothing #hide
````

Now let us test and see how good a job it does approximating the density matrix ``\rho``.

````@example usage
# Initialize KPM density matrix approxiatmion
ρ_kpm = similar(H)

# Initialize additional matrices to avoid dynamic memory allocation.
mtmp = zeros(L,L,3)

# Calculate KPM density matrix approximation.
kpm_eval!(ρ_kpm, H, kpm_expansion, mtmp)

# Check how good an approximation it is.
println("Matrix Error = ", norm(ρ_kpm - ρ)/norm(ρ) )
````

It is also possible to efficiently multiply vectors by KPM approximation to the density matrix ``\rho``.
Let us test this functionality on a random vector, seeing how accurate the result is.

````@example usage
# Initialize random vector.
v = randn(L)

# Calculate exact product with density matrix.
ρv = ρ * v

# Initialize vector to contain approximate product with densith matrix.
ρv_kpm = similar(v)

# Initialize to avoid dynamic memory allocation.
vtmp = zeros(L, 3)

# Calculate approximate product with density matrix.
kpm_mul!(ρv_kpm, H, kpm_expansion, v, vtmp)

# Check how good the approximation is.
println("Vector Error = ", norm(ρv_kpm - ρv) / norm(ρv) )
````

Let us now provide the KPM approximation with better bounds on the eigenspectrum
and also increase the order the of the expansion and see how the result improves.

````@example usage
# Define eigenspectrum bounds.
bounds = (-2.5t, 2.5t)

# Define order of Chebyshev expansion used in KPM approximation.
M = 100

# Update KPM approximation.
update_kpmexpansion!(kpm_expansion, x -> fermi(x, μ, β), bounds, M)

# Calculate KPM density matrix approximation.
kpm_eval!(ρ_kpm, H, kpm_expansion, mtmp)

# Check how good an approximation it is.
println("Matrix Error = ", norm(ρ_kpm - ρ)/norm(ρ) )

# Calculate approximate product with density matrix.
kpm_mul!(ρv_kpm, H, kpm_expansion, v, vtmp)

# Check how good the approximation is.
println("Vector Error = ", norm(ρv_kpm - ρv) / norm(ρv) )
````

