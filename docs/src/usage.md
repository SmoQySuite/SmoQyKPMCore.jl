```@meta
EditURL = "../../examples/usage.jl"
```

# Usage

Here we demonstrate the basic usage for the [SmoQyKPMCore](https://github.com/SmoQySuite/SmoQyKPMCore.jl) package.
Let us first import the relevant packages we will want to use in this example.

````@example usage
using LinearAlgebra
using SparseArrays
````

Package for making figures

````@example usage
using CairoMakie
CairoMakie.activate!(type = "svg")

using SmoQyKPMCore
````

In this usage example we will consider a 1D chain tight-binding model
```math
\hat{H} = -t \sum_{i,\sigma} (\hat{c}_{i+1,\sigma}^\dagger \hat{c}_{i,\sigma} + {\rm H.c.})
        = \sum_\sigma \hat{c}_{i,\sigma}^\dagger [H_{i,j}] \hat{c}_{j,\sigma},
```
where ``\hat{c}_{i,\sigma}^\dagger \ (\hat{c}_{i,\sigma})`` creates (annihilates) a spin-``\sigma`` electron on site ``i`` in the lattice,
and ``t`` is the nearest-neighbor hopping amplitude.

## Density Matrix Approximation

With ``H`` the Hamiltonian matrix, the corresponding density matrix is given by ``\rho = f(H)`` where
```math
f(\epsilon) = \frac{1}{1+e^{\beta (\epsilon-\mu)}} = \frac{1}{2} \left[ 1 + \tanh\left(\tfrac{\beta(\epsilon-\mu)}{2}\right) \right]
```
is the Fermi function, with ``\beta = 1/T`` the inverse temperature and ``\mu`` the chemical potential.
Here we will applying the kernel polynomial method (KPM) algorithm to approximate the density matrix ``\rho`` by
a Chebyshev polynomial expansion.

Let us first define the relevant model parameter values.

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

Now let use the [`KPMExpansion`](@ref) type to approximate ``\rho``.
We will need to define the order ``M`` of the expansion and give approximate bounds for the eigenspectrum of ``H``,
making sure to overestimate the the true interval spanned by the eigenvalues of ``H``.
Because we are considering the simple non-interacting model, the exact eigenspectrum of ``H`` is known and spans
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

Now let us test and see how good a job it does approximating the density matrix ``\rho``,
using the [`kpm_eval!`](@ref) function to do so.

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
Let us test this functionality with the [`kpm_mul!`](@ref) on a random vector, seeing how accurate the result is.

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
We will use the [`kpm_update!`](@ref) function to update the [`KPMExpansion`](@ref) in-place.

````@example usage
# Define eigenspectrum bounds.
bounds = (-2.5t, 2.5t)

# Define order of Chebyshev expansion used in KPM approximation.
M = 100

# Update KPM approximation.
kpm_update!(kpm_expansion, x -> fermi(x, μ, β), bounds, M)

# Calculate KPM density matrix approximation.
kpm_eval!(ρ_kpm, H, kpm_expansion, mtmp)

# Check how good an approximation it is.
println("Matrix Error = ", norm(ρ_kpm - ρ)/norm(ρ) )

# Calculate approximate product with density matrix.
kpm_mul!(ρv_kpm, H, kpm_expansion, v, vtmp)

# Check how good the approximation is.
println("Vector Error = ", norm(ρv_kpm - ρv) / norm(ρv) )
````

Now let me quickly demonstrate how we may approximate the the trace ``\rho``
using a set of random vector ``R_n`` using the [`kpm_dot`](@ref) function.

````@example usage
# Calculate exact trace.
trρ = tr(ρ)
println("Exact trace = ", trρ)

# Number of random vectors
N = 100

# Initialize random vectors.
R = randn(L, N)

# Initialize array to avoid dynamic memory allocation
Rtmp = zeros(L, N, 3)

# Approximate trace of density matrix.
trρ_approx = kpm_dot(H, kpm_expansion, R, Rtmp)
println("Approximate trace = ", trρ_approx)

# Report the error in the approximation.
println("Trace estimate error = ", abs(trρ_approx - trρ)/trρ)
````

## Density of States Approximation

Next let us demonstrate how we may approximate the density of states ``\mathcal{N}(\epsilon)``
for a 1D chain tight-binding model. The first step a very larger Hamiltonian matrix ``H,``
which we represent as a spare matrix.

````@example usage
# Size of 1D chain considered
L = 10_000

# Construct sparse Hamiltonian matrix.
rows = Int[]
cols = Int[]
for i in 1:L
    j = mod1(i+1, L)
    push!(rows,i)
    push!(cols,j)
    push!(rows,j)
    push!(cols,i)
end
vals = fill(-t, length(rows))
H = sparse(rows, cols, vals)
````

Let us now calculate the fist ``M`` moments ``\mu_m`` using ``N`` random vectors
using the function [`kpm_moments`](@ref).

````@example usage
# Number of moments to calculate.
M = 250

# Number of random vectors used to approximate moments.
N = 100

# Initialize random vectors.
R = randn(L, N)

# Calculate the moments.
μ_kpm = kpm_moments(M, H, bounds, R);
nothing #hide
````

Having calculate the moments, let us next evaluate the density of states ``\mathcal{N}(\epsilon)`` at ``P`` points
with the [`kpm_density`](@ref) function.

````@example usage
# Number of points at which to evaluate the density of states.
P = 1000

# Evaluate density of states.
dos, ϵ = kpm_density(P, μ_kpm, bounds);
nothing #hide
````

Without regularization, the approximate for the density of states generated above will have clearly visible Gibbs oscillations.
To partially suppress these artifacts, we apply the Jackson kernel to the moments ``\mu`` using
the [`apply_jackson_kernel`](@ref) function.

````@example usage
# Apply Jackson kernel.
μ_jackson = apply_jackson_kernel(μ_kpm)

# Evaluate density of states.
dos_jackson, ϵ_jackson = kpm_density(P, μ_jackson, bounds);
nothing #hide
````

Having approximated the density of states, let us now plot it.

````@example usage
fig = Figure(
    size = (700, 400),
    fonts = (; regular= "CMU Serif"),
    figure_padding = 10
)

ax = Axis(
    fig[1, 1],
    aspect = 7/4,
    xlabel = L"\epsilon", ylabel = L"\mathcal{N}(\epsilon)",
    xticks = range(start = bounds[1], stop = bounds[2], length = 5),
    xlabelsize = 30, ylabelsize = 30,
    xticklabelsize = 24, yticklabelsize = 24,
)

lines!(
    ϵ, dos,
    linewidth = 2.0,
    alpha = 1.0,
    color = :red,
    linestyle = :solid,
    label = "Dirichlet"
)

lines!(
    ϵ_jackson, dos_jackson,
    linewidth = 3.0,
    alpha = 1.0,
    color = :black,
    linestyle = :solid,
    label = "Jackson"
)

xlims!(
    ax, bounds[1], bounds[2]
)

ylims!(
    ax, 0.0, 1.05 * maximum(dos)
)

axislegend(
    ax, halign = :center, valign = :top, labelsize = 30
)

fig
````

