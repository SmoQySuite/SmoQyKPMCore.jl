# using Revise
using LinearAlgebra
using CUDA
using SmoQyKPMCore

## Fermi function.
fermi(ϵ, μ, β) = (1+tanh(β*(ϵ-μ)/2))/2

## Nearest-neighbor hopping amplitude.
const t = 1.0f0

## Chemical potential.
const μ = 0.0f0

## System size.
const L = 16

## Inverse temperature.
const β = 4.0f0

# define hamiltonian matrix
H = zeros(eltype(t),L,L)
for i in 1:L
    j = mod1(i+1,L)
    H[j,i], H[i,j] = -t, -t
end
Hc = CUDA.CuArray(H)

## Define eigenspectrum bounds.
bounds = (-2.5f0t, 2.5f0t)

## Define order of Chebyshev expansion used in KPM approximation.
M = 10

## Initialize KPM expansion
kpm_expansion = KPMExpansionCUDA(x -> fermi(x, μ, β), bounds, M)