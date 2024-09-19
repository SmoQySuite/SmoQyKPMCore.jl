@doc raw"""
    mutable struct KPMExpansion{T<:AbstractFloat, Tfft<:FFTW.r2rFFTWPlan}

A type to represent the Chebyshev polynomial expansion used in the KPM algorithm.

# Fields
- `M::Int`: Order of the Chebyshev polynomial expansion.
- `bounds::NTuple{2,T}`: Bounds on eigenspectrum in the KPM algorithm.
- `buf::Vector{T}`: The first `M` elements of this vector of the Chebyshev expansion coefficients.
- `r2rplan::Tfft`: Plan for performing DCT to efficiently calculate the Chebyshev expansion coefficients via Chebyshev-Gauss quadrature.
"""
mutable struct KPMExpansion{T<:AbstractFloat, Tfft<:FFTW.r2rFFTWPlan}

    M::Int
    bounds::NTuple{2,T}
    buf::Vector{T}
    r2rplan::Tfft
end

@doc raw"""
    KPMExpansion(func::Function, bounds, M::Int, N::Int = 2*M)

Initialize an instance of the [`KPMExpansion`](@ref) type to approximate the univariate function
`func`, called as `func(x)`, with a order `M` Chebyshev polynomial expansion on the interval
`bounds[1] < x bounds[2]`. Here, `N ≥ M` is the number of points at which `func` is evaluated
on that specified interval, which are then used to calculate the expansion coeffiencents
via Chebyshev-Gauss quadrature.
"""
function KPMExpansion(func::Function, bounds, M::Int, N::Int = 2*M)

    @assert N ≥ M
    @assert (length(bounds) == 2) && (bounds[1] < bounds[2]) && (eltype(bounds) <: AbstractFloat)
    buf = zeros(eltype(bounds), N)
    r2rplan = FFTW.plan_r2r!(buf, FFTW.REDFT10)
    kpm_expansion = KPMExpansion(M, tuple(bounds...), buf, r2rplan)
    _kpm_coefs!(kpm_expansion, func)

    return kpm_expansion
end

# recompute the cofficients for an existing KPMExpansion
function _kpm_coefs!(kpm_expansion::KPMExpansion, func::Function)

    coefs = @view kpm_expansion.buf[1:kpm_expansion.M]
    kpm_coefs!(
        coefs, func, kpm_expansion.bounds, kpm_expansion.buf, kpm_expansion.r2rplan
    )

    return nothing
end


@doc raw"""
    update_expansion!(
        kpm_expansion::KPMExpansion{T}, func::Function, bounds, M::Int, N::Int = 2*M
    ) where {T<:AbstractFloat}

In-place update an instance of [`KPMExpansion`](@ref) to reflect new values for eigenspectrum `bounds`,
expansion order `M`, and number of points at which the expanded function is evaluated when computing the
expansion coefficients. This includes recomputing the expansion coefficients.
"""
function update_expansion!(
    kpm_expansion::KPMExpansion{T}, func::Function, bounds, M::Int, N::Int = 2*M
) where {T<:AbstractFloat}

    @assert N ≥ M
    @assert (length(bounds) == 2) && (bounds[1] < bounds[2]) && (eltype(bounds) == T)
    kpm_expansion.bounds = tuple(bounds...)
    kpm_expansion.M = M
    resize!(kpm_expansion.buf, N)
    kpm_expansion.r2rplan = FFTW.plan_r2r!(kpm_expansion.buf, FFTW.REDFT10)
    _kpm_coefs!(kpm_expansion, func)

    return nothing
end


@doc raw"""
    update_expansion_bounds!(
        kpm_expansion::KPMExpansion{T}, func::Function, bounds
    ) where {T<:AbstractFloat}

In-place update an instance of [`KPMExpansion`](@ref) to reflect new values for eigenspectrum `bounds`,
recomputing the expansion coefficients.
"""
function update_expansion_bounds!(
    kpm_expansion::KPMExpansion{T}, func::Function, bounds
) where {T<:AbstractFloat}

    @assert (length(bounds) == 2) && (bounds[1] < bounds[2]) && (eltype(bounds) == T)
    kpm_expansion.bounds = tuple(bounds...)
    _kpm_coefs!(kpm_expansion, func)

    return nothing
end


@doc raw"""
    update_expansion_order!(
        kpm_expansion::KPMExpansion, func::Function, M::Int, N::Int = 2*M
    )

In-place update the expansion order `M` for an instance of [`KPMExpansion`](@ref),
recomputing the expansion coefficients. It is also possible to udpate the number of
point `N` the function `func` is evaluated at to calculate the expansion coefficients.
"""
function update_expansion_order!(
    kpm_expansion::KPMExpansion, func::Function, M::Int, N::Int = 2*M
)

    @assert N ≥ M
    kpm_expansion.M = M
    resize!(kpm_expansion.buf, N)
    kpm_expansion.r2rplan = FFTW.plan_r2r!(kpm_expansion.buf, FFTW.REDFT10)
    _kpm_coefs!(kpm_expansion, func)

    return nothing
end


@doc raw"""
    kpm_mul(A, kpm_expansion::KPMExpansion, v::T) where {T<:AbstractVector}

Evaluate and return the vector ``v^\prime = F(A) \cdot v`` where ``F(A)`` is represented by the Chebyshev expansion.
For more information refer to [`kpm_mul!`](@ref).
"""
function kpm_mul(A, kpm_expansion::KPMExpansion, v::T) where {T<:AbstractVector}

    v′ = similar(v)
    kpm_mul!(v′, A, kpm_expansion, v)

    return v′
end

@doc raw"""
    kpm_mul!(
        v′::T, A, kpm_expansion::KPMExpansion, v::T,
        α₁::T = similar(v), α₂::T = similar(v), α₃::T = similar(v)
    ) where {T<:AbstractVecOrMat}

Evaluates ``v^\prime = F(A) \cdot v``, writing the result to `v′`, where ``F(A)`` is represented by the Chebyshev expansion.
Here `A` is either a function that can be called as `A(u,v)` to evaluate
``u = A\cdot v``, modifying `u` in-place, or is a type for which the operation `mul!(u, A, v)` is defined.
Lastly, the vectors `(α₁, α₂, α₃)` are passed to avoid dynamic memory allocations.
"""
function kpm_mul!(
    v′::T, A, kpm_expansion::KPMExpansion, v::T,
    α₁::T = similar(v), α₂::T = similar(v), α₃::T = similar(v)
) where {T<:AbstractVecOrMat}

    (; M, bounds, buf) = kpm_expansion
    coefs = @view buf[1:M]
    kpm_mul!(v′, A, coefs, bounds, v, α₁, α₂, α₃)

    return nothing
end


@doc raw"""
    kpm_eval(x::T, kpm_expansion::KPMExpansion{T}) where {T<:AbstractFloat}

Evaluate ``F(x)`` where ``x`` is real number in the interval `bounds[1] < x < bound[2]`,
and the function ``F(\bullet)`` is represented by a Chebyshev expansion with coefficients given
by the vector `coefs`.
"""
function kpm_eval(x::T, kpm_expansion::KPMExpansion{T}) where {T<:AbstractFloat}

    (; buf, bounds, M) = kpm_expansion
    coefs = @view buf[1:M]
    F = kpm_eval(x, coefs, bounds)

    return F
end

@doc raw"""
    kpm_eval(A::AbstractMatrix{T}, kpm_expansion::KPMExpansion{T}) where {T<:AbstractFloat}

Evaluate and return the matrix ``F(A),`` where ``A`` is an operator with strictly real eigenvalues
and the function ``F(\bullet)`` is represented by a Chebyshev expansion with coefficients given by the vector `coefs`.
"""
function kpm_eval(A::AbstractMatrix{T}, kpm_expansion::KPMExpansion{T}) where {T<:AbstractFloat}

    F = similar(A)
    kpm_eval!(F, A, kpm_expansion)

    return F
end

@doc raw"""
    kpm_eval!(
        F::T, A, kpm_expansion::KPMExpansion,
        T₁::T = similar(F), T₂::T = similar(F), T₃::T = similar(F)
    ) where {T<:AbstractMatrix}

Evaluate and write the matrix ``F(A)`` to `F`, where ``A`` is an operator with strictly real eigenvalues
and the function ``F(\bullet)`` is represented by a Chebyshev expansion with coefficients given by the vector `coefs`.
Lastly, the matrices `(T₁, T₂, T₃)` are used to avoid dynamic memory allocations.
"""
function kpm_eval!(
    F::T, A, kpm_expansion::KPMExpansion,
    T₁::T = similar(F), T₂::T = similar(F), T₃::T = similar(F)
) where {T<:AbstractMatrix}

    (; M, bounds, buf) = kpm_expansion
    coefs = @view buf[1:M]
    kpm_eval!(F, A, coefs, bounds, T₁, T₂, T₃)

    return nothing
end

@doc raw"""
    apply_jackson_kernel!(kpm_expansion::KPMExpansion)

Modify the Chebyshev expansion coefficients by applying the Jackson kernel to them.
"""
function apply_jackson_kernel!(kpm_expansion::KPMExpansion)

    (; M, buf) = kpm_expansion
    coefs = @view buf[1:M]
    apply_jackson_kernel!(coefs)

    return nothing
end