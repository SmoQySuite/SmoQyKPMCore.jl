@doc raw"""
    kpm_coefs(func::Function, bounds, M::Int, N::Int = 2*M)
    
Calculate and return the Chebyshev expansion coefficients.
Refer to [`kpm_coefs!`](@ref) for more information.
"""
function kpm_coefs(func::Function, bounds, M::Int, N::Int = 2*M)

    coefs = zeros(eltype(bounds), M)
    buf = zeros(eltype(bounds), N)
    r2rplan = FFTW.plan_r2r!(buf, FFTW.REDFT10)
    kpm_coefs!(
        coefs, func, bounds, buf, r2rplan
    )

    return coefs
end

@doc raw"""
    kpm_coefs!(
        coefs::AbstractVector{T}, func::Function, bounds,
        buf::AbstractVector{T} = zeros(T, 2*length(coefs)),
        r2rplan = FFTW.plan_r2r!(buf, FFTW.REDFT10)
    ) where {T<:AbstractFloat}

Calculate and record the Chebyshev polynomial expansion coefficients to order `M` in the vector `ceofs`
for the function `func` on the interval `(bounds[1], bounds[2])`. Let `length(buf)` be the number of evenly
spaced points on the interval for which `func` is evaluated when performing Chebyshev-Gauss
quadrature to compute the Chebyshev polynomial expansion coefficients.
"""
function kpm_coefs!(
    coefs::AbstractVector{T}, func::Function, bounds,
    buf::AbstractVector{T} = zeros(T, 2*length(coefs)),
    r2rplan = FFTW.plan_r2r!(buf, FFTW.REDFT10)
) where {T<:AbstractFloat}

    M = length(coefs)
    N = length(buf)
    @assert N ≥ M
    @assert eltype(bounds) == T
    a, b = _rescaling_coefficients(bounds)
    for i in eachindex(buf)
        A′ = cos((i-0.5)π / N)
        A = (A′ - b)/a
        buf[i] = func(A)
    end
    mul!(buf, r2rplan, buf)
    buf ./= N
    buf[1] /= 2
    @views @. coefs = buf[1:M]

    return nothing
end


@doc raw"""
    kpm_moments(
        M::Int, A, bounds, R::T,
        tmp = zeros(eltype(R), size(R)..., 3)
    ) where {T<:AbstractVecOrMat}

If ``R`` is a vector, calculate and return the first ``M`` moments as
```math
\mu_m = \langle R | T_m(A^\prime) | R \rangle
```
where ``A^\prime`` is the rescaled version of ``A`` using the `bounds`.
If ``R`` is a matrix then calculate the moments as
```math
\mu_m = \frac{1}{N} \sum_{n=1}^N \langle R_n | T_m(A^\prime) | R_n \rangle,
```
where ``| R_n \rangle`` is the n'th column of ``R``.
"""
function kpm_moments(
    M::Int, A, bounds, R::T,
    tmp = zeros(eltype(R), size(R)..., 3)
) where {T<:AbstractVecOrMat}

    μ = zeros(eltype(R), M)
    kpm_moments!(μ, A, bounds, R, tmp)

    return μ
end

@doc raw"""
    kpm_moments(
        M::Int, A, bounds, U::T, V::T,
        tmp = zeros(eltype(V), size(V)..., 3)
    ) where {T<:AbstractVecOrMat}

If ``U`` and ``V`` are vector then calculate and return the first ``M`` moments
```math
\mu_m = \langle U | T_m(A^\prime) | V \rangle,
```
where ``A^\prime`` is the rescaled version of ``A`` using the `bounds`.
If ``U`` and ``V`` are matrices then calculate the moments as
```math
\mu_m = \frac{1}{N} \sum_{n=1}^N \langle U_n | T_m(A^\prime) | V_n \rangle,
```
where ``| U_n \rangle`` and  ``| V_n \rangle`` are are the n'th columns of each matrix.
"""
function kpm_moments(
    M::Int, A, bounds, U::T, V::T,
    tmp = zeros(eltype(V), size(V)..., 3)
) where {T<:AbstractVecOrMat}

    μ = zeros(eltype(U), M)
    kpm_moments!(μ, A, bounds, U, V, tmp)

    return μ
end

@doc raw"""
    kpm_moments!(
        μ::AbstractVector, A, bounds, R::T,
        tmp = zeros(eltype(R), size(R)..., 3)
    ) where {T<:AbstractVecOrMat}

If ``R`` is a vector, calculate and return the first ``M`` moments as
```math
\mu_m = \langle R | T_m(A^\prime) | R \rangle
```
where ``A^\prime`` is the rescaled version of ``A`` using the `bounds`.
If ``R`` is a matrix then calculate the moments as
```math
\mu_m = \frac{1}{N} \sum_{n=1}^N \langle R_n | T_m(A^\prime) | R_n \rangle,
```
where ``| R_n \rangle`` is the n'th column of ``R``.
"""
function kpm_moments!(
    μ::AbstractVector, A, bounds, R::T,
    tmp = zeros(eltype(R), size(R)..., 3)
) where {T<:AbstractVecOrMat}

    @assert iseven(length(μ))
    αₘ   = selectdim(tmp, ndims(tmp), 1)
    αₘ₋₁ = selectdim(tmp, ndims(tmp), 2)
    αₘ₋₂ = selectdim(tmp, ndims(tmp), 3)
    # order of expansion
    N = length(μ)
    # number of random vector
    Nᵣᵥ = size(R, 2)
    # calulate rescaling coefficients
    a, b = _rescaling_coefficients(bounds)
    # α₁ = R
    copyto!(αₘ₋₂, R)
    # α₂ = A′⋅α₁ where A′ is scaled A
    _scaled_matrix_multiply!(αₘ₋₁, A, αₘ₋₂, a, b)
    # μ₁ = tr[Rᵀ⋅R]
    μ₁ = μ[1] = dot(αₘ₋₂, αₘ₋₂)/Nᵣᵥ
    # μ₂ = tr[Rᵀ⋅α₂]
    μ₂ = μ[2] = dot(αₘ₋₁, αₘ₋₂)/Nᵣᵥ
    # iterate over remaining terms in sum
    for n in 1:(N÷2)
        # μ₂ₘ₋₁ = 2⋅αₘ₋₂ᵀ⋅αₘ₋₂ - μ₁
        μ[2*n-1] = 2 * dot(αₘ₋₂,αₘ₋₂)/Nᵣᵥ - μ₁
        # μ₂ₘ = 2⋅αₘ₋₁ᵀ⋅αₘ₋₂ - μ₂
        μ[2*n] = 2 * dot(αₘ₋₁,αₘ₋₂)/Nᵣᵥ - μ₂
        # αₘ′ = (a⋅A + b⋅I)⋅αₘ₋₁
        _scaled_matrix_multiply!(αₘ, A, αₘ₋₁, a, b)
        # αₘ = (2-δₘ₋₂)⋅αₘ′ - αₘ₋₂
        axpby!(-1, αₘ₋₂, 2, αₘ)
        # rename αₘ, αₘ₋₁, αₘ₋₂ = αₘ₋₂, αₘ, αₘ₋₁
        αₘ, αₘ₋₁, αₘ₋₂ = αₘ₋₂, αₘ, αₘ₋₁
    end

    return nothing
end

@doc raw"""
    kpm_moments!(
        μ::AbstractVector, A, bounds, U::T, V::T,
        tmp = zeros(eltype(V), size(V)..., 3)
    ) where {T<:AbstractVecOrMat}

If ``U`` and ``V`` are vector then calculate and return the first ``M`` moments
```math
\mu_m = \langle U | T_m(A^\prime) | V \rangle,
```
where ``A^\prime`` is the rescaled version of ``A`` using the `bounds`.
If ``U`` and ``V`` are matrices then calculate the moments as
```math
\mu_m = \frac{1}{N} \sum_{n=1}^N \langle U_n | T_m(A^\prime) | V_n \rangle,
```
where ``| U_n \rangle`` and  ``| V_n \rangle`` are are the n'th columns of each matrix.
"""
function kpm_moments!(
    μ::AbstractVector, A, bounds, U::T, V::T,
    tmp = zeros(eltype(V), size(V)..., 3)
) where {T<:AbstractVecOrMat}

    αₘ = selectdim(tmp, ndims(tmp), 1)
    αₘ₋₁ = selectdim(tmp, ndims(tmp), 2)
    αₘ₋₂ = selectdim(tmp, ndims(tmp), 3)

    λmin, λmax = bounds
    a, b = _rescaling_coefficients(λmin, λmax)
    # number of random vector
    Nᵣᵥ = size(U, 2)
    # αₘ₋₂ = -V
    @. αₘ₋₂ = -V
    # αₘ₋₁ = 0
    fill!(αₘ₋₁, 0)
    # αₘ = 0
    fill!(αₘ, 0)
    # iterate of order of expansion n = 1,...,N
    for n in eachindex(μ)
        # αₘ′ = (a⋅A + b⋅I)⋅αₘ₋₁
        _scaled_matrix_multiply!(αₘ, A, αₘ₋₁, a, b)
        # αₘ = (2-δₘ₋₂)⋅αₘ′ - αₘ₋₂
        axpby!(-1, αₘ₋₂, 2 - isone(n-1), αₘ)
        # μₘ = Uᵀ⋅αₘ
        μ[n] = dot(U, αₘ)/Nᵣᵥ
        # rename αₘ, αₘ₋₁, αₘ₋₂ = αₘ₋₂, αₘ, αₘ₋₁
        αₘ, αₘ₋₁, αₘ₋₂ = αₘ₋₂, αₘ, αₘ₋₁
    end

    return nothing
end


@doc raw"""
    kpm_density(
        N::Int, μ::T, W
    ) where {T<:AbstractVector}

Given the KPM moments ``\mu``, evaluate the corresponding spectral
density ``\rho(\epsilon)`` at ``N`` energy points ``\epsilon``, where
``W`` is the spectral bandwidth or the bounds of the spectrum.
This function returns both ``\rho(\epsilon)`` and ``\epsilon``.
"""
function kpm_density(
    N::Int, μ::T, W
) where {T<:AbstractVector}

    ρ = T(undef, N)
    ϵ = T(undef, N)
    kpm_density!(ρ, ϵ, μ, W)

    return (ρ, ϵ)
end

@doc raw"""
    kpm_density!(
        ρ::AbstractVector{T},
        ϵ::AbstractVector{T},
        μ::AbstractVector{T},
        bounds,
        r2rplan = FFTW.plan_r2r!(ρ, FFTW.REDFT01)
    ) where {T<:AbstractFloat}

Given the KPM moments ``\mu``, calculate the spectral density ``\rho(\epsilon)``
and corresponding energies ``\epsilon`` where it is evaluated. Here `bounds`
is bounds the eigenspecturm, such that the bandwidth is given by `W = bounds[2] - bounds[1]`.
"""
function kpm_density!(
    ρ::AbstractVector{T},
    ϵ::AbstractVector{T},
    μ::AbstractVector{T},
    bounds,
    r2rplan = FFTW.plan_r2r!(ρ, FFTW.REDFT01)
) where {T<:AbstractFloat}

    W = bounds[2] - bounds[1]
    kpm_density!(ρ, ϵ, μ, W, r2rplan)

    return nothing
end

@doc raw"""
    kpm_density!(
        ρ::AbstractVector{T},
        ϵ::AbstractVector{T},
        μ::AbstractVector{T},
        W::T,
        r2rplan = FFTW.plan_r2r!(ρ, FFTW.REDFT01),
    ) where {T<:AbstractFloat}

Given the KPM moments ``\mu``, calculate the spectral density ``\rho(\epsilon)``
and corresponding energies ``\epsilon`` where it is evaluated.
Here ``W`` is the spectral bandwidth.
"""
function kpm_density!(
    ρ::AbstractVector{T},
    ϵ::AbstractVector{T},
    μ::AbstractVector{T},
    W::T,
    r2rplan = FFTW.plan_r2r!(ρ, FFTW.REDFT01),
) where {T<:AbstractFloat}

    fill!(ρ, 0.0)
    fill!(ϵ, 0.0)
    copyto!(ρ, μ)
    mul!(ρ, r2rplan, ρ)
    N = length(ρ)
    @. ϵ  = cos(π*((1:N)-0.5)/N)
    @. ρ *= inv(π*sqrt(1-ϵ^2)) / (sqrt(2π)*N*W/2)
    @. ϵ *= (W/2)

    return nothing
end

@doc raw"""
    kpm_dot(
        coefs::T, moments::T
    ) where {T<:AbstractVector}

Calculate the inner product
```math
\langle c | \mu \rangle = \sum_{m=1}^M c_m \cdot \mu_m,
```
where ``c_m`` are the KPM expansion coefficients, and ``\mu_m`` are the KPM moments.
"""
function kpm_dot(
    coefs::T, moments::T
) where {T<:AbstractVector}

    return dot(coefs, moments)
end

@doc raw"""
    kpm_dot(
        A, coefs::AbstractVector, bounds, R::T,
        tmp = zeros(eltype(R), size(R)..., 3)
    ) where {T<:AbstractVecOrMat}

If ``R`` is a single vector, then calculate the inner product
```math
\begin{align*}
S & = \langle R | F(A^\prime) | R \rangle \\
  & = \sum_{m=1}^M \langle R | c_m T_m(A^\prime) | R \rangle
\end{align*},
```
where ``A^\prime`` is the scaled version of ``A`` using the `bounds`.
If ``R`` is a matrix, then calculate
```math
\begin{align*}
S & = \langle R | F(A^\prime) | R \rangle \\
 & = \frac{1}{N} \sum_{n=1}^N \sum_{m=1}^M \langle R_n | c_m T_m(A^\prime) | R_n \rangle
\end{align*},
```
where ``| R_n \rangle`` is a column of ``R``.
"""
function kpm_dot(
    A, coefs::AbstractVector, bounds, R::T,
    tmp = zeros(eltype(R), size(R)..., 3)
) where {T<:AbstractVecOrMat}

    @assert iseven(length(coefs))
    αₘ   = selectdim(tmp, ndims(tmp), 1)
    αₘ₋₁ = selectdim(tmp, ndims(tmp), 2)
    αₘ₋₂ = selectdim(tmp, ndims(tmp), 3)
    # order of expansion
    N = length(coefs)
    # number of random vector
    Nᵣᵥ = size(R, 2)
    # initialize S = 0
    S = zero(eltype(R))
    # calulate rescaling coefficients
    a, b = _rescaling_coefficients(bounds)
    # α₁ = v
    copyto!(αₘ₋₂, R)
    # α₂ = A′⋅α₁ where A′ is scaled A
    _scaled_matrix_multiply!(αₘ₋₁, A, αₘ₋₂, a, b)
    # μ₁ = tr[Rᵀ⋅R] = tr[α₁ᵀ⋅α₁]
    μ₁ = dot(αₘ₋₂, αₘ₋₂)/Nᵣᵥ
    # μ₂ = tr[α₂ᵀ⋅R] = tr[α₂ᵀ⋅α₁]
    μ₂ = dot(αₘ₋₁, αₘ₋₂)/Nᵣᵥ
    # iterate over remaining terms in sum
    for n in 1:(N÷2)
        # S = S + c₂ₘ₋₁⋅(2⋅αₘ₋₂ᵀ⋅αₘ₋₂ - μ₁)
        S += coefs[2*n-1] * (2 * dot(αₘ₋₂,αₘ₋₂)/Nᵣᵥ - μ₁)
        # S = S + c₂ₘ⋅(2⋅αₘ₋₁ᵀ⋅αₘ₋₂ - μ₂)
        S += coefs[2*n] * (2 * dot(αₘ₋₁,αₘ₋₂)/Nᵣᵥ - μ₂)
        # αₘ′ = (a⋅A + b⋅I)⋅αₘ₋₁
        _scaled_matrix_multiply!(αₘ, A, αₘ₋₁, a, b)
        # αₘ = 2⋅αₘ′ - αₘ₋₂ = 2⋅(a⋅A + b⋅I)⋅αₘ₋₁ - αₘ₋₂
        axpby!(-1, αₘ₋₂, 2, αₘ)
        # rename αₘ, αₘ₋₁, αₘ₋₂ = αₘ₋₂, αₘ, αₘ₋₁
        αₘ, αₘ₋₁, αₘ₋₂ = αₘ₋₂, αₘ, αₘ₋₁
    end

    return S
end

@doc raw"""
    kpm_dot(
        A, coefs::AbstractVector, bounds, U::T, V::T,
        tmp = zeros(eltype(V), size(V)..., 3)
    ) where {T<:AbstractVecOrMat}

If ``U`` and ``V`` are single vectors, then calculate the inner product
```math
\begin{align*}
S & = \langle U | F(A^\prime) | V \rangle \\
  & = \sum_{m=1}^M \langle U | c_m T_m(A^\prime) | V \rangle
\end{align*},
```
where ``A^\prime`` is the scaled version of ``A`` using the `bounds`.
If ``U`` and ``V`` are matrices, then calculate
```math
\begin{align*}
S & = \langle U | F(A^\prime) | V \rangle \\
  & = \frac{1}{N} \sum_{n=1}^N \sum_{m=1}^M \langle U_n | c_m T_m(A^\prime) | V_n \rangle
\end{align*},
```
where ``| U_n \rangle`` and ``| V_n \rangle`` are the columns of each matrix.
"""
function kpm_dot(
    A, coefs::AbstractVector, bounds, U::T, V::T,
    tmp = zeros(eltype(V), size(V)..., 3)
) where {T<:AbstractVecOrMat}

    αₘ = selectdim(tmp, ndims(tmp), 1)
    αₘ₋₁ = selectdim(tmp, ndims(tmp), 2)
    αₘ₋₂ = selectdim(tmp, ndims(tmp), 3)
    λmin, λmax = bounds
    a, b = _rescaling_coefficients(λmin, λmax)
    # initialize S = 0
    S = zero(eltype(V))
    # αₘ₋₂ = -V
    @. αₘ₋₂ = -V
    # αₘ₋₁ = 0
    fill!(αₘ₋₁, 0)
    # αₘ = 0
    fill!(αₘ, 0)
    # iterate of order of expansion n = 1,...,N
    for (n,cₘ) in enumerate(coefs)
        # αₘ′ = (a⋅A + b⋅I)⋅αₘ₋₁
        _scaled_matrix_multiply!(αₘ, A, αₘ₋₁, a, b)
        # αₘ = (2-δₘ₋₂)⋅αₘ′ - αₘ₋₂
        axpby!(-1, αₘ₋₂, 2 - isone(n-1), αₘ)
        # S = S + cₘ⋅⟨U|αₘ⟩
        S += cₘ * dot(U, αₘ) / size(U, 2)
        # αₘ, αₘ₋₁, αₘ₋₂ = αₘ₋₂, αₘ, αₘ₋₁
        αₘ, αₘ₋₁, αₘ₋₂ = αₘ₋₂, αₘ, αₘ₋₁
    end

    return S
end


@doc raw"""
    kpm_mul(
        A, coefs::AbstractVector, bounds, v::T, tmp = zeros(eltype(v), size(v)..., 3)
    ) where {T<:AbstractVecOrMat}

Evaluate and return the vector ``v^\prime = F(A) \cdot v`` where ``F(A)`` is represented by the Chebyshev expansion.
For more information refer to [`kpm_mul!`](@ref).
"""
function kpm_mul(
    A, coefs::AbstractVector, bounds, v::T, tmp = zeros(eltype(v), size(v)..., 3)
) where {T<:AbstractVecOrMat}

    v′ = similar(v)
    kpm_mul!(v′, A, coefs, bounds, v, tmp)

    return v′
end

@doc raw"""
    kpm_mul!(
        v′::T, A, coefs::AbstractVector, bounds, v::T,
        tmp = zeros(eltype(v), size(v)..., 3)
    ) where {T<:AbstractVecOrMat}

Evaluates ``v^\prime = F(A) \cdot v``, writing the result to ``v^\prime``, where ``F(A)`` is represented by the Chebyshev expansion.
Here `A` is either a function that can be called as `A(u,v)` to evaluate
``u = A\cdot v``, modifying `u` in-place, or is a type for which the operation `mul!(u, A, v)` is defined.
The vector `coefs` contains Chebyshev expansion coefficients to approximate ``F(A)``, where the eigenspectrum
of ``A`` is contained in the interval `(bounds[1], bounds[2])` specified by the `bounds` argument.
The vector `v` is vector getting multiplied by the Chebyshev expansion for ``F(A)``.
Lastly, `tmp` is an array used to avoid dynamic memory allocations.
"""
function kpm_mul!(
    v′::T, A, coefs::AbstractVector, bounds, v::T,
    tmp = zeros(eltype(v), size(v)..., 3)
) where {T<:AbstractVecOrMat}

    αₘ = selectdim(tmp, ndims(tmp), 1)
    αₘ₋₁ = selectdim(tmp, ndims(tmp), 2)
    αₘ₋₂ = selectdim(tmp, ndims(tmp), 3)
    λmin, λmax = bounds
    a, b = _rescaling_coefficients(λmin, λmax)
    # αₘ₋₂ = -v
    @. αₘ₋₂ = -v
    # αₘ₋₁ = 0
    fill!(αₘ₋₁, 0)
    # αₘ = 0
    fill!(αₘ, 0)
    # initialize v′ = 0.0
    fill!(v′, 0)
    # iterate of order of expansion n = 1,...,N
    for (n,cₘ) in enumerate(coefs)
        # αₘ′ = (a⋅A + b⋅I)⋅αₘ₋₁
        _scaled_matrix_multiply!(αₘ, A, αₘ₋₁, a, b)
        # αₘ = (2-δₘ₋₂)⋅αₘ′ - αₘ₋₂
        axpby!(-1, αₘ₋₂, 2 - isone(n-1), αₘ)
        # v′ = v′ + cₘ⋅αₘ
        axpy!(cₘ, αₘ, v′)
        # αₘ, αₘ₋₁, αₘ₋₂ = αₘ₋₂, αₘ, αₘ₋₁
        αₘ, αₘ₋₁, αₘ₋₂ = αₘ₋₂, αₘ, αₘ₋₁
    end

    return nothing
end


@doc raw"""
    kpm_lmul!(
        A, coefs::AbstractVector, v::T, bounds,
        tmp = zeros(eltype(v), size(v)..., 3)
    ) where {T<:AbstractVecOrMat}

Evaluates ``v = F(A) \cdot v``, modifying ``v`` in-place, where ``F(A)`` is represented by the Chebyshev expansion.
Here `A` is either a function that can be called as `A(u,v)` to evaluate
``u = A\cdot v``, modifying `u` in-place, or is a type for which the operation `mul!(u, A, v)` is defined.
The vector `coefs` contains Chebyshev expansion coefficients to approximate ``F(A)``, where the eigenspectrum
of ``A`` is contained in the interval `(bounds[1], bounds[2])` specified by the `bounds` argument.
The vector `v` is vector getting multiplied by the Chebyshev expansion for ``F(A)``.
Lastly, `tmp` is an array used to avoid dynamic memory allocations.
"""
function kpm_lmul!(
    A, coefs::AbstractVector, v::T, bounds,
    tmp = zeros(eltype(v), size(v)..., 3)
) where {T<:AbstractVecOrMat}

    kpm_mul!(v, A, coefs, bounds, v, tmp)

    return nothing
end


@doc raw"""
    kpm_eval(x::AbstractFloat, coefs, bounds)

Evaluate ``F(x)`` where ``x`` is real number in the interval `bounds[1] < x < bound[2]`,
and the function ``F(\bullet)`` is represented by a Chebyshev expansion with coefficients given
by the vector `coefs`.
"""
function kpm_eval(x::AbstractFloat, coefs, bounds)

    # calulate rescaling coefficients
    a, b = _rescaling_coefficients(bounds)
    # calclate x′, a rescaling of x
    x′ = a * x + b
    # T₁, T₂ = 1, x′
    T₁, T₂ = one(x′), x′
    # F = c₁⋅T₁ + c₂⋅T₂
    F = coefs[1]*T₁ + coefs[2]*T₂
    # rename Tₘ₋₁, Tₘ₋₂ = T₂, T₁
    Tₘ₋₁, Tₘ₋₂ = T₂, T₁
    for cₘ in @view coefs[3:end]
        # Tₘ = 2⋅x′⋅Tₘ₋₁ - Tₘ₋₂
        Tₘ = 2*x′*Tₘ₋₁ - Tₘ₋₂
        # F = cₘ⋅Tₘ + F
        F = cₘ * Tₘ + F
        # rename Tₘ₋₁, Tₘ₋₂ = Tₘ, Tₘ₋₁
        Tₘ₋₁, Tₘ₋₂ = Tₘ, Tₘ₋₁
    end

    return F
end

@doc raw"""
    kpm_eval(A::AbstractMatrix, coefs, bounds)

Evaluate and return the matrix ``F(A),`` where ``A`` is an operator with strictly real eigenvalues that
fall in the interval `(bounds[1], bounds[2])` specified by the `bounds` argument, and the function
``F(\bullet)`` is represented by a Chebyshev expansion with coefficients given by the vector `coefs`.
"""
function kpm_eval(A::AbstractMatrix, coefs, bounds)

    F = similar(A)
    kpm_eval!(
        F, A, coefs, bounds
    )

    return F
end

@doc raw"""
    kpm_eval!(
        F::AbstractMatrix, A, coefs::AbstractVector, bounds,
        tmp = zeros(eltype(F), size(F)..., 4)
    )

Evaluate and write the matrix ``F(A)`` to `F`, where ``A`` is an operator with strictly real eigenvalues that
fall in the interval `(bounds[1], bounds[2])` specified by the `bounds` argument, and the function
``F(\bullet)`` is represented by a Chebyshev expansion with coefficients given by the vector `coefs`.
Lastly, `tmp` is used to avoid dynamic memory allocations.
"""
function kpm_eval!(
    F::AbstractMatrix, A, coefs::AbstractVector, bounds,
    tmp = zeros(eltype(F), size(F)..., 3)
)

    copyto!(F, I)
    kpm_mul!(F, A, coefs, bounds, F, tmp)

    return nothing
end

@doc raw"""
    apply_jackson_kernel!(coefs)

Modify the Chebyshev expansion coefficients by applying the Jackson kernel to them.
"""
function apply_jackson_kernel!(coefs)

    Mp = lastindex(coefs)
    for i in eachindex(coefs)
        m = i-1
        coefs[i] *= (1/Mp)*((Mp-m)cos(m*π/Mp) + sin(m*π/Mp)/tan(π/Mp))
    end

    return nothing
end

@doc raw"""
    apply_jackson_kernel(coefs)

Return the Chebyshev expansion coefficients transformed by the Jackson kernel.
"""
function apply_jackson_kernel(coefs)

    jackson_coefs = copy(coefs)
    apply_jackson_kernel!(jackson_coefs)

    return jackson_coefs
end

# calculate rescaling coefficient for given bounds
_rescaling_coefficients(bounds) = _rescaling_coefficients(bounds[1], bounds[2]) 

function _rescaling_coefficients(λ_min, λ_max)

    λ_diff = λ_max - λ_min
    λ_sum  = λ_max + λ_min
    return 2/λ_diff, -λ_sum/λ_diff
end

# evaluate y = A′⋅x, modifying y in-place where A′ is the scaled version of A
function _scaled_matrix_multiply!(y, A, x, a, b)

    _matrix_multiply!(y, A, x)
    axpby!(b, x, a, y)

    return nothing
end

# evaluate y = A⋅x, modifying y in-place
_matrix_multiply!(y, A::Function, x) = A(y,x)
_matrix_multiply!(y, A, x) = mul!(y, A, x)