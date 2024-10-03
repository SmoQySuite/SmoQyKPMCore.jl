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
        buf::Vector{T} = zeros(T, 2*length(coefs)),
        r2rplan = FFTW.plan_r2r!(zeros(T, 2*length(coefs)), FFTW.REDFT10)
    ) where {T<:AbstractFloat}

Calculate and record the Chebyshev polynomial expansion coefficients to order `M` in the vector `ceofs`
for the function `func` on the interval `(bounds[1], bounds[2])`. Let `length(buf)` be the number of evenly
spaced points on the interval for which `func` is evaluated when performing Chebyshev-Gauss
quadrature to compute the Chebyshev polynomial expansion coefficients.
"""
function kpm_coefs!(
    coefs::AbstractVector{T}, func::Function, bounds,
    buf::Vector{T} = zeros(T, 2*length(coefs)),
    r2rplan = FFTW.plan_r2r!(zeros(T, 2*length(coefs)), FFTW.REDFT10)
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
        M::Int, A, bounds, R::T, tmp = zeros(eltype(R), size(R)..., 3)
    ) where {T<:AbstractVecOrMat}

Calculate and return the first ``M`` moments
```math
\mu_m = \langle R | T_m(A) | R \rangle
```
if ``R`` is a vector, and if ``R`` is a matrix then
```math
\mu_m = \frac{1}{N} \sum_{n=1}^N \langle R_n | T_m(A) | R_n \rangle,
```
where ``| R_n \rangle`` is the n'th column of ``R``.
"""
function kpm_moments(
    M::Int, A, bounds, R::T, tmp = zeros(eltype(R), size(R)..., 3)
) where {T<:AbstractVecOrMat}

    μ = zeros(eltype(R), M)
    kpm_moments!(μ, A, bounds, R, tmp)

    return μ
end

@doc raw"""
    kpm_moments(
        M::Int, A, bounds, U::T, V::T, tmp = zeros(eltype(V), size(V)..., 3)
    ) where {T<:AbstractVecOrMat}

Calculate and return the first ``M`` moments
```math
\mu_m = \langle U | T_m(A) | V \rangle
```
if ``U`` and ``V`` are vector, and if they are matrices then
```math
\mu_m = \frac{1}{N} \sum_{n=1}^N \langle U_n | T_m(A) | V_n \rangle,
```
where ``| U_n \rangle`` and  ``| V_n \rangle`` are are the n'th columns of each matrix.
"""
function kpm_moments(
    M::Int, A, bounds, U::T, V::T, tmp = zeros(eltype(V), size(V)..., 3)
) where {T<:AbstractVecOrMat}

    μ = zeros(eltype(R), M)
    kpm_moments!(μ, A, bounds, U, V, tmp)

    return μ
end

@doc raw"""
    kpm_moments!(
        μ::AbstractVector, A, bounds, R::T, tmp = zeros(eltype(R), size(R)..., 3)
    ) where {T<:AbstractVecOrMat}

Calculate the moments
```math
\mu_m = \langle R | T_m(A) | R \rangle
```
if ``R`` is a vector, and if ``R`` is a matrix then
```math
\mu_m = \frac{1}{N} \sum_{n=1}^N \langle R_n | T_m(A) | R_n \rangle,
```
where ``| R_n \rangle`` is the n'th column of ``R``.
"""
function kpm_moments!(
    μ::AbstractVector, A, bounds, R::T, tmp = zeros(eltype(R), size(R)..., 3)
) where {T<:AbstractVecOrMat}

    kpm_moments!(μ, A, bounds, R, R, tmp)

    return nothing
end

@doc raw"""
    kpm_moments!(
        μ::AbstractVector, A, bounds, U::T, V::T, tmp = zeros(eltype(V), size(V)..., 3)
    ) where {T<:AbstractVecOrMat}

Calculate the moments
```math
\mu_m = \langle U | T_m(A) | V \rangle
```
if ``U`` and ``V`` are vector, and if they are matrices then
```math
\mu_m = \frac{1}{N} \sum_{n=1}^N \langle U_n | T_m(A) | V_n \rangle,
```
where ``| U_n \rangle`` and  ``| V_n \rangle`` are are the n'th columns of each matrix.
"""
function kpm_moments!(
    μ::AbstractVector, A, bounds, U::T, V::T, tmp = zeros(eltype(V), size(V)..., 3)
) where {T<:AbstractVecOrMat}

    αₙ = selectdim(tmp, ndims(tmp), 1)
    αₙ₋₁ = selectdim(tmp, ndims(tmp), 2)
    αₙ₋₂ = selectdim(tmp, ndims(tmp), 3)

    λmin, λmax = bounds
    a, b = _rescaling_coefficients(λmin, λmax)
    # αₙ₋₂ = -V
    @. αₙ₋₂ = -V
    # αₙ₋₁ = 0
    fill!(αₙ₋₁, 0)
    # αₙ = 0
    fill!(αₙ, 0)
    # iterate of order of expansion n = 1,...,N
    for n in eachindex(μ)
        # αₙ′ = (a⋅A + b⋅I)⋅αₙ₋₁
        _scaled_matrix_multiply!(αₙ, A, αₙ₋₁, a, b)
        # αₙ = (2-δₙ₋₂)⋅αₙ′ - αₙ₋₂
        axpby!(-1, αₙ₋₂, 2 - isone(n-1), αₙ)
        # μₙ = ⟨U|αₙ⟩
        μ[n] = dot(U, αₙ) / size(U, 2)
        # rename αₙ, αₙ₋₁, αₙ₋₂ = αₙ₋₂, αₙ, αₙ₋₁
        αₙ, αₙ₋₁, αₙ₋₂ = αₙ₋₂, αₙ, αₙ₋₁
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

Evaluates ``v^\prime = F(A) \cdot v``, writing the result to `v′`, where ``F(A)`` is represented by the Chebyshev expansion.
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

    αₙ = selectdim(tmp, ndims(tmp), 1)
    αₙ₋₁ = selectdim(tmp, ndims(tmp), 2)
    αₙ₋₂ = selectdim(tmp, ndims(tmp), 3)
    λmin, λmax = bounds
    a, b = _rescaling_coefficients(λmin, λmax)
    # initialize v′ = 0.0
    fill!(v′, 0)
    # αₙ₋₂ = -v
    @. αₙ₋₂ = -v
    # αₙ₋₁ = 0
    fill!(αₙ₋₁, 0)
    # αₙ = 0
    fill!(αₙ, 0)
    # iterate of order of expansion n = 1,...,N
    for (n,cₙ) in enumerate(coefs)
        # αₙ′ = (a⋅A + b⋅I)⋅αₙ₋₁
        _scaled_matrix_multiply!(αₙ, A, αₙ₋₁, a, b)
        # αₙ = (2-δₙ₋₂)⋅αₙ′ - αₙ₋₂
        axpby!(-1, αₙ₋₂, 2 - isone(n-1), αₙ)
        # v′ = v′ + cₙ⋅αₙ
        axpy!(cₙ, αₙ, v′)
        # αₙ, αₙ₋₁, αₙ₋₂ = αₙ₋₂, αₙ, αₙ₋₁
        αₙ, αₙ₋₁, αₙ₋₂ = αₙ₋₂, αₙ, αₙ₋₁
    end

    # α₁ = selectdim(tmp, ndims(tmp), 1)
    # α₂ = selectdim(tmp, ndims(tmp), 2)
    # α₃ = selectdim(tmp, ndims(tmp), 3)
    # # calulate rescaling coefficients
    # a, b = _rescaling_coefficients(bounds)
    # # α₁ = v
    # copyto!(α₁, v)
    # # v′ = c₁⋅α₁
    # @. v′ = coefs[1] * α₁
    # # α₂ = A′⋅α₁ where A′ is scaled A
    # _scaled_matrix_multiply!(α₂, A, α₁, a, b)
    # # v′ = c₂⋅α₂ + v′
    # axpy!(coefs[2], α₂, v′)
    # # αₙ, αₙ₋₁, αₙ₋₂ = α₃, α₂, α₁
    # αₙ, αₙ₋₁, αₙ₋₂ = α₃, α₂, α₁
    # # iterate over remaining terms in sum
    # for cₙ in @view coefs[3:end]
    #     # αₙ = A′⋅αₙ₋₁ where A′ is scaled A
    #     _scaled_matrix_multiply!(αₙ, A, αₙ₋₁, a, b)
    #     # αₙ = 2⋅A′⋅αₙ₋₁ - αₙ₋₂
    #     axpby!(-1, αₙ₋₂, 2, αₙ)
    #     # v′ = cₙ⋅αₙ + v′
    #     axpy!(cₙ, αₙ, v′)
    #     # αₙ, αₙ₋₁, αₙ₋₂ = αₙ₋₂, αₙ, αₙ₋₁
    #     αₙ, αₙ₋₁, αₙ₋₂ = αₙ₋₂, αₙ, αₙ₋₁
    # end

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
    # rename Tₙ₋₁, Tₙ₋₂ = T₂, T₁
    Tₙ₋₁, Tₙ₋₂ = T₂, T₁
    for cₙ in @view coefs[3:end]
        # Tₙ = 2⋅x′⋅Tₙ₋₁ - Tₙ₋₂
        Tₙ = 2*x′*Tₙ₋₁ - Tₙ₋₂
        # F = cₙ⋅Tₙ + F
        F = cₙ * Tₙ + F
        # rename Tₙ₋₁, Tₙ₋₂ = Tₙ, Tₙ₋₁
        Tₙ₋₁, Tₙ₋₂ = Tₙ, Tₙ₋₁
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
        tmp = zeros(eltype(F), size(F)..., 3)
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

    Tₙ = selectdim(tmp, 3, 1)
    Tₙ₋₁ = selectdim(tmp, 3, 2)
    Tₙ₋₂ = selectdim(tmp, 3, 3)
    # calulate rescaling coefficients
    a, b = _rescaling_coefficients(bounds)
    # initialize F = 0
    fill!(F, 0)
    # Tₙ₋₂ = -I
    copyto!(Tₙ₋₂, I)
    @. Tₙ₋₂ = -Tₙ₋₂
    # Tₙ₋₁ = 0
    fill!(Tₙ₋₁, 0)
    # Tₙ = 0
    fill!(Tₙ, 0)
    # Iterate over expansion orders n = 1,...,N
    for (n,cₙ) in enumerate(coefs)
        # Tₙ′ = (a⋅A + b⋅I)⋅Tₙ₋₁
        _scaled_matrix_multiply!(Tₙ, A, Tₙ₋₁, a, b)
        # Tₙ = (2-δₙ₋₂)⋅Tₙ′ - Tₙ₋₂ = (2-δₙ₋₂)⋅(a⋅A + b⋅I)⋅Tₙ₋₁ - Tₙ₋₂
        axpby!(-1, Tₙ₋₂, 2-isone(n-1), Tₙ)
        # F = cₙ⋅Tₙ + F
        axpy!(cₙ, Tₙ, F)
        # rename Tₙ, Tₙ₋₁, Tₙ₋₂ = Tₙ₋₂, Tₙ, Tₙ₋₁
        Tₙ, Tₙ₋₁, Tₙ₋₂ = Tₙ₋₂, Tₙ, Tₙ₋₁
    end

    # T₁ = selectdim(tmp, 3, 1)
    # T₂ = selectdim(tmp, 3, 2)
    # T₃ = selectdim(tmp, 3, 3)
    # # calulate rescaling coefficients
    # a, b = _rescaling_coefficients(bounds)
    # # T₁ = I
    # copyto!(T₁, I)
    # # T₂ = A
    # _matrix_multiply!(T₂, A, T₁)
    # # T₂ = A′ where A′ is rescaled A
    # axpby!(b, T₁, a, T₂)
    # # F = c₁⋅T₁ + c₂⋅T₂
    # @. F = coefs[1]*T₁ + coefs[2]*T₂
    # # rename Tₙ, Tₙ₋₁, Tₙ₋₂ = T₃, T₂, T₁
    # Tₙ, Tₙ₋₁, Tₙ₋₂ = T₃, T₂, T₁
    # # iterate over remaining terms in sum
    # for cₙ in @view coefs[3:end]
    #     # Tₙ = A⋅Tₙ₋₁
    #     _matrix_multiply!(Tₙ, A, Tₙ₋₁)
    #     # Tₙ = A′⋅Tₙ₋₁ where A′ is scaled A
    #     axpby!(b, Tₙ₋₁, a, Tₙ)
    #     # Tₙ = 2⋅A′⋅Tₙ₋₁ - Tₙ₋₂
    #     axpby!(-1, Tₙ₋₂, 2, Tₙ)
    #     # F = cₙ⋅Tₙ + F
    #     axpy!(cₙ, Tₙ, F)
    #     # rename Tₙ, Tₙ₋₁, Tₙ₋₂ = Tₙ₋₂, Tₙ, Tₙ₋₁
    #     Tₙ, Tₙ₋₁, Tₙ₋₂ = Tₙ₋₂, Tₙ, Tₙ₋₁
    # end

    return nothing
end

@doc raw"""
    apply_jackson_kernel!(coefs)

Modify the Chebyshev expansion coefficients by applying the Jackson kernel to them.
"""
function apply_jackson_kernel!(coefs)

    Mp = lastindex(coefs) + 2
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