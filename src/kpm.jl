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
        r2rplan::FFTW.r2rFFTWPlan = FFTW.plan_r2r!(zeros(T, 2*length(coefs)), FFTW.REDFT10)
    ) where {T<:AbstractFloat}

Calculate and record the Chebyshev expansion coefficients to order `M` in the vector `ceofs`
for the function `func` on the bounded interval `(bounds[1], bounds[2])`. Let `N` be the number of evenly
spaced points on the specified interval for which `func` is evaluated when performing Chebyshev-Gauss
quadrature to compute the coefficients.
"""
function kpm_coefs!(
    coefs::AbstractVector{T}, func::Function, bounds,
    buf::Vector{T} = zeros(T, 2*length(coefs)),
    r2rplan::FFTW.r2rFFTWPlan = FFTW.plan_r2r!(zeros(T, 2*length(coefs)), FFTW.REDFT10)
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


function kpm_mul(
    A, coefs::AbstractVector, bounds, v::T,
    α₁::T = similar(v), α₂::T = similar(v), α₃::T = similar(v)
) where {T<:AbstractVecOrMat}

    v′ = similar(v)
    kpm_mul!(
        v′, A, coefs, bounds, v,
        α₁, α₂, α₃
    )

    return v′
end

function kpm_mul!(
    v′::T, A, coefs::AbstractVector, bounds, v::T,
    α₁::T = similar(v), α₂::T = similar(v), α₃::T = similar(v)
) where {T<:AbstractVecOrMat}

    # calulate rescaling coefficients
    a, b = _rescaling_coefficients(bounds)
    # α₁ = v
    copyto!(α₁, v)
    # v′ = c₁⋅α₁
    @. v′ = coefs[1] * α₁
    # α₂ = A⋅α₁
    _matrix_multiply!(α₂, A, α₁)
    # α₂ = A′⋅α₁ where A′ is scaled A
    axpby!(b, α₁, a, α₂)
    # v′ = c₂⋅α₂ + v′
    axpy!(coefs[2], α₂, v′)
    # αₙ, αₙ₋₁, αₙ₋₂ = α₃, α₂, α₁
    αₙ, αₙ₋₁, αₙ₋₂ = α₃, α₂, α₁
    # iterate over remaining terms in sum
    for cₙ in @view coefs[3:end]
        # αₙ = A⋅αₙ₋₁
        _matrix_multiply!(αₙ, A, αₙ₋₁)
        # αₙ = A′⋅αₙ₋₁ where A′ is scaled A
        axpby!(b, αₙ₋₁, a, αₙ)
        # αₙ = 2⋅A′⋅αₙ₋₁ - αₙ₋₂
        axpby!(-1, αₙ₋₂, 2, αₙ)
        # v′ = cₙ⋅αₙ + v′
        axpy!(cₙ, αₙ, v′)
        # αₙ, αₙ₋₁, αₙ₋₂ = αₙ₋₂, αₙ, αₙ₋₁
        αₙ, αₙ₋₁, αₙ₋₂ = αₙ₋₂, αₙ, αₙ₋₁
    end

    return nothing
end


function kpm_eval(A::AbstractFloat, coefs, bounds)

    # calulate rescaling coefficients
    a, b = _rescaling_coefficients(bounds)
    # calclate A′, a rescaling of A
    A′ = a * A + b
    # T₁, T₂ = 1, A′
    T₁, T₂ = one(A′), A′
    # F = c₁⋅T₁ + c₂⋅T₂
    F = coefs[1]*T₁ + coefs[2]*T₂
    # rename Tₙ₋₁, Tₙ₋₂ = T₂, T₁
    Tₙ₋₁, Tₙ₋₂ = T₂, T₁
    for cₙ in @view coefs[3:end]
        # Tₙ = 2⋅A′⋅Tₙ₋₁ - Tₙ₋₂
        Tₙ = 2*A′*Tₙ₋₁ - Tₙ₋₂
        # F = cₙ⋅Tₙ + F
        F = cₙ * Tₙ + F
        # rename Tₙ₋₁, Tₙ₋₂ = Tₙ, Tₙ₋₁
        Tₙ₋₁, Tₙ₋₂ = Tₙ, Tₙ₋₁
    end

    return F
end

function kpm_eval(A::AbstractMatrix, coefs, bounds)

    F = similar(A)
    kpm_eval!(
        F, A, coefs, bounds
    )

    return F
end

function kpm_eval!(
    F::T, A,
    coefs::AbstractVector, bounds,
    T₁::T = similar(F), T₂::T = similar(F), T₃::T = similar(F)
) where {T<:AbstractMatrix}

    # calulate rescaling coefficients
    a, b = _rescaling_coefficients(bounds)
    # T₁ = I
    copyto!(T₁, I)
    # T₂ = A
    _matrix_multiply!(T₂, A, T₁)
    # T₂ = A′ where A′ is rescaled A
    axpby!(b, T₁, a, T₂)
    # F = c₁⋅T₁ + c₂⋅T₂
    @. F = coefs[1]*T₁ + coefs[2]*T₂
    # rename Tₙ, Tₙ₋₁, Tₙ₋₂ = T₃, T₂, T₁
    Tₙ, Tₙ₋₁, Tₙ₋₂ = T₃, T₂, T₁
    # iterate over remaining terms in sum
    for cₙ in @view coefs[3:end]
        # Tₙ = A⋅Tₙ₋₁
        _matrix_multiply!(Tₙ, A, Tₙ₋₁)
        # Tₙ = A′⋅Tₙ₋₁ where A′ is scaled A
        axpby!(b, Tₙ₋₁, a, Tₙ)
        # Tₙ = 2⋅A′⋅Tₙ₋₁ - Tₙ₋₂
        axpby!(-1, Tₙ₋₂, 2, Tₙ)
        # F = cₙ⋅Tₙ + F
        axpy!(cₙ, Tₙ, F)
        # rename Tₙ, Tₙ₋₁, Tₙ₋₂ = Tₙ₋₂, Tₙ, Tₙ₋₁
        Tₙ, Tₙ₋₁, Tₙ₋₂ = Tₙ₋₂, Tₙ, Tₙ₋₁
    end

    return nothing
end


function apply_jackson_kernel!(coefs)

    Mp = lastindex(coefs) + 2
    for i in eachindex(coefs)
        m = i-1
        coefs[i] *= (1/Mp)*((Mp-m)cos(m*π/Mp) + sin(m*π/Mp)/tan(π/Mp))
    end

    return nothing
end

function apply_jackson_kernel(coefs)

    jackson_coefs = copy(coefs)
    apply_jackson_kernel!(jackson_coefs)

    return jackson_coefs
end


# calculate rescaling coefficient for given bounds
function _rescaling_coefficients(bounds)

    λ_min, λ_max = bounds
    λ_diff = λ_max - λ_min
    λ_sum  = λ_max + λ_min
    return 2/λ_diff, -λ_sum/λ_diff
end

# evaluate y = A⋅x, modifying y in-place
_matrix_multiply!(y, A::Function, x) = A(y,x)
_matrix_multiply!(y, A, x) = mul!(y, A, x)