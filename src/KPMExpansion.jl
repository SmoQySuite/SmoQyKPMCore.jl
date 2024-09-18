mutable struct KPMExpansion{T<:AbstractFloat, Tfft<:FFTW.r2rFFTWPlan}

    M::Int
    bounds::NTuple{2,T}
    buf::Vector{T}
    r2rplan::Tfft
end

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


function update_expansion!(kpm_expansion::KPMExpansion, func::Function, bounds, M::Int, N::Int = 2*M)

    @assert N ≥ M
    @assert (length(bounds) == 2) && (bounds[1] < bounds[2]) && (eltype(bounds) == T)
    kpm_expansion.bounds = tuple(bounds...)
    kpm_expansion.M = M
    kpm_expansion.N = N
    resize!(kpm_expansion.buf, N)
    kpm_expansion.r2rplan = FTW.plan_r2r!(kpm_expansion.buf, FFTW.REDFT10)
    _kpm_coefs!(kpm_expansion, func)

    return nothing
end


function update_expansion_bounds!(kpm_expansion::KPMExpansion{T}, func::Function, bounds) where {T<:AbstractFloat}

    @assert (length(bounds) == 2) && (bounds[1] < bounds[2]) && (eltype(bounds) == T)
    kpm_expansion.bounds = tuple(bounds...)
    _kpm_coefs!(kpm_expansion, func)

    return nothing
end


function update_expansion_order!(kpm_expansion::KPMExpansion, func::Function, M::Int, N::Int = 2*M)

    @assert N ≥ M
    kpm_expansion.M = M
    kpm_expansion.N = N
    resize!(kpm_expansion.buf, N)
    kpm_expansion.r2rplan = FTW.plan_r2r!(kpm_expansion.buf, FFTW.REDFT10)
    _kpm_coefs!(kpm_expansion, func)

    return nothing
end


function kpm_mul(A, kpm_expansion::KPMExpansion, v::T) where {T<:AbstractVector}

    v′ = similar(v)
    kpm_mul!(v′, A, kpm_expansion, v)

    return v′
end

function kpm_mul!(
    v′::T, A, kpm_expansion::KPMExpansion, v::T,
    α₁::T = similar(v), α₂::T = similar(v), α₃::T = similar(v)
) where {T<:AbstractVecOrMat}

    (; M, bounds, buf) = kpm_expansion
    coefs = @view buf[1:M]
    kpm_mul!(v′, A, coefs, bounds, v, α₁, α₂, α₃)

    return nothing
end


function kpm_eval(
    A::T, kpm_expansion::KPMExpansion{T}
) where {T<:AbstractFloat}

    (; buf, bounds, M) = kpm_expansion
    coefs = @view buf[1:M]
    F = kpm_eval(A, coefs, bounds)

    return F
end

function kpm_eval(
    A::AbstractMatrix{T}, kpm_expansion::KPMExpansion{T}
) where {T<:AbstractFloat}

    F = similar(A)
    kpm_eval!(F, A, kpm_expansion)

    return F
end

function kpm_eval!(
    F::T, A, kpm_expansion::KPMExpansion,
    T₁::T = similar(F), T₂::T = similar(F), T₃::T = similar(F)
) where {T<:AbstractMatrix}

    (; M, bounds, buf) = kpm_expansion
    coefs = @view buf[1:M]
    kpm_eval!(F, A, coefs, bounds, T₁, T₂, T₃)

    return nothing
end