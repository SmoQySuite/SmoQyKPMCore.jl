mutable struct KPMExpansionCUDA{T<:AbstractFloat, Tfft<:AbstractFFTs.Plan}

    M::Int
    bounds::NTuple{2,T}
    coefs::CuVector{T}
    buf::CuVector{T}
    tmp::CuVector{Complex{T}}
    fftplan::Tfft
end

# convert/adapt KPMExpansionCUDA type into CUDA compatible bitstypes
Adapt.@adapt_structure KPMExpansionCUDA

function SmoQyKPMCore.KPMExpansionCUDA(
    func::Function, bounds::NTuple{2,T}, M::Int, N::Int = 2*M
) where {T<:AbstractFloat}

    coefs = CUDA.zeros(T, M)
    buf = CUDA.zeros(T, N)
    tmp = CUDA.zeros(Complex{T}, 2*N)
    fftplan = CUDA.CUFFT.plan_fft!(tmp)
    kpm_expansion = KPMExpansionCUDA(M, bounds, coefs, buf, tmp, fftplan)
    _kpm_coefs!(kpm_expansion, func)

    return kpm_expansion
end

# recompute the cofficients for an existing KPMExpansion
function _kpm_coefs!(kpm_expansion::KPMExpansionCUDA, func::Function)

    (; coefs, buf, tmp, fftplan, bounds) = kpm_expansion
    kpm_coefs!(coefs, func, bounds, tmp, buf, fftplan)

    return nothing
end

# update KPM expansion bounds and order
function SmoQyKPMCore.kpm_update!(
    kpm_expansion::KPMExpansionCUDA{T}, func::Function,
    bounds::NTuple{2,T}, M::Int, N::Int = 2*M
) where {T<:AbstractFloat}

    kpm_expansion.bounds = bounds
    kpm_expansion.M = M
    resize!(kpm_expansion.coefs, M)
    resize!(kpm_expansion.buf, N)
    resize!(kpm_expansion.tmp, 2*N)
    kpm_expansion.fftplan = CUDA.CUFFT.plan_fft!(kpm_expansion.tmp)
    _kpm_coefs!(kpm_expansion, func)

    return nothing
end

# update KPM bounds
function SmoQyKPMCore.kpm_update_bounds!(
    kpm_expansion::KPMExpansionCUDA{T}, func::Function, bounds::NTuple{2,T}
) where {T<:AbstractFloat}

    kpm_expansion.bounds = bounds
    _kpm_coefs!(kpm_expansion, func)

    return nothing
end

# update KPM expansion order
function SmoQyKPMCore.kpm_update_order!(
    kpm_expansion::KPMExpansionCUDA{T}, func::Function, M::Int, N::Int = 2*M
) where {T<:AbstractFloat}

    kpm_expansion.M = M
    resize!(kpm_expansion.coefs, M)
    resize!(kpm_expansion.buf, N)
    resize!(kpm_expansion.tmp, 2*N)
    kpm_expansion.fftplan = CUDA.CUFFT.plan_fft!(kpm_expansion.tmp)
    _kpm_coefs!(kpm_expansion, func)

    return nothing
end

# calculate μₘ = ⟨R|Tₘ(A′)|R⟩
function SmoQyKPMCore.kpm_moments!(
    μ::CuVector{E}, A, kpm_expansion::KPMExpansionCUDA{E}, R::C,
    tmp::CuArray{T} = CUDA.zeros(eltype(R), size(R)..., 3)
) where {E<:AbstractFloat, T<:Number, C<:Union{CuVector{T}, CuMatrix{T}}}

    (; bounds) = kpm_expansion
    SmoQyKPMCore.kpm_moments!(μ, A, bounds, R, tmp)

    return nothing
end

# calculate μₘ = ⟨U|Tₘ(A′)|V⟩
function SmoQyKPMCore.kpm_moments!(
    μ::CuVector{E}, A, kpm_expansion::KPMExpansionCUDA{E}, U::C, V::C,
    tmp::CuArray{T} = CUDA.zeros(eltype(R), size(R)..., 3)
) where {E<:AbstractFloat, T<:Number, C<:Union{CuVector{T}, CuMatrix{T}}}

    (; bounds) = kpm_expansion
    SmoQyKPMCore.kpm_moments!(μ, A, bounds, U, V, tmp)

    return nothing
end

# calculate ⟨R|F(A)|R⟩
function SmoQyKPMCore.kpm_dot(
    A, kpm_expansion::KPMExpansionCUDA{E}, R::C,
    tmp::CuArray{T} = CUDA.zeros(eltype(R), size(R)..., 3)
) where {E<:AbstractFloat, T<:Number, C<:Union{CuVector{T}, CuMatrix{T}}}

    (; coefs, bounds) = kpm_expansion
    S = SmoQyKPMCore.kpm_dot(A, coefs, bounds, R, tmp)

    return S
end

# calculate ⟨U|F(A)|V⟩
function SmoQyKPMCore.kpm_dot(
    A, kpm_expansion::KPMExpansionCUDA{E}, U::C, V::C,
    tmp::CuArray{T} = CUDA.zeros(eltype(R), size(R)..., 3)
) where {E<:AbstractFloat, T<:Number, C<:Union{CuVector{T}, CuMatrix{T}}}

    (; coefs, bounds) = kpm_expansion
    S = SmoQyKPMCore.kpm_dot(A, coefs, bounds, U, V, tmp)

    return S
end

# calculate v′ = F(A)⋅v
function SmoQyKPMCore.kpm_mul!(
    v′::C, A, kpm_expansion::KPMExpansionCUDA{E}, v::C,
    tmp::CuArray{T} = CUDA.zeros(eltype(v), size(v)..., 3)
) where {E<:AbstractFloat, T<:Number, C<:Union{CuVector{T}, CuMatrix{T}}}

    (; coefs, bounds) = kpm_expansion
    SmoQyKPMCore.kpm_mul!(v′, A, coefs, bounds, v, tmp)

    return nothing
end

# calculate v = F(A)⋅v
function SmoQyKPMCore.kpm_lmul!(
    A, kpm_expansion::KPMExpansionCUDA{E}, v::M,
    tmp::CuArray{T} = CUDA.zeros(eltype(v), size(v)..., 3)
) where {E<:AbstractFloat, T<:Number, M<:Union{CuVector{T}, CuMatrix{T}}}

    SmoQyKPMCore.kpm_mul!(v, A, kpm_expansion, v, tmp)

    return nothing
end
