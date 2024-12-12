# calculate kpm expansion coefficients
function SmoQyKPMCore.kpm_coefs!(
    coefs::CuVector{T},
    func::Function,
    bounds::Tuple{T,T},
    tmp::CuVector{Complex{T}} = CUDA.zeros(Complex{T}, 4*length(coefs)),
    buf::CuVector{T} = CUDA.zeros(T, 2*length(coefs)),
    fftplan::AbstractFFTs.Plan = CUDA.CUFFT.plan_fft!(tmp)
) where {T<:AbstractFloat}


    M = length(coefs)
    N = length(buf)
    a, b = SmoQyKPMCore._rescaling_coefficients(bounds)
    half = T(0.5)
    @. buf = 1:N
    @. buf = func((cos(((buf-half)π/N)-b)/a))
    dctii_fft!(buf, tmp, fftplan)
    @views @. coefs = buf[1:M] / N / (1 + isone(1:M))

    return nothing
end


# calculate μₘ = ⟨R|Tₘ(A′)|R⟩
function SmoQyKPMCore.kpm_moments!(
    μ::CuVector{E}, A, bounds::Tuple{E,E}, R::C,
    tmp::CuArray{T} = CUDA.zeros(eltype(R), size(R)..., 3)
) where {E<:Number, T<:Number, C<:Union{CuVector{T}, CuMatrix{T}}}

    @assert iseven(length(μ))
    αₘ   = selectdim(tmp, ndims(tmp), 1)
    αₘ₋₁ = selectdim(tmp, ndims(tmp), 2)
    αₘ₋₂ = selectdim(tmp, ndims(tmp), 3)
    # order of expansion
    M = length(μ)
    # number of vectors
    Nᵥ = size(R, 2)
    # calulate rescaling coefficients
    a, b = SmoQyKPMCore._rescaling_coefficients(bounds)
    # α₁ = R
    copyto!(αₘ₋₂, R)
    # α₂ = A′⋅α₁
    _scaled_matrix_multiply!(αₘ₋₁, A, αₘ₋₂, a, b)
    # μ₁ = tr[α₁ᵀ⋅α₁] = tr[Rᵀ⋅R]
    μ₁ = dot(αₘ₋₂, αₘ₋₂)/Nᵥ
    # μ₂ = tr[α₁ᵀ⋅α₂] = tr[Rᵀ⋅α₂]
    μ₂ = dot(αₘ₋₁, αₘ₋₂)/Nᵥ
    # record first two moments
    CUDA.@allowscalar μ[1], μ[2] = μ₁, μ₂
    # iterate over remaining terms in sum
    for m in 2:(M÷2)
        # αₘ′ = (a⋅A + b⋅I)⋅αₘ₋₁
        _scaled_matrix_multiply!(αₘ, A, αₘ₋₁, a, b)
        # αₘ = 2⋅αₘ′ - αₘ₋₂
        @. αₘ = 2*αₘ - αₘ₋₂
        # rename αₘ, αₘ₋₁, αₘ₋₂ = αₘ₋₂, αₘ, αₘ₋₁
        αₘ, αₘ₋₁, αₘ₋₂ = αₘ₋₂, αₘ, αₘ₋₁
        # μ₂ₘ₋₁ = 2⋅αₘ₋₂ᵀ⋅αₘ₋₂ - μ₁
        μ₂ₘ₋₁ = 2 * real(dot(αₘ₋₂,αₘ₋₂))/Nᵥ - μ₁
        # μ₂ₘ = 2⋅αₘ₋₁ᵀ⋅αₘ₋₂ - μ₂
        μ₂ₘ = 2 * real(dot(αₘ₋₁,αₘ₋₂))/Nᵥ - μ₂
        # record momentums
        CUDA.@allowscalar μ[2*m-1], μ[2*m] = μ₂ₘ₋₁, μ₂ₘ
    end

    return nothing
end


# calculate μₘ = ⟨U|Tₘ(A′)|V⟩
function SmoQyKPMCore.kpm_moments!(
    μ::CuVector{E}, A, bounds::Tuple{E,E}, U::C, V::C,
    tmp::CuArray{T} = CUDA.zeros(eltype(V), size(V)..., 3)
) where {E<:Number, T<:Number, C<:Union{CuVector{T}, CuMatrix{T}}}

    αₘ = selectdim(tmp, ndims(tmp), 1)
    αₘ₋₁ = selectdim(tmp, ndims(tmp), 2)
    αₘ₋₂ = selectdim(tmp, ndims(tmp), 3)
    a, b = SmoQyKPMCore._rescaling_coefficients(bounds)
    # get expansion order M
    M = length(μ)
    # number of vectors
    Nᵥ = size(U, 2)
    # αₘ₋₂ = V
    copyto!(αₘ₋₂, V)
    # αₘ₋₁ = A′⋅αₘ₋₂ = (a⋅A + b⋅I)⋅αₘ₋₂
    _scaled_matrix_multiply!(αₘ₋₁, A, αₘ₋₂, a, b)
    # calculate μ₁ = Uᵀ⋅α₁
    μ₁ = real(dot(U, α₁))/Nᵥ
    # calculate μ₂ = Uᵀ⋅α₂
    μ₂ = real(dot(U, α₂))/Nᵥ
    # record first two moments
    CUDA.@allowscalar μ[1], μ[2] = μ₁, μ₂
    # iterate of order of expansion m = 3,...,M
    for m in 3:M
        # αₘ′ = (a⋅A + b⋅I)⋅αₘ₋₁
        _scaled_matrix_multiply!(αₘ, A, αₘ₋₁, a, b)
        # αₘ = 2⋅αₘ′ - αₘ₋₂
        @. αₘ = 2 * αₘ - αₘ₋₂
        # μₘ = Uᵀ⋅αₘ
        μₘ = real(dot(U, αₘ))/Nᵥ
        # record μₘ moment
        CUDA.@allowscalar μ[m] = μₘ
        # rename αₘ, αₘ₋₁, αₘ₋₂ = αₘ₋₂, αₘ, αₘ₋₁
        αₘ, αₘ₋₁, αₘ₋₂ = αₘ₋₂, αₘ, αₘ₋₁
    end

    return nothing
end


# calculate density ρ(ϵ) and energies ϵ given the moments μ
function SmoQyKPMCore.kpm_density!(
    ρ::CuVector{T},
    ϵ::CuVector{T},
    μ::CuVector{T},
    W::T,
    tmp::CuVector{Complex{T}} = CUDA.zeros(Complex{T}, 2*length(μ)),
    ifftplan::AbstractFFTs.Plan = CUDA.CUFFT.plan_ifft!(tmp)
) where {T<:AbstractFloat}

    fill!(ρ, 0.0)
    fill!(ϵ, 0.0)
    copyto!(ρ, μ)
    dctiii_fft!(ρ, tmp, ifftplan)
    N = length(ρ)
    half = T(1/2)
    @. ϵ  = cos(π*((1:N)-half)/N)
    @. ρ *= inv(π*sqrt(1-ϵ^2)) / (sqrt(2π)*N*W/2)
    @. ϵ *= (W/2)

    return nothing
end


# calculate ⟨R|F(A)|R⟩
function SmoQyKPMCore.kpm_dot(
    A, coefs::CuVector, bounds::Tuple{E,E}, R::C,
    tmp::CuArray{T} = CUDA.zeros(eltype(R), size(R)..., 3)
) where {E<:AbstractFloat, T<:Number, C<:Union{CuVector{T}, CuMatrix{T}}}

    @assert iseven(length(coefs))
    αₘ   = selectdim(tmp, ndims(tmp), 1)
    αₘ₋₁ = selectdim(tmp, ndims(tmp), 2)
    αₘ₋₂ = selectdim(tmp, ndims(tmp), 3)
    # order of expansion
    M = length(coefs)
    # number of vectors
    Nᵥ = size(R, 2)
    # calulate rescaling coefficients
    a, b = SmoQyKPMCore._rescaling_coefficients(bounds)
    # α₁ = v
    copyto!(αₘ₋₂, R)
    # α₂ = A′⋅α₁
    _scaled_matrix_multiply!(αₘ₋₁, A, αₘ₋₂, a, b)
    # μ₁ = tr[Rᵀ⋅R] = tr[α₁ᵀ⋅α₁]
    μ₁ = real(dot(αₘ₋₂, αₘ₋₂))/Nᵥ
    # μ₂ = tr[α₂ᵀ⋅R] = tr[α₂ᵀ⋅α₁]
    μ₂ = real(dot(αₘ₋₁, αₘ₋₂))/Nᵥ
    # get the first two coefficients c₁ and c₂
    CUDA.@allowscalar c₁, c₂ = coefs[1], coefs[2]
    # S = c₁⋅μ₁ + c₂⋅μ₂
    S = c₁ * μ₁ + c₂ * μ₂
    # iterate over remaining terms in sum
    for m in 2:(M÷2)
        # αₘ′ = (a⋅A + b⋅I)⋅αₘ₋₁
        _scaled_matrix_multiply!(αₘ, A, αₘ₋₁, a, b)
        # αₘ = 2⋅αₘ′ - αₘ₋₂ = 2⋅A′⋅αₘ₋₁ - αₘ₋₂
        @. αₘ = 2*αₘ - αₘ₋₂
        # rename αₘ, αₘ₋₁, αₘ₋₂ = αₘ₋₂, αₘ, αₘ₋₁
        αₘ, αₘ₋₁, αₘ₋₂ = αₘ₋₂, αₘ, αₘ₋₁
        # μ₂ₘ₋₁ = 2⋅Tr[αₘ₋₂ᵀ⋅αₘ₋₂] - μ₁
        μ₂ₘ₋₁ = 2 * real(dot(αₘ₋₂,αₘ₋₂))/Nᵥ - μ₁
        # μ₂ₘ = 2⋅Tr[αₘ₋₁ᵀ⋅αₘ₋₂] - μ₂
        μ₂ₘ = 2 * real(dot(αₘ₋₁,αₘ₋₂))/Nᵥ - μ₂
        # get the coefficients c₂ₘ₋₁ and c₂ₘ
        CUDA.@allowscalar c₂ₘ₋₁, c₂ₘ = coefs[2*m-1], coefs[2*m]
        # S = S + c₂ₘ₋₁⋅μ₂ₘ₋₁ + c₂ₘ⋅μ₂ₘ
        S = S + c₂ₘ₋₁ * μ₂ₘ₋₁ + c₂ₘ * μ₂ₘ
    end

    return S
end


# calculate ⟨U|F(A)|V⟩
function SmoQyKPMCore.kpm_dot(
    A, coefs::CuVector, bounds::Tuple{E,E}, U::C, V::C,
    tmp::CuArray{T} = CUDA.zeros(eltype(R), size(R)..., 3)
) where {E<:AbstractFloat, T<:Number, C<:Union{CuVector{T}, CuMatrix{T}}}

    αₘ = selectdim(tmp, ndims(tmp), 1)
    αₘ₋₁ = selectdim(tmp, ndims(tmp), 2)
    αₘ₋₂ = selectdim(tmp, ndims(tmp), 3)
    a, b = SmoQyKPMCore._rescaling_coefficients(bounds)
    # get expansion order M
    M = length(coefs)
    # number of vectors
    Nᵥ = size(U, 2)
    # αₘ₋₂ = V
    copyto!(αₘ₋₂, V)
    # αₘ₋₁ = A′⋅αₘ₋₂ = (a⋅A + b⋅I)⋅αₘ₋₂
    _scaled_matrix_multiply!(αₘ₋₁, A, αₘ₋₂, a, b)
    # get expansion coefficients c₁ and c₂
    CUDA.@allowscalar c₁, c₂ = coefs[1], coefs[2]
    # S = c₁⋅⟨U|α₁⟩ + c₂⋅⟨U|α₂⟩
    S = c₁ * real(dot(U, αₘ₋₂))/Nᵥ + c₂ * real(dot(U, αₘ₋₁))/Nᵥ
    # iterate of order of expansion m = 3,...,M
    for m in 3:M
        # αₘ′ = A′⋅αₘ₋₁ = (a⋅A + b⋅I)⋅αₘ₋₁
        _scaled_matrix_multiply!(αₘ, A, αₘ₋₁, a, b)
        # αₘ = 2⋅αₘ′ - αₘ₋₂
        @. αₘ = 2 * αₘ - αₘ₋₂
        # μₘ = ⟨U|αₘ⟩
        μₘ = real(dot(U, αₘ))/Nᵥ
        # get cₘ expansion coefficient
        CUDA.@allowscalar cₘ = coefs[m]
        # S = S + cₘ⋅μₘ = S + cₘ⋅⟨U|αₘ⟩
        S = S + cₘ * μₘ
        # αₘ, αₘ₋₁, αₘ₋₂ = αₘ₋₂, αₘ, αₘ₋₁
        αₘ, αₘ₋₁, αₘ₋₂ = αₘ₋₂, αₘ, αₘ₋₁
    end

    return S
end
 

# calculate v′ = F(A)⋅v
function SmoQyKPMCore.kpm_mul!(
    v′::C, A, coefs::CuVector, bounds::Tuple{E,E}, v::C,
    tmp::CuArray{T} = CUDA.zeros(eltype(v), size(v)..., 3)
) where {E<:AbstractFloat, T<:Number, C<:Union{CuVector{T}, CuMatrix{T}}}

    αₘ = selectdim(tmp, ndims(tmp), 1)
    αₘ₋₁ = selectdim(tmp, ndims(tmp), 2)
    αₘ₋₂ = selectdim(tmp, ndims(tmp), 3)
    a, b = _rescaling_coefficients(bounds)
    # get expansion order
    M = length(coefs)
    # get c₁ and c₂ expansion coefficients
    CUDA.@allowscalar c₁, c₂ = coefs[1], coefs[2]
    # α₁ = v
    copyto!(αₘ₋₂, v)
    # α₂ = A′⋅α₁ = (a⋅A + b⋅I)⋅α₁
    _scaled_matrix_multiply!(αₘ₋₁, A, αₘ₋₂, a, b)
    # v′ = c₁⋅α₁ + c₂⋅α₂
    @. v′ = c₁ * αₘ₋₂ + c₂ * αₘ₋₁
    # iterate of order of expansion m = 3,...,N
    for m in 3:M
        # αₘ′ = A′⋅αₘ₋₁ = (a⋅A + b⋅I)⋅αₘ₋₁
        _scaled_matrix_multiply!(αₘ, A, αₘ₋₁, a, b)
        # αₘ = 2⋅αₘ′ - αₘ₋₂
        @. αₘ = 2 * αₘ - αₘ₋₂
        # get cₘ expansion coefficient
        CUDA.@allowscalar cₘ = coefs[m]
        # v′ = v′ + cₘ⋅αₘ
        @. v′ = v′ + cₘ * αₘ
        # αₘ, αₘ₋₁, αₘ₋₂ = αₘ₋₂, αₘ, αₘ₋₁
        αₘ, αₘ₋₁, αₘ₋₂ = αₘ₋₂, αₘ, αₘ₋₁
    end

    return nothing
end


# calculate v = F(A)⋅v
function SmoQyKPMCore.kpm_lmul!(
    A, coefs::CuVector, bounds::Tuple{E,E}, v::M,
    tmp::CuArray{T} = CUDA.zeros(eltype(v), size(v)..., 3)
) where {E<:AbstractFloat, T<:Number, M<:Union{CuVector{T}, CuMatrix{T}}}

    SmoQyKPMCore.kpm_mul!(v, A, coefs, bounds, v, tmp)

    return nothing
end


# evaluate y = A′⋅x, modifying y in-place where A′ is the scaled version of A
function _scaled_matrix_multiply!(y, A, x, a, b)

    SmoQyKPMCore._matrix_multiply!(y, A, x)
    @. y = a*y + b*x

    return nothing
end