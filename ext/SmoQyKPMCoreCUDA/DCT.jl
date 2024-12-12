# perform in-place DCT-II transformation using FFT
function dctii_fft!(
    v::CuVector{T},
    tmp::CuVector{Complex{T}} = CUDA.zeros(Complex{T}, 2*length(v)),
    ipfftplan::AbstractFFTs.Plan = CUDA.CUFFT.plan_fft!(tmp)
) where {T<:AbstractFloat}

    N = length(v)
    @views @. tmp[1:N] = v[N:-1:1]
    @views @. tmp[N+1:2N] = v
    mul!(tmp, ipfftplan, tmp)
    half = T(1/2)
    @views @. v = real( tmp[1:N] * exp(im*π*(N-half)*(0:(N-1))/N) )

    return nothing
end

# perform in-place DCT-III transformation using FFT
function dctiii_fft!(
    v::CuVector{T},
    tmp::CuVector{Complex{T}} = CUDA.zeros(Complex{T}, 2*length(v)),
    ipifftplan::AbstractFFTs.Plan = CUDA.CUFFT.plan_ifft!(tmp)
) where {T<:AbstractFloat}

    N = length(v)
    fill!(tmp, zero(T))
    half = T(1/2)
    @. tmp[1:N] = v / exp(im*π*(N-half)*(0:(N-1))/N)
    @views @. tmp[N+2:2N] = conj(tmp[N:-1:2])
    mul!(tmp, ipifftplan, tmp)
    @views @. v = real( tmp[N+1:2N] )

    return nothing
end