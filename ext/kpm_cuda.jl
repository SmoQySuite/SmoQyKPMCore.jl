function kpm_mul!(
    v′, A, C, v,
    λmin, λmax,
    αₙ, αₙ₋₁, αₙ₋₂
)

    a, b = _rescaling_coefficients(λmin, λmax)
    @. αₙ₋₂ = -v
    @. αₙ₋₁ = 0
    @. αₙ   = 0
    for (n,cₙ) in enumerate(C)
        _scaled_matrix_multiply!(αₙ, A, αₙ₋₁, a, b)
        @. αₙ = (2 - isone(n-1)) * αₙ - αₙ₋₂
        @. v′ += cₙ * αₙ
        αₙ, αₙ₋₁, αₙ₋₂ = αₙ₋₂, αₙ, αₙ₋₁
    end

    return nothing
end