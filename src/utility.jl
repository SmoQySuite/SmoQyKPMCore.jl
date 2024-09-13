@inline cheb_scale(x, bounds) = 2(x-bounds[1])/(bounds[2]-bounds[1]) - 1
@inline cheb_unscale(x, bounds) = (x+1)*(bounds[2]-bounds[1])/2 + bounds[1]

function apply_scaled_func!(v′, func!, v, bounds)

    λ_diff = bounds[2] - bounds[1]
    λ_sum  = bounds[2] + bounds[1]
    func!(v′, func!, v)
    axpby!(-λ_sum/λ_diff, v, 2/λ_diff, v′)

    return nothing
end