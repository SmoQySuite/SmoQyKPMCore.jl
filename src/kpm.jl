# calculate coefficients for chebyshev expansion
function cheb_coefs()

    return nothing
end

function cheb_coefs!()

    return nothing
end


# multiply vector by chebyshev expansion using recursion relations
function cheb_mul(
    func!::Function, v::AbstractArray{T}, bounds::NTuple{2,E}, coef::AbstractVector{E}
) where {T<:Number, E<:AbstractFloat}

    return nothing
end

function cheb_mul!(
    v′::AbstractArray{T}, func!::Function, v::AbstractArray{T}, bounds::NTuple{2,E}, coef::AbstractVector{E}
) where {T<:Number, E<:AbstractFloat}

    return nothing
end

# evaluate chebyshev expansion of scalar function using recursion relations
function cheb_eval(x::T, bounds, coefs) where {T<:Number}

    x = cheb_scale(x, bounds)
    Tn2, Tn1 = one(x), x
    ret = coefs[1]*Tn2 + coefs[2]*Tn1
    for coef in @view coefs[3:end]
        Tn3 = 2x*Tn1 - Tn2
        ret += coef*Tn3
        Tn1, Tn2 = Tn3, Tn1
    end

    return f
end

# evaluate chebyshev expansion of matrix function using recursion relations
function cheb_eval(X::AbstractMatrix{T}, bounds, coefs) where {T<:Number}

    return F
end

function cheb_eval!(F::AbstractMatrix{T}, X::AbstractMatrix{T}, bounds, coefs) where {T<:Number}

    return nothing
end