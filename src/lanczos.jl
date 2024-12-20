
# Use Lanczos iterations to find a truncated tridiagonal representation of A S,
# up to similarity transformation. Here, A is any Hermitian matrix, while S is
# both Hermitian and positive definite. Traditional Lanczos uses the identity
# matrix, S = I. The extension to non-identity matrices S is as follows: Each
# matrix-vector product A v becomes A S v, and each vector inner product w† v
# becomes w† S v. The implementation below follows Wikipedia, and is the most
# stable of the four variants considered by Paige [1]. This implementation
# introduces additional vector storage so that each Lanczos iteration requires
# only one matrix-vector multiplication for A and S, respectively.

# Similar generalizations of Lanczos have been considered in [2] and [3].
#
# 1. C. C. Paige, IMA J. Appl. Math., 373-381 (1972),
# https://doi.org/10.1093%2Fimamat%2F10.3.373.
# 2. H. A. van der Vorst, Math. Comp. 39, 559-561 (1982),
# https://doi.org/10.1090/s0025-5718-1982-0669648-0
# 3. M. Grüning, A. Marini, X. Gonze, Comput. Mater. Sci. 50, 2148-2156 (2011),
# https://doi.org/10.1016/j.commatsci.2011.02.021.

@doc raw"""
    lanczos(niters, v, A, S = I, rng = Random.default_rng())

Use `niters` Lanczos iterations to find a truncated tridiagonal representation of ``A\cdot S``, up to similarity transformation.
Here, ``A`` is any Hermitian matrix, while ``S`` is both Hermitian and positive definite.
Traditional Lanczos uses the identity matrix, ``S = I``.
The extension to non-identity matrices ``S`` is as follows:
Each matrix-vector product ``A\cdot v`` becomes ``(A S) \cdot v``, and each vector inner product ``w^\dagger \cdot v`` becomes ``w^\dagger \cdot S \cdot v``.
The implementation below follows Wikipedia, and is the most stable of the four variants considered by Paige [1].
This implementation introduces additional vector storage so that each Lanczos iteration requires only one matrix-vector multiplication for ``A`` and ``S``, respectively.

This function returns a [`SymTridiagonal`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.SymTridiagonal) matrix.
Note that the [`eigmin`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.eigmin)
and [`eigmax`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.eigmax) routines have
[specialized implementations](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#Elementary-operations) for a
[`SymTridiagonal`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.SymTridiagonal) matrix type.

# Similar generalizations of Lanczos have been considered in [2] and [3].
#
# 1. C. C. Paige, IMA J. Appl. Math., 373-381 (1972),
# https://doi.org/10.1093%2Fimamat%2F10.3.373.
# 2. H. A. van der Vorst, Math. Comp. 39, 559-561 (1982),
# https://doi.org/10.1090/s0025-5718-1982-0669648-0
# 3. M. Grüning, A. Marini, X. Gonze, Comput. Mater. Sci. 50, 2148-2156 (2011),
# https://doi.org/10.1016/j.commatsci.2011.02.021.
"""
function lanczos(niters, v, A, S = I; rng = Random.default_rng())

    αs = zeros(real(eltype(v)), niters, size(v)[2:end]...)
    βs = zeros(real(eltype(v)), niters-1, size(v)[2:end]...)

    symtridiag = lanczos!(αs, βs, v, A, S; rng = rng)

    return symtridiag
end

@doc raw"""
    lanczos!(
        αs::AbstractVector, βs::AbstractVector, v::AbstractVector,
        A, S = I;
        tmp::AbstractMatrix = zeros(eltype(v), length(v), 5),
        rng = Random.default_rng()
    )

    lanczos!(
        αs::AbstractMatrix, βs::AbstractMatrix, v::AbstractMatrix,
        A, S = I;
        tmp::AbstractArray = zeros(eltype(v), size(v)..., 5),
        rng = Random.default_rng()
    )

Use Lanczos iterations to find a truncated tridiagonal representation of ``A\cdot S``, up to similarity transformation.
Here, ``A`` is any Hermitian matrix, while ``S`` is both Hermitian and positive definite.
Traditional Lanczos uses the identity matrix, ``S = I``.
The extension to non-identity matrices ``S`` is as follows:
Each matrix-vector product ``A\cdot v`` becomes ``(A S) \cdot v``, and each vector inner product ``w^\dagger \cdot v`` becomes ``w^\dagger \cdot S \cdot v``.
The implementation below follows Wikipedia, and is the most stable of the four variants considered by Paige [1].
This implementation introduces additional vector storage so that each Lanczos iteration requires only one matrix-vector multiplication for ``A`` and ``S``, respectively.

The number of Lanczos iterations performed equals `niters = length(αs)`, and `niters - 1 == length(βs)`.
This function returns a [`SymTridiagonal`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.SymTridiagonal) matrix
based on the contents of the vectors `αs` and `βs`. Note that the [`eigmin`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.eigmin)
and [`eigmax`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.eigmax) routines have
[specialized implementations](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#Elementary-operations) for a
[`SymTridiagonal`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.SymTridiagonal) matrix type.

Note that if `αs`, `βs` and `v` are all matrices, then each column of `v` is treated as a seperate vector, and a
vector of [`SymTridiagonal`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.SymTridiagonal)
of length `size(v,2)` will be returned.

# Similar generalizations of Lanczos have been considered in [2] and [3].
#
# 1. C. C. Paige, IMA J. Appl. Math., 373-381 (1972), https://doi.org/10.1093%2Fimamat%2F10.3.373.
# 2. H. A. van der Vorst, Math. Comp. 39, 559-561 (1982), https://doi.org/10.1090/s0025-5718-1982-0669648-0
# 3. M. Grüning, A. Marini, X. Gonze, Comput. Mater. Sci. 50, 2148-2156 (2011), https://doi.org/10.1016/j.commatsci.2011.02.021.
"""
function lanczos!(
    αs::AbstractVector, βs::AbstractVector, v::AbstractVector,
    A, S = I;
    tmp::AbstractMatrix = zeros(eltype(v), length(v), 5),
    rng = Random.default_rng()
)

    (vp, Sv, Svp, w, Sw) = eachcol(tmp)

    niters = length(αs)
    @assert niters - 1 == length(βs)
    
    if iszero(norm(v))
        randn!(rng, v)
    end

    _matrix_multiply!(Sv, S, v)
    c = sqrt(dot(v, Sv))
    @. v /= c
    @. Sv /= c

    _matrix_multiply!(w, A, Sv)
    α = dot(w, Sv)
    @. w = w - α * v
    _matrix_multiply!(Sw, S, w)
    αs[1] = real(α)

    for n in 2:niters
        β = real(sqrt(dot(Sw, w)))
        iszero(β) && break
        @. vp = w / β
        @. Svp = Sw / β
        _matrix_multiply!(w, A, Svp)
        α = dot(w, Svp)
        @. w = w - α * vp - β * v
        _matrix_multiply!(Sw, S, w)
        @. v = vp
        αs[n] = real(α)
        βs[n-1] = real(β)
    end

    return SymTridiagonal(αs, βs)
end

function lanczos!(
    αs::AbstractMatrix, βs::AbstractMatrix, v::AbstractMatrix,
    A, S = I;
    tmp::AbstractArray = zeros(eltype(v), size(v)..., 5),
    rng = Random.default_rng()
)

    (vp, Sv, Svp, w, Sw) = eachslice(tmp, dims = 3)

    niters = size(αs,1)
    @assert niters - 1 == size(βs,1)
    @assert size(αs,2) == size(βs,2) == size(v,2)
    
    if all(iszero, v)
        randn!(rng, v)
    end

    _matrix_multiply!(Sv, S, v)
    β0 = @view βs[1,:]
    _vdot!(β0, v, Sv)
    @. β0 = sqrt(β0)
    @. v /= β0'
    @. Sv /= β0'

    _matrix_multiply!(w, A, Sv)
    α1 = @view αs[1,:]
    _vdot!(α1, w, Sv)
    @. w = w - α1' * v
    _matrix_multiply!(Sw, S, w)

    for n in 2:niters
        βnm1 = @view βs[n-1,:]
        _vdot!(βnm1, Sw, w)
        @. βnm1 = sqrt(βnm1)
        any(iszero, βnm1) && break
        @. vp = w / βnm1'
        @. Svp = Sw / βnm1'
        _matrix_multiply!(w, A, Svp)
        αn = @view αs[n,:]
        _vdot!(αn, w, Svp)
        @. w = w - αn' * vp - βnm1' * v
        _matrix_multiply!(Sw, S, w)
        @. v = vp
    end

    return [SymTridiagonal(view(αs,:,n), view(βs,:,n)) for n in axes(v,2)]
end

_vdot!(uv, u, v) = map!(_rdot, uv, eachcol(u), eachcol(v))
_rdot(u,v) = real(dot(u,v))