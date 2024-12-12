function lanczos!(
    αs::CuVector, βs::CuVector, v::CuVector,
    A, S = I;
    tmp::CuArray = CUDA.zeros(eltype(v), length(v), 1, 5),
    rng = CUDA.default_rng()
)

    αs′ = reshape(αs, (length(αs), 1))
    βs′ = reshape(βs, (length(βs), 1))
    v′  = reshape(v,  (length(v),  1))
    SmoQyKPMCore._lanczos!(αs′, βs′, v′, A, S, tmp, rng)

    return nothing
end

function lanczos!(
    αs::CuMatrix, βs::CuMatrix, v::CuMatrix,
    A, S = I;
    tmp::CuArray = CUDA.zeros(eltype(v), size(v,1), size(v,2), 5),
    rng = CUDA.default_rng()
)

    SmoQyKPMCore._lanczos!(αs, βs, v, A, S, tmp, rng)

    return nothing
end