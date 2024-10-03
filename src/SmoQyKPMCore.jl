module SmoQyKPMCore

using LinearAlgebra
using Random
using FFTW

# functional api implementation of KPM algorithm
include("kpm.jl")
export kpm_coefs, kpm_coefs!
export kpm_moments, kpm_moments!
export kpm_density, kpm_density!
export kpm_dot
export kpm_mul, kpm_mul!
export kpm_eval, kpm_eval!
export apply_jackson_kernel, apply_jackson_kernel!

include("KPMExpansion.jl")
export KPMExpansion
export kpm_update!, kpm_update_bounds!, kpm_update_order!

include("lanczos.jl")
export lanczos, lanczos!

# ensure FFTW uses only a single thread
function __init__()
    FFTW.set_num_threads(1)
    return nothing
end

end
