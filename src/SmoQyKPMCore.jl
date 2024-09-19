module SmoQyKPMCore

using LinearAlgebra
using FFTW

# functional api implementation of KPM algorithm
include("kpm.jl")
export kpm_coefs, kpm_coefs!
export kpm_mul, kpm_mul!
export kpm_eval, kpm_eval!
export apply_jackson_kernel, apply_jackson_kernel!

include("KPMExpansion.jl")
export KPMExpansion
export update_kpmexpansion!, update_kpmexpansion_bounds!, update_kpmexpansion_order!

# ensure FFTW uses only a single thread
function __init__()
    FFTW.set_num_threads(1)
    return nothing
end

end
