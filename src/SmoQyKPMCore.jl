module SmoQyKPMCore

using LinearAlgebra
using FFTW

import Base: resize!
import LinearAlgebra: mul!

# define private user function
include("utility.jl")

# functional api implementation of KPM algorithm
include("kpm.jl")

# define KPMExapnsion type for more convenient user interface
include("KPMExpansion.jl")

# ensure FFTW uses only a single thread
function __init__()
    FFTW.set_num_threads(1)
    return nothing
end

end
