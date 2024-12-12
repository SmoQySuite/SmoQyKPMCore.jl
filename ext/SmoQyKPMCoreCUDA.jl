module SmoQyKPMCoreCUDA

using SmoQyKPMCore
using CUDA
using Adapt
using LinearAlgebra
using FFTW

# implement DCT-II & III using FFTs via CUFFT
include("SmoQyKPMCoreCUDA/DCT.jl")

include("SmoQyKPMCoreCUDA/kpm_cuda.jl")

include("SmoQyKPMCoreCUDA/KPMExpansionCUDA.jl")

include("SmoQyKPMCoreCUDA/lanczos_cuda.jl")

end