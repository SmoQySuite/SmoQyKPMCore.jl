var documenterSearchIndex = {"docs":
[{"location":"api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Note that for many methods there are two verions, one that relies on taking an instance of the KPMExpansion type as an argument, and a lower level one that does not.","category":"page"},{"location":"api/","page":"API","title":"API","text":"KPMExpansion\nkpm_update!\nkpm_update_bounds!\nkpm_update_order!\nkpm_coefs\nkpm_coefs!\nkpm_moments\nkpm_moments!\nkpm_density\nkpm_density!\nkpm_dot\nkpm_mul\nkpm_mul!\nkpm_eval\nkpm_eval!\napply_jackson_kernel\napply_jackson_kernel!\nlanczos\nlanczos!","category":"page"},{"location":"api/","page":"API","title":"API","text":"KPMExpansion\nKPMExpansion(::Function, ::Any, ::Int, ::Int)\nkpm_update!\nkpm_update_bounds!\nkpm_update_order!\nkpm_coefs\nkpm_coefs!\nkpm_moments\nkpm_moments!\nkpm_density\nkpm_density!\nkpm_dot\nkpm_mul\nkpm_mul!\nkpm_eval\nkpm_eval!\napply_jackson_kernel\napply_jackson_kernel!\nlanczos\nlanczos!","category":"page"},{"location":"api/#SmoQyKPMCore.KPMExpansion","page":"API","title":"SmoQyKPMCore.KPMExpansion","text":"mutable struct KPMExpansion{T<:AbstractFloat, Tfft<:FFTW.r2rFFTWPlan}\n\nA type to represent the Chebyshev polynomial expansion used in the KPM algorithm.\n\nFields\n\nM::Int: Order of the Chebyshev polynomial expansion.\nbounds::NTuple{2,T}: Bounds on eigenspectrum in the KPM algorithm.\nbuf::Vector{T}: The first M elements of this vector of the Chebyshev expansion coefficients.\nr2rplan::Tfft: Plan for performing DCT to efficiently calculate the Chebyshev expansion coefficients via Chebyshev-Gauss quadrature.\n\n\n\n\n\n","category":"type"},{"location":"api/#SmoQyKPMCore.KPMExpansion-Tuple{Function, Any, Int64, Int64}","page":"API","title":"SmoQyKPMCore.KPMExpansion","text":"KPMExpansion(func::Function, bounds, M::Int, N::Int = 2*M)\n\nInitialize an instance of the KPMExpansion type to approximate the univariate function func, called as func(x), with a order M Chebyshev polynomial expansion on the interval bounds[1] < x bounds[2]. Here, N ≥ M is the number of points at which func is evaluated on that specified interval, which are then used to calculate the expansion coeffiencents via Chebyshev-Gauss quadrature.\n\n\n\n\n\n","category":"method"},{"location":"api/#SmoQyKPMCore.kpm_update!","page":"API","title":"SmoQyKPMCore.kpm_update!","text":"kpm_update!(\n    kpm_expansion::KPMExpansion{T}, func::Function, bounds, M::Int, N::Int = 2*M\n) where {T<:AbstractFloat}\n\nIn-place update an instance of KPMExpansion to reflect new values for eigenspectrum bounds, expansion order M, and number of points at which the expanded function is evaluated when computing the expansion coefficients. This includes recomputing the expansion coefficients.\n\n\n\n\n\n","category":"function"},{"location":"api/#SmoQyKPMCore.kpm_update_bounds!","page":"API","title":"SmoQyKPMCore.kpm_update_bounds!","text":"kpm_update_bounds!(\n    kpm_expansion::KPMExpansion{T}, func::Function, bounds\n) where {T<:AbstractFloat}\n\nIn-place update an instance of KPMExpansion to reflect new values for eigenspectrum bounds, recomputing the expansion coefficients.\n\n\n\n\n\n","category":"function"},{"location":"api/#SmoQyKPMCore.kpm_update_order!","page":"API","title":"SmoQyKPMCore.kpm_update_order!","text":"kpm_update_order!(\n    kpm_expansion::KPMExpansion, func::Function, M::Int, N::Int = 2*M\n)\n\nIn-place update the expansion order M for an instance of KPMExpansion, recomputing the expansion coefficients. It is also possible to udpate the number of point N the function func is evaluated at to calculate the expansion coefficients.\n\n\n\n\n\n","category":"function"},{"location":"api/#SmoQyKPMCore.kpm_coefs","page":"API","title":"SmoQyKPMCore.kpm_coefs","text":"kpm_coefs(func::Function, bounds, M::Int, N::Int = 2*M)\n\nCalculate and return the Chebyshev expansion coefficients. Refer to kpm_coefs! for more information.\n\n\n\n\n\n","category":"function"},{"location":"api/#SmoQyKPMCore.kpm_coefs!","page":"API","title":"SmoQyKPMCore.kpm_coefs!","text":"kpm_coefs!(\n    coefs::AbstractVector{T}, func::Function, bounds,\n    buf::Vector{T} = zeros(T, 2*length(coefs)),\n    r2rplan = FFTW.plan_r2r!(zeros(T, 2*length(coefs)), FFTW.REDFT10)\n) where {T<:AbstractFloat}\n\nCalculate and record the Chebyshev polynomial expansion coefficients to order M in the vector ceofs for the function func on the interval (bounds[1], bounds[2]). Let length(buf) be the number of evenly spaced points on the interval for which func is evaluated when performing Chebyshev-Gauss quadrature to compute the Chebyshev polynomial expansion coefficients.\n\n\n\n\n\n","category":"function"},{"location":"api/#SmoQyKPMCore.kpm_moments","page":"API","title":"SmoQyKPMCore.kpm_moments","text":"kpm_moments(\n    M::Int, A, bounds, R::T,\n    tmp = zeros(eltype(R), size(R)..., 3)\n) where {T<:AbstractVecOrMat}\n\nIf R is a vector, calculate and return the first M moments as\n\nmu_m = langle R  T_m(A^prime)  R rangle\n\nwhere A^prime is the rescaled version of A using the bounds. If R is a matrix then calculate the moments as\n\nmu_m = frac1N sum_n=1^N langle R_n  T_m(A^prime)  R_n rangle\n\nwhere  R_n rangle is the n'th column of R.\n\n\n\n\n\nkpm_moments(\n    M::Int, A, bounds, U::T, V::T,\n    tmp = zeros(eltype(V), size(V)..., 3)\n) where {T<:AbstractVecOrMat}\n\nIf U and V are vector then calculate and return the first M moments\n\nmu_m = langle U  T_m(A^prime)  V rangle\n\nwhere A^prime is the rescaled version of A using the bounds. If U and V are matrices then calculate the moments as\n\nmu_m = frac1N sum_n=1^N langle U_n  T_m(A^prime)  V_n rangle\n\nwhere  U_n rangle and   V_n rangle are are the n'th columns of each matrix.\n\n\n\n\n\n","category":"function"},{"location":"api/#SmoQyKPMCore.kpm_moments!","page":"API","title":"SmoQyKPMCore.kpm_moments!","text":"kpm_moments!(\n    μ::AbstractVector, A, bounds, R::T,\n    tmp = zeros(eltype(R), size(R)..., 3)\n) where {T<:AbstractVecOrMat}\n\nIf R is a vector, calculate and return the first M moments as\n\nmu_m = langle R  T_m(A^prime)  R rangle\n\nwhere A^prime is the rescaled version of A using the bounds. If R is a matrix then calculate the moments as\n\nmu_m = frac1N sum_n=1^N langle R_n  T_m(A^prime)  R_n rangle\n\nwhere  R_n rangle is the n'th column of R.\n\n\n\n\n\nkpm_moments!(\n    μ::AbstractVector, A, bounds, U::T, V::T,\n    tmp = zeros(eltype(V), size(V)..., 3)\n) where {T<:AbstractVecOrMat}\n\nIf U and V are vector then calculate and return the first M moments\n\nmu_m = langle U  T_m(A^prime)  V rangle\n\nwhere A^prime is the rescaled version of A using the bounds. If U and V are matrices then calculate the moments as\n\nmu_m = frac1N sum_n=1^N langle U_n  T_m(A^prime)  V_n rangle\n\nwhere  U_n rangle and   V_n rangle are are the n'th columns of each matrix.\n\n\n\n\n\n","category":"function"},{"location":"api/#SmoQyKPMCore.kpm_density","page":"API","title":"SmoQyKPMCore.kpm_density","text":"kpm_density(\n    N::Int, μ::T, W\n) where {T<:AbstractVector}\n\nGiven the KPM moments mu, evaluate the corresponding spectral density rho(epsilon) at N energy points epsilon, where W is the spectral bandwidth or the bounds of the spectrum. This function returns both rho(epsilon) and epsilon.\n\n\n\n\n\n","category":"function"},{"location":"api/#SmoQyKPMCore.kpm_density!","page":"API","title":"SmoQyKPMCore.kpm_density!","text":"kpm_density!(\n    ρ::AbstractVector{T},\n    ϵ::AbstractVector{T},\n    μ::AbstractVector{T},\n    bounds,\n    r2rplan = FFTW.plan_r2r!(ρ, FFTW.REDFT01)\n) where {T<:AbstractFloat}\n\nGiven the KPM moments mu, calculate the spectral density rho(epsilon) and corresponding energies epsilon where it is evaluated. Here bounds is bounds the eigenspecturm, such that the bandwidth is given by W = bounds[2] - bounds[1].\n\n\n\n\n\nkpm_density!(\n    ρ::AbstractVector{T},\n    ϵ::AbstractVector{T},\n    μ::AbstractVector{T},\n    W::T,\n    r2rplan = FFTW.plan_r2r!(ρ, FFTW.REDFT01),\n) where {T<:AbstractFloat}\n\nGiven the KPM moments mu, calculate the spectral density rho(epsilon) and corresponding energies epsilon where it is evaluated. Here W is the spectral bandwidth.\n\n\n\n\n\n","category":"function"},{"location":"api/#SmoQyKPMCore.kpm_dot","page":"API","title":"SmoQyKPMCore.kpm_dot","text":"kpm_dot(\n    A, coefs::AbstractVector, bounds, R::T,\n    tmp = zeros(eltype(v), size(v)..., 3)\n) where {T<:AbstractVecOrMat}\n\nIf R is a single vector, then calculate the inner product\n\nbeginalign*\nS  = langle R  F(A^prime)  R rangle \nS  = sum_m=1^M langle R  c_m T_m(A^prime)  R rangle\nendalign*\n\nwhere A^prime is the scaled version of A using the bounds. If R is a matrix, then calculate\n\nbeginalign*\nS  = langle R  F(A^prime)  R rangle \nS  = frac1N sum_n=1^N sum_m=1^M langle R_n  c_m T_m(A^prime)  R_n rangle\nendalign*\n\nwhere  R_n rangle is a column of R.\n\n\n\n\n\nkpm_dot(\n    A, coefs::AbstractVector, bounds, U::T, V::T,\n    tmp = zeros(eltype(v), size(v)..., 3)\n) where {T<:AbstractVecOrMat}\n\nIf U and V are single vectors, then calculate the inner product\n\nbeginalign*\nS  = langle U  F(A^prime)  V rangle \n   = sum_m=1^M langle U  c_m T_m(A^prime)  V rangle\nendalign*\n\nwhere A^prime is the scaled version of A using the bounds. If U and V are matrices, then calculate\n\nbeginalign*\nS  = langle U  F(A^prime)  V rangle \n   = frac1N sum_n=1^N sum_m=1^M langle U_n  c_m T_m(A^prime)  V_n rangle\nendalign*\n\nwhere  U_n rangle and  V_n rangle are the columns of each matrix.\n\n\n\n\n\nkpm_dot(\n    A, kpm_expansion::KPMExpansion, R::T,\n    tmp = zeros(eltype(R), size(R)..., 3)\n) where {T<:AbstractVecOrMat}\n\nIf R is a single vector, then calculate the inner product\n\nbeginalign*\nS  = langle R  F(A)  R rangle \nS  = sum_m=1^M langle R  c_m T_m(A^prime)  R rangle\nendalign*\n\nwhere A^prime is the scaled version of A using the bounds. If R is a matrix, then calculate\n\nbeginalign*\nS  = langle R  F(A)  R rangle \nS  = frac1N sum_n=1^N sum_m=1^M langle R_n  c_m T_m(A^prime)  R_n rangle\nendalign*\n\nwhere  R_n rangle is a column of R.\n\n\n\n\n\nkpm_dot(\n    A, kpm_expansion::KPMExpansion, U::T, V::T,\n    tmp = zeros(eltype(V), size(V)..., 3)\n) where {T<:AbstractVecOrMat}\n\nIf U and V are single vectors, then calculate the inner product\n\nbeginalign*\nS  = langle U  F(A)  V rangle \n   = sum_m=1^M langle U  c_m T_m(A^prime)  V rangle\nendalign*\n\nwhere A^prime is the scaled version of A using the bounds. If U and V are matrices, then calculate\n\nbeginalign*\nS  = langle U  F(A)  V rangle \n   = frac1N sum_n=1^N sum_m=1^M langle U_n  c_m T_m(A^prime)  V_n rangle\nendalign*\n\nwhere  U_n rangle and  V_n rangle are the columns of each matrix.\n\n\n\n\n\n","category":"function"},{"location":"api/#SmoQyKPMCore.kpm_mul","page":"API","title":"SmoQyKPMCore.kpm_mul","text":"kpm_mul(\n    A, coefs::AbstractVector, bounds, v::T, tmp = zeros(eltype(v), size(v)..., 3)\n) where {T<:AbstractVecOrMat}\n\nEvaluate and return the vector v^prime = F(A) cdot v where F(A) is represented by the Chebyshev expansion. For more information refer to kpm_mul!.\n\n\n\n\n\nkpm_mul(A, kpm_expansion::KPMExpansion, v::T) where {T<:AbstractVecOrMat}\n\nEvaluate and return the vector v^prime = F(A) cdot v where F(A) is represented by the Chebyshev expansion. For more information refer to kpm_mul!.\n\n\n\n\n\n","category":"function"},{"location":"api/#SmoQyKPMCore.kpm_mul!","page":"API","title":"SmoQyKPMCore.kpm_mul!","text":"kpm_mul!(\n    v′::T, A, coefs::AbstractVector, bounds, v::T,\n    tmp = zeros(eltype(v), size(v)..., 3)\n) where {T<:AbstractVecOrMat}\n\nEvaluates v^prime = F(A) cdot v, writing the result to v′, where F(A) is represented by the Chebyshev expansion. Here A is either a function that can be called as A(u,v) to evaluate u = Acdot v, modifying u in-place, or is a type for which the operation mul!(u, A, v) is defined. The vector coefs contains Chebyshev expansion coefficients to approximate F(A), where the eigenspectrum of A is contained in the interval (bounds[1], bounds[2]) specified by the bounds argument. The vector v is vector getting multiplied by the Chebyshev expansion for F(A). Lastly, tmp is an array used to avoid dynamic memory allocations.\n\n\n\n\n\nkpm_mul!(\n    v′::T, A, kpm_expansion::KPMExpansion, v::T,\n    tmp = zeros(eltype(v), size(v)..., 3)\n) where {T<:AbstractVecOrMat}\n\nEvaluates v^prime = F(A) cdot v, writing the result to v′, where F(A) is represented by the Chebyshev expansion. Here A is either a function that can be called as A(u,v) to evaluate u = Acdot v, modifying u in-place, or is a type for which the operation mul!(u, A, v) is defined. Lastly, the array tmp is used to avoid dynamic memory allocations.\n\n\n\n\n\n","category":"function"},{"location":"api/#SmoQyKPMCore.kpm_eval","page":"API","title":"SmoQyKPMCore.kpm_eval","text":"kpm_eval(x::AbstractFloat, coefs, bounds)\n\nEvaluate F(x) where x is real number in the interval bounds[1] < x < bound[2], and the function F(bullet) is represented by a Chebyshev expansion with coefficients given by the vector coefs.\n\n\n\n\n\nkpm_eval(A::AbstractMatrix, coefs, bounds)\n\nEvaluate and return the matrix F(A) where A is an operator with strictly real eigenvalues that fall in the interval (bounds[1], bounds[2]) specified by the bounds argument, and the function F(bullet) is represented by a Chebyshev expansion with coefficients given by the vector coefs.\n\n\n\n\n\nkpm_eval(x::T, kpm_expansion::KPMExpansion{T}) where {T<:AbstractFloat}\n\nEvaluate F(x) where x is real number in the interval bounds[1] < x < bound[2], and the function F(bullet) is represented by a Chebyshev expansion with coefficients given by the vector coefs.\n\n\n\n\n\nkpm_eval(A::AbstractMatrix{T}, kpm_expansion::KPMExpansion{T}) where {T<:AbstractFloat}\n\nEvaluate and return the matrix F(A) where A is an operator with strictly real eigenvalues and the function F(bullet) is represented by a Chebyshev expansion with coefficients given by the vector coefs.\n\n\n\n\n\n","category":"function"},{"location":"api/#SmoQyKPMCore.kpm_eval!","page":"API","title":"SmoQyKPMCore.kpm_eval!","text":"kpm_eval!(\n    F::AbstractMatrix, A, coefs::AbstractVector, bounds,\n    tmp = zeros(eltype(F), size(F)..., 3)\n)\n\nEvaluate and write the matrix F(A) to F, where A is an operator with strictly real eigenvalues that fall in the interval (bounds[1], bounds[2]) specified by the bounds argument, and the function F(bullet) is represented by a Chebyshev expansion with coefficients given by the vector coefs. Lastly, tmp is used to avoid dynamic memory allocations.\n\n\n\n\n\nkpm_eval!(\n    F::AbstractMatrix, A, kpm_expansion::KPMExpansion,\n    tmp = zeros(eltype(F), size(F)..., 3)\n)\n\nEvaluate and write the matrix F(A) to F, where A is an operator with strictly real eigenvalues and the function F(bullet) is represented by a Chebyshev expansion with coefficients given by the vector coefs. Lastly, the array tmp is used to avoid dynamic memory allocations.\n\n\n\n\n\n","category":"function"},{"location":"api/#SmoQyKPMCore.apply_jackson_kernel","page":"API","title":"SmoQyKPMCore.apply_jackson_kernel","text":"apply_jackson_kernel(coefs)\n\nReturn the Chebyshev expansion coefficients transformed by the Jackson kernel.\n\n\n\n\n\n","category":"function"},{"location":"api/#SmoQyKPMCore.apply_jackson_kernel!","page":"API","title":"SmoQyKPMCore.apply_jackson_kernel!","text":"apply_jackson_kernel!(coefs)\n\nModify the Chebyshev expansion coefficients by applying the Jackson kernel to them.\n\n\n\n\n\napply_jackson_kernel!(kpm_expansion::KPMExpansion)\n\nModify the Chebyshev expansion coefficients by applying the Jackson kernel to them.\n\n\n\n\n\n","category":"function"},{"location":"api/#SmoQyKPMCore.lanczos","page":"API","title":"SmoQyKPMCore.lanczos","text":"lanczos(niters, v, A, S = I, rng = Random.default_rng())\n\nUse niters Lanczos iterations to find a truncated tridiagonal representation of Acdot S, up to similarity transformation. Here, A is any Hermitian matrix, while S is both Hermitian and positive definite. Traditional Lanczos uses the identity matrix, S = I. The extension to non-identity matrices S is as follows: Each matrix-vector product Acdot v becomes (A S) cdot v, and each vector inner product w^dagger cdot v becomes w^dagger cdot S cdot v. The implementation below follows Wikipedia, and is the most stable of the four variants considered by Paige [1]. This implementation introduces additional vector storage so that each Lanczos iteration requires only one matrix-vector multiplication for A and S, respectively.\n\nThis function returns a SymTridiagonal matrix. Note that the eigmin and eigmax routines have specialized implementations for a SymTridiagonal matrix type.\n\nSimilar generalizations of Lanczos have been considered in [2] and [3].\n\n\n\n1. C. C. Paige, IMA J. Appl. Math., 373-381 (1972),\n\nhttps://doi.org/10.1093%2Fimamat%2F10.3.373.\n\n2. H. A. van der Vorst, Math. Comp. 39, 559-561 (1982),\n\nhttps://doi.org/10.1090/s0025-5718-1982-0669648-0\n\n3. M. Grüning, A. Marini, X. Gonze, Comput. Mater. Sci. 50, 2148-2156 (2011),\n\nhttps://doi.org/10.1016/j.commatsci.2011.02.021.\n\n\n\n\n\n","category":"function"},{"location":"api/#SmoQyKPMCore.lanczos!","page":"API","title":"SmoQyKPMCore.lanczos!","text":"lanczos!(\n    αs::AbstractVector, βs::AbstractVector, v::AbstractVector,\n    A, S = I,\n    tmp::AbstractMatrix = zeros(eltype(v), length(v), 5);\n    rng = Random.default_rng()\n)\n\nlanczos!(\n    αs::AbstractMatrix, βs::AbstractMatrix, v::AbstractMatrix,\n    A, S = I,\n    tmp::AbstractArray = zeros(eltype(v), size(v)..., 5);\n    rng = Random.default_rng()\n)\n\nUse Lanczos iterations to find a truncated tridiagonal representation of Acdot S, up to similarity transformation. Here, A is any Hermitian matrix, while S is both Hermitian and positive definite. Traditional Lanczos uses the identity matrix, S = I. The extension to non-identity matrices S is as follows: Each matrix-vector product Acdot v becomes (A S) cdot v, and each vector inner product w^dagger cdot v becomes w^dagger cdot S cdot v. The implementation below follows Wikipedia, and is the most stable of the four variants considered by Paige [1]. This implementation introduces additional vector storage so that each Lanczos iteration requires only one matrix-vector multiplication for A and S, respectively.\n\nThe number of Lanczos iterations performed equals niters = length(αs), and niters - 1 == length(βs). This function returns a SymTridiagonal matrix based on the contents of the vectors αs and βs. Note that the eigmin and eigmax routines have specialized implementations for a SymTridiagonal matrix type.\n\nNote that if αs, βs and v are all matrices, then each column of v is treated as a seperate vector, and a vector of SymTridiagonal of length size(v,2) will be returned.\n\nSimilar generalizations of Lanczos have been considered in [2] and [3].\n\n\n\n1. C. C. Paige, IMA J. Appl. Math., 373-381 (1972),\n\nhttps://doi.org/10.1093%2Fimamat%2F10.3.373.\n\n2. H. A. van der Vorst, Math. Comp. 39, 559-561 (1982),\n\nhttps://doi.org/10.1090/s0025-5718-1982-0669648-0\n\n3. M. Grüning, A. Marini, X. Gonze, Comput. Mater. Sci. 50, 2148-2156 (2011),\n\nhttps://doi.org/10.1016/j.commatsci.2011.02.021.\n\n\n\n\n\n","category":"function"},{"location":"usage/","page":"Usage","title":"Usage","text":"EditURL = \"../../examples/usage.jl\"","category":"page"},{"location":"usage/#Usage","page":"Usage","title":"Usage","text":"","category":"section"},{"location":"usage/","page":"Usage","title":"Usage","text":"Here we demonstrate the basic usage for the SmoQyKPMCore package. Let us first import the relevant packages we will want to use in this example.","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"using LinearAlgebra\nusing SparseArrays","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"Package for making figures","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"using CairoMakie\nCairoMakie.activate!(type = \"svg\")\n\nusing SmoQyKPMCore","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"In this usage example we will consider a 1D chain tight-binding model","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"hatH = -t sum_isigma (hatc_i+1sigma^dagger hatc_isigma + rm Hc)\n        = sum_sigma hatc_isigma^dagger H_ij hatc_jsigma","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"where hatc_isigma^dagger  (hatc_isigma) creates (annihilates) a spin-sigma electron on site i in the lattice, and t is the nearest-neighbor hopping amplitude.","category":"page"},{"location":"usage/#Density-Matrix-Approximation","page":"Usage","title":"Density Matrix Approximation","text":"","category":"section"},{"location":"usage/","page":"Usage","title":"Usage","text":"With H the Hamiltonian matrix, the corresponding density matrix is given by rho = f(H) where","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"f(epsilon) = frac11+e^beta (epsilon-mu) = frac12 left 1 + tanhleft(tfracbeta(epsilon-mu)2right) right","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"is the Fermi function, with beta = 1T the inverse temperature and mu the chemical potential. Here we will applying the kernel polynomial method (KPM) algorithm to approximate the density matrix rho by a Chebyshev polynomial expansion.","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"Let us first define the relevant model parameter values.","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"# Nearest-neighbor hopping amplitude.\nt = 1.0\n\n# Chemical potential.\nμ = 0.0\n\n# System size.\nL = 16\n\n# Inverse temperature.\nβ = 4.0;\nnothing #hide","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"Now we need to construct the Hamiltonian matrix H.","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"H = zeros(L,L)\nfor i in 1:L\n    j = mod1(i+1,L)\n    H[j,i], H[i,j] = -t, -t\nend\nH","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"We also need to define the Fermi function.","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"# Fermi function.\nfermi(ϵ, μ, β) = (1+tanh(β*(ϵ-μ)/2))/2","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"Let us calculate the exact density matrix rho so that we may asses the accuracy of the KPM method.","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"# Diagonalize the Hamiltonian matrix.\nϵ, U = eigen(H)\n\n# Calculate the density matrix.\nρ = U * Diagonal(fermi.(ϵ, μ, β)) * adjoint(U)","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"Now let use the KPMExpansion type to approximate rho. We will need to define the order M of the expansion and give approximate bounds for the eigenspectrum of H, making sure to overestimate the the true interval spanned by the eigenvalues of H. Because we are considering the simple non-interacting model, the exact eigenspectrum of H is known and spans the interval epsilon in -2t 2t","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"# Define eigenspectrum bounds.\nbounds = (-3.0t, 3.0t)\n\n# Define order of Chebyshev expansion used in KPM approximation.\nM = 10\n\n# Initialize expansion.\nkpm_expansion = KPMExpansion(x -> fermi(x, μ, β), bounds, M);\nnothing #hide","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"Now let us test and see how good a job it does approximating the density matrix rho, using the kpm_eval! function to do so.","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"# Initialize KPM density matrix approxiatmion\nρ_kpm = similar(H)\n\n# Initialize additional matrices to avoid dynamic memory allocation.\nmtmp = zeros(L,L,3)\n\n# Calculate KPM density matrix approximation.\nkpm_eval!(ρ_kpm, H, kpm_expansion, mtmp)\n\n# Check how good an approximation it is.\nprintln(\"Matrix Error = \", norm(ρ_kpm - ρ)/norm(ρ) )","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"It is also possible to efficiently multiply vectors by KPM approximation to the density matrix rho. Let us test this functionality with the kpm_mul! on a random vector, seeing how accurate the result is.","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"# Initialize random vector.\nv = randn(L)\n\n# Calculate exact product with density matrix.\nρv = ρ * v\n\n# Initialize vector to contain approximate product with densith matrix.\nρv_kpm = similar(v)\n\n# Initialize to avoid dynamic memory allocation.\nvtmp = zeros(L, 3)\n\n# Calculate approximate product with density matrix.\nkpm_mul!(ρv_kpm, H, kpm_expansion, v, vtmp)\n\n# Check how good the approximation is.\nprintln(\"Vector Error = \", norm(ρv_kpm - ρv) / norm(ρv) )","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"Let us now provide the KPM approximation with better bounds on the eigenspectrum and also increase the order the of the expansion and see how the result improves. We will use the kpm_update! function to update the KPMExpansion in-place.","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"# Define eigenspectrum bounds.\nbounds = (-2.5t, 2.5t)\n\n# Define order of Chebyshev expansion used in KPM approximation.\nM = 100\n\n# Update KPM approximation.\nkpm_update!(kpm_expansion, x -> fermi(x, μ, β), bounds, M)\n\n# Calculate KPM density matrix approximation.\nkpm_eval!(ρ_kpm, H, kpm_expansion, mtmp)\n\n# Check how good an approximation it is.\nprintln(\"Matrix Error = \", norm(ρ_kpm - ρ)/norm(ρ) )\n\n# Calculate approximate product with density matrix.\nkpm_mul!(ρv_kpm, H, kpm_expansion, v, vtmp)\n\n# Check how good the approximation is.\nprintln(\"Vector Error = \", norm(ρv_kpm - ρv) / norm(ρv) )","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"Now let me quickly demonstrate how we may approximate the the trace rho using a set of random vector R_n using the kpm_dot function.","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"# Calculate exact trace.\ntrρ = tr(ρ)\nprintln(\"Exact trace = \", trρ)\n\n# Number of random vectors\nN = 100\n\n# Initialize random vectors.\nR = randn(L, N)\n\n# Initialize array to avoid dynamic memory allocation\nRtmp = zeros(L, N, 3)\n\n# Approximate trace of density matrix.\ntrρ_approx = kpm_dot(H, kpm_expansion, R, Rtmp)\nprintln(\"Approximate trace = \", trρ_approx)\n\n# Report the error in the approximation.\nprintln(\"Trace esitimate error = \", abs(trρ_approx - trρ)/trρ)","category":"page"},{"location":"usage/#Density-of-States-Approximation","page":"Usage","title":"Density of States Approximation","text":"","category":"section"},{"location":"usage/","page":"Usage","title":"Usage","text":"Next let us demonstrate how we may approximate the density of states mathcalN(epsilon) for a 1D chain tight-binding model. The first step a very larger Hamiltonian matrix H which we represent as a spare matrix.","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"# Size of 1D chain considered\nL = 10_000\n\n# Construct sparse Hamiltonian matrix.\nrows = Int[]\ncols = Int[]\nfor i in 1:L\n    j = mod1(i+1, L)\n    push!(rows,i)\n    push!(cols,j)\n    push!(rows,j)\n    push!(cols,i)\nend\nvals = fill(-t, length(rows))\nH = sparse(rows, cols, vals)","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"Let us now calculate the fist M moments mu_m using N random vectors using the function kpm_moments.","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"# Number of moments to calculate.\nM = 250\n\n# Number of random vectors used to approximate moments.\nN = 100\n\n# Initialize random vectors.\nR = randn(L, N)\n\n# Calculate the moments.\nμ_kpm = kpm_moments(M, H, bounds, R);\nnothing #hide","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"Having calculate the moments, let us next evaluate the density of states mathcalN(epsilon) at P points with the kpm_density function.","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"# Number of points at which to evaluate the density of states.\nP = 1000\n\n# Evaluate density of states.\ndos, ϵ = kpm_density(P, μ_kpm, bounds);\nnothing #hide","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"Without regularization, the approximate for the density of states generated above will have clearly visible Gibbs oscillations. To partially suppress these artifacts, we apply the Jackson kernel to the moments mu using the apply_jackson_kernel function.","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"# Apply Jackson kernel.\nμ_jackson = apply_jackson_kernel(μ_kpm)\n\n# Evaluate density of states.\ndos_jackson, ϵ_jackson = kpm_density(P, μ_jackson, bounds);\nnothing #hide","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"Have approximated the density of states, let us now plot the result.","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"fig = Figure(\n    size = (700, 400),\n    fonts = (; regular= \"CMU Serif\"),\n    figure_padding = 10\n)\n\nax = Axis(\n    fig[1, 1],\n    aspect = 7/4,\n    xlabel = L\"\\epsilon\", ylabel = L\"\\mathcal{N}(\\epsilon)\",\n    xticks = range(start = bounds[1], stop = bounds[2], length = 5),\n    xlabelsize = 30, ylabelsize = 30,\n    xticklabelsize = 24, yticklabelsize = 24,\n)\n\nlines!(\n    ϵ, dos,\n    linewidth = 2.0,\n    alpha = 1.0,\n    color = :red,\n    linestyle = :solid,\n    label = \"Dirichlet\"\n)\n\nlines!(\n    ϵ_jackson, dos_jackson,\n    linewidth = 3.0,\n    alpha = 1.0,\n    color = :black,\n    linestyle = :solid,\n    label = \"Jackson\"\n)\n\nxlims!(\n    ax, bounds[1], bounds[2]\n)\n\nylims!(\n    ax, 0.0, 1.05 * maximum(dos)\n)\n\naxislegend(\n    ax, halign = :center, valign = :top, labelsize = 30\n)\n\nfig","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = SmoQyKPMCore","category":"page"},{"location":"#SmoQyKPMCore","page":"Home","title":"SmoQyKPMCore","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for SmoQyKPMCore package. The SmoQyKPMCore package implements and exports an optimized, low-level implementation of the Kernel Polynomial Method (KPM) algorithm for approximating functions of operators with strictly real, bounded eigenvalues via a Chebyshev polynomial expansion.","category":"page"},{"location":"#Funding","page":"Home","title":"Funding","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The development of this code was supported by the U.S. Department of Energy, Office of Science, Basic Energy Sciences, under Award Number DE-SC0022311.","category":"page"}]
}
