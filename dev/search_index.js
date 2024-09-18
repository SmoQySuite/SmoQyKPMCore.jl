var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = SmoQyKPMCore","category":"page"},{"location":"#SmoQyKPMCore","page":"Home","title":"SmoQyKPMCore","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for SmoQyKPMCore.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [SmoQyKPMCore]","category":"page"},{"location":"#SmoQyKPMCore.kpm_coefs","page":"Home","title":"SmoQyKPMCore.kpm_coefs","text":"kpm_coefs(func::Function, bounds, M::Int, N::Int = 2*M)\n\nCalculate and return the Chebyshev expansion coefficients. Refer to kpm_coefs! for more information.\n\n\n\n\n\n","category":"function"},{"location":"#SmoQyKPMCore.kpm_coefs!-Union{Tuple{T}, Tuple{AbstractVector{T}, Function, Any}, Tuple{AbstractVector{T}, Function, Any, Vector{T}}, Tuple{AbstractVector{T}, Function, Any, Vector{T}, FFTW.r2rFFTWPlan}} where T<:AbstractFloat","page":"Home","title":"SmoQyKPMCore.kpm_coefs!","text":"kpm_coefs!(\n    coefs::AbstractVector{T}, func::Function, bounds,\n    buf::Vector{T} = zeros(T, 2*length(coefs)),\n    r2rplan::FFTW.r2rFFTWPlan = FFTW.plan_r2r!(zeros(T, 2*length(coefs)), FFTW.REDFT10)\n) where {T<:AbstractFloat}\n\nCalculate and record the Chebyshev expansion coefficients to order M in the vector ceofs for the function func on the bounded interval (bounds[1], bounds[2]). Let N be the number of evenly spaced points on the specified interval for which func is evaluated when performing Chebyshev-Gauss quadrature to compute the coefficients.\n\n\n\n\n\n","category":"method"}]
}
