@testitem "Fermi Function" begin

    using LinearAlgebra

    # define fermi function
    fermi(ϵ, β) = (1-tanh(β*ϵ/2))/2

    # define 1D chain dispersion relation
    eval_ϵ(k,t,μ) = -2*t*cos(k)-μ

    # define system size
    N = 16

    # define nearest-neighbor hopping amplitude
    t = 1.0

    # define chemical potential
    μ = 0.1

    # define inverse temperature
    β = 4.0

    # get relevant k-points given system size
    k = collect(range(start = 0.0, stop = 2π, length = N+1)[1:end-1])

    # evaluate energy at each k-point
    ϵ = eval_ϵ.(k,t,μ)

    # represent energies as diagonal matrix
    E = Diagonal(ϵ)

    # evaluate fermi function for each k-point
    fϵ = fermi.(ϵ, β)

    # define function that we will expand
    func(x) = fermi(x, β)

    # bound expansion spectrum
    bounds = (-3.0t-μ, 3.0t-μ)

    # define order of expansion
    M = 10

    # define kpm expansion
    kpm_expansion = KPMExpansion(func, bounds, M)

    # evaluate expansion as scalar expansion
    fϵ_scalar = [kpm_eval(ϵk, kpm_expansion) for ϵk in ϵ]

    # evaluate expansion as matrix expansion
    Fϵ_matrix = kpm_eval(E, kpm_expansion)
    fϵ_matrix = diag(Fϵ_matrix)

    # evaluate expansion as matrix expansion times vector
    v = ones(N)
    fϵ_vector = kpm_mul(E, kpm_expansion, v)

    # test that all three forms of the expansion agree with the analytic solution
    @test fϵ_scalar ≈ fϵ_matrix ≈ fϵ_vector

    # new expansion order
    M = 100

    # new bounds
    bounds = (-2.5t-μ, 2.5t-μ)

    # update expansion
    update_kpmexpansion!(kpm_expansion, func, bounds, M)

    # evaluate expansion as scalar expansion
    fϵ_scalar = [kpm_eval(ϵk, kpm_expansion) for ϵk in ϵ]

    # evaluate expansion as matrix expansion
    Fϵ_matrix = kpm_eval(E, kpm_expansion)
    fϵ_matrix = diag(Fϵ_matrix)

    # evaluate expansion as matrix expansion times vector
    v = ones(N)
    fϵ_vector = kpm_mul(E, kpm_expansion, v)

    # test that all three forms of the expansion agree with the analytic solution
    @test fϵ ≈ fϵ_scalar ≈ fϵ_matrix ≈ fϵ_vector
end