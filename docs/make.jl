using SmoQyKPMCore
using Documenter
using Literate
using LinearAlgebra

# Process usage example
name = "usage"
usage_source = joinpath(pkgdir(SmoQyKPMCore, "examples"), "$name.jl")
Literate.markdown(
    usage_source, joinpath(@__DIR__, "src");
    credit=false
)

DocMeta.setdocmeta!(SmoQyKPMCore, :DocTestSetup, :(using SmoQyKPMCore); recursive=true)

makedocs(;
    clean = false,
    modules=[SmoQyKPMCore],
    authors="Benjamin Cohen-Stead <benwcs@gmail.com>, Steven Johnston <sjohn145@utk.edu>, Kipton Barros <kbarros@lanl.gov>",
    sitename="SmoQyKPMCore.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://SmoQySuite.github.io/SmoQyKPMCore.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Usage" => "usage.md",
        "API" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/SmoQySuite/SmoQyKPMCore.jl",
    devbranch="main",
)
