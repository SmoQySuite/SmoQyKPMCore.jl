using SmoQyKPMCore
using Documenter
using Literate
using LinearAlgebra

DocMeta.setdocmeta!(SmoQyKPMCore, :DocTestSetup, :(using SmoQyKPMCore); recursive=true)

# package directory
pkg_dir = pkgdir(SmoQyKPMCore)
# examples directory
examples_dir = joinpath(pkg_dir, "examples")
# docs directory
docs_dir = joinpath(pkg_dir, "docs")
# docs/src directory
docs_src_dir = joinpath(docs_dir, "src")

# Process usage example
usage_literate = joinpath(examples_dir, "usage.jl")
Literate.markdown(usage_literate, docs_src_dir; credit = false)

makedocs(;
    clean = true,
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
