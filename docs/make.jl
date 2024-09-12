using SmoQyKPMCore
using Documenter

DocMeta.setdocmeta!(SmoQyKPMCore, :DocTestSetup, :(using SmoQyKPMCore); recursive=true)

makedocs(;
    modules=[SmoQyKPMCore],
    authors="Benjamin Cohen-Stead <benwcs@gmail.com>, Steven Johnston <sjohn145@utk.edu>, Kipton Barros <kbarros@lanl.gov>",
    sitename="SmoQyKPMCore.jl",
    format=Documenter.HTML(;
        canonical="https://SmoQySuite.github.io/SmoQyKPMCore.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/SmoQySuite/SmoQyKPMCore.jl",
    devbranch="main",
)
