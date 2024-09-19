using SmoQyKPMCore
using Documenter
using Literate
using LinearAlgebra

# generates script and notebook versions of tutorials based on literate example
function build_examples(example_sources, destdir)
    assetsdir = joinpath(fill("..", length(splitpath(destdir)))..., "assets")

    destpath = joinpath(@__DIR__, "src", destdir)
    isdir(destpath) && rm(destpath; recursive=true)

    # Transform each Literate source file to Markdown for subsequent processing by
    # Documenter.
    for source in example_sources
        # Extract "example" from "path/example.jl"
        name = splitext(basename(source))[1]
        
        # Preprocess each example by adding a notebook download link at the top. The
        # relative path is hardcoded according to the layout of `gh-pages` branch,
        # which is set up by `Documenter.deploydocs`.
        function preprocess(str)
            """
            # Download this example as [Jupyter notebook]($assetsdir/notebooks/$name.ipynb) or [Julia script]($assetsdir/scripts/$name.jl).

            """ * str
        end
        # Write to `src/$destpath/$name.md`
        Literate.markdown(source, destpath; preprocess, credit=false)
    end

    # Create Jupyter notebooks and Julia script for each Literate example. These
    # will be stored in the `assets/` directory of the hosted docs.
    for source in example_sources
        # Build notebooks
        Literate.notebook(source, notebooks_path; execute=false, credit=false)

        # Build julia scripts
        Literate.script(source, scripts_path; credit=false)
    end

    # Return paths `$destpath/$name.md` for each new Markdown file (relative to
    # `src/`)
    return map(example_sources) do source
        name = splitext(basename(source))[1]
        joinpath(destdir, "$name.md")
    end
end

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
        "API" => "api.md"
    ],
)

deploydocs(;
    repo="github.com/SmoQySuite/SmoQyKPMCore.jl",
    devbranch="main",
)
