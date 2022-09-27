using ITensorPartialDiffEq
using Documenter

DocMeta.setdocmeta!(ITensorPartialDiffEq, :DocTestSetup, :(using ITensorPartialDiffEq); recursive=true)

makedocs(;
    modules=[ITensorPartialDiffEq],
    authors="Matthew Fishman <mfishman@flatironinstitute.org> and contributors",
    repo="https://github.com/mtfishman/ITensorPartialDiffEq.jl/blob/{commit}{path}#{line}",
    sitename="ITensorPartialDiffEq.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://mtfishman.github.io/ITensorPartialDiffEq.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/mtfishman/ITensorPartialDiffEq.jl",
    devbranch="main",
)
