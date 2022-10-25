using ITensorQTT
using Documenter

DocMeta.setdocmeta!(ITensorQTT, :DocTestSetup, :(using ITensorQTT); recursive=true)

makedocs(;
    modules=[ITensorQTT],
    authors="Matthew Fishman <mfishman@flatironinstitute.org> and contributors",
    repo="https://github.com/mtfishman/ITensorQTT.jl/blob/{commit}{path}#{line}",
    sitename="ITensorQTT.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://mtfishman.github.io/ITensorQTT.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/mtfishman/ITensorQTT.jl",
    devbranch="main",
)
