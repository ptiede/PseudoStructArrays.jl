using PseudoStructArrays
using Documenter

DocMeta.setdocmeta!(PseudoStructArrays, :DocTestSetup, :(using PseudoStructArrays); recursive=true)

makedocs(;
    modules=[PseudoStructArrays],
    authors="Paul Tiede <ptiede91@gmail.com> and contributors",
    sitename="PseudoStructArrays.jl",
    format=Documenter.HTML(;
        canonical="https://ptiede.github.io/PseudoStructArrays.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ptiede/PseudoStructArrays.jl",
    devbranch="main",
)
