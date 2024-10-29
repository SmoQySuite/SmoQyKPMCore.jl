# SmoQyKPMCore

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://SmoQySuite.github.io/SmoQyKPMCore.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://SmoQySuite.github.io/SmoQyKPMCore.jl/dev/)
[![Build Status](https://github.com/SmoQySuite/SmoQyKPMCore.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/SmoQySuite/SmoQyKPMCore.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/SmoQySuite/SmoQyKPMCore.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/SmoQySuite/SmoQyKPMCore.jl)

This package implements and exports an optimized, low-level implementation of the Kernel Polynomial Method (KPM) algorithm for
approximating functions of operators with strictly real, bounded eigenvalues via a Chebyshev polynomial expansion.

## Installation

To install [SmoQyKPMCore](https://github.com/SmoQySuite/SmoQyKPMCore.jl.git),
simply open the Julia REPL and run the commands
```julia
julia> ]
pkg> add SmoQyKPMCore
```
or equivalently via `Pkg` do
```julia
julia> using Pkg; Pkg.add("SmoQyKPMCore")
```

## Funding

The development of this package was supported by the National Science Foundation under Grant No. OAC-2410280.