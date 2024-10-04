# API

Note that for many methods there are two verions, one that relies on taking
an instance of the [`KPMExpansion`](@ref) type as an argument, and a lower
level one that does not.

- [`KPMExpansion`](@ref)
- [`kpm_update!`](@ref)
- [`kpm_update_bounds!`](@ref)
- [`kpm_update_order!`](@ref)
- [`kpm_coefs`](@ref)
- [`kpm_coefs!`](@ref)
- [`kpm_moments`](@ref)
- [`kpm_moments!`](@ref)
- [`kpm_density`](@ref)
- [`kpm_density!`](@ref)
- [`kpm_dot`](@ref)
- [`kpm_mul`](@ref)
- [`kpm_mul!`](@ref)
- [`kpm_lmul!`](@ref)
- [`kpm_eval`](@ref)
- [`kpm_eval!`](@ref)
- [`apply_jackson_kernel`](@ref)
- [`apply_jackson_kernel!`](@ref)
- [`lanczos`](@ref)
- [`lanczos!`](@ref)

```@docs
KPMExpansion
KPMExpansion(::Function, ::Any, ::Int, ::Int)
kpm_update!
kpm_update_bounds!
kpm_update_order!
kpm_coefs
kpm_coefs!
kpm_moments
kpm_moments!
kpm_density
kpm_density!
kpm_dot
kpm_mul
kpm_mul!
kpm_lmul!
kpm_eval
kpm_eval!
apply_jackson_kernel
apply_jackson_kernel!
lanczos
lanczos!
```