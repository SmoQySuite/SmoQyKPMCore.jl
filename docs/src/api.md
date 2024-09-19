# API

Note that for many methods there are two verions, one that relies on taking
an instance of the [`KPMExpansion`](@ref) type as an argument, and a lower
level one that does not.

- [`KPMExpansion`](@ref)
- [`update_kpmexpansion!`](@ref)
- [`update_kpmexpansion_bounds!`](@ref)
- [`update_kpmexpansion_order!`](@ref)
- [`kpm_coefs`](@ref)
- [`kpm_coefs!`](@ref)
- [`kpm_mul`](@ref)
- [`kpm_mul!`](@ref)
- [`kpm_eval`](@ref)
- [`kpm_eval!`](@ref)
- [`apply_jackson_kernel!`](@ref)
- [`apply_jackson_kernel`](@ref)

```@docs
KPMExpansion
KPMExpansion(::Function, ::Any, ::Int, ::Int)
update_kpmexpansion!
update_kpmexpansion_bounds!
update_kpmexpansion_order!
kpm_coefs
kpm_coefs!
kpm_mul
kpm_mul!
kpm_eval
kpm_eval!
apply_jackson_kernel!
apply_jackson_kernel
```