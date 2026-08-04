# NOTREKS utilities

This package contains method-independent NOTREKS pair validation and
value/gradient kernels. It is not a causal-discovery method and it does not
depend on DAGMA.

Continuous methods may optionally add `NoTreksPenalty` through their existing
structural-penalty interface. With no component, vanilla optimizer behaviour is
unchanged.

The default kernel name is `fast`, an alias for the selected-column inverse
kernel. It exactly matches the dense inverse value and gradient while avoiding
materializing unnecessary inverse columns when the constrained pair set is
sparse. Reproducibility and ablation names remain available:

- `notreks_reference` / `dense_current`
- `dense_inv`
- `selected_inv` (the implementation behind `fast`)
- `poly_selected_walk`
- `poly_selected_exp`

The fast kernel implements `function="inv"`. Choose an appropriate named
alternative explicitly for other matrix functions.
