# GRTresnaIDX

`GRTresnaIDX` is a CarpetX-native initial data reader thorn that loads GRTresna
HDF5 output at `t=0` and populates `ADMBaseX` spacetime fields.

## Supported input (current implementation)

- HDF5 file with root metadata attributes:
  - `num_levels`
  - `num_components`
  - `component_0`, `component_1`, ...
- AMR hierarchy groups: `level_0`, `level_1`, ..., `level_N`
- Multiple boxes per level via `boxes` dataset
- Source data datasets under each level:
  - `boxes`
  - `data:offsets=0` (or `data:offsets`)
  - `data:datatype=0` (or `data:datatype=1` / `data`)

The reader samples from the finest level that contains each query point and
falls back to coarser levels when needed.

## Variable basis handling

The reader autodetects one of:

- ADM basis: `gxx..gzz`, `kxx..kzz`, `alp|lapse`, `betax..betaz|shift1..3`
- BSSN-like basis: `chi`, `h11..h33`, `K`, `A11..A33`, `lapse`, `shift1..3`

BSSN-like fields are converted to ADM in-place before writing `ADMBaseX`.

## Coordinates

Source cell-center mapping follows:

- `x = (i + 0.5) * dx - center`

`center` is either:
- auto: inferred from `level_0` source index extents, or
- user-provided via `use_source_center=yes` and `source_center[3]`.

## Minimal parfile block

```ccl
ActiveThorns = " ... GRTresnaIDX ... "
GRTresnaIDX::grtresna_filename = "PATH/InitialDataFinal.3d.hdf5"
GRTresnaIDX::interpolation     = "trilinear"
```
