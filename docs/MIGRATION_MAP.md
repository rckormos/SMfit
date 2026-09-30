# Donor-to-public migration map

All paths below are relative to `donor/src/asn_kinetics/`. This is an intact
working-tree snapshot, not an already-generic backend.

| Donor module | Reuse | Required adaptation |
| --- | --- | --- |
| `schnell_mendoza.py` | Differentiable log-substrate Newton solve, product conversion | Keep generic math; test broadcasting and zero-substrate/zero-enzyme bypasses in wrapper. |
| `model.py` | Bounded parameters, profiled baselines, fitted/fixed optical scale; optional first-order reporter convolution and capacity | Replace shared Vmax with shared kcat and per-trace E; add signal sign and effective pathlength; separate timing conventions; rename impurity pools generically. |
| `objective.py` | Tensor traces and equal-trace, dynamic-range-normalized residuals | Add compound trace IDs, enzyme and optical metadata, original indices/masks; validate finite inputs and use selected data for scales. |
| `optimize.py` | Adam/L-BFGS orchestration, finite checks and best-state restoration | Generalize physical parameter/history schema to kcat, epsilon, timing; multi-start and identifiability diagnostics. |
| `diagnostics.py` | Residual and exhaustion diagnostics | Remove single enzyme concentration post-hoc kcat conversion and the default NADH constant; add units, assumptions, and group/trace IDs. |
| `plotting.py` | Progress, residual, history and initial-rate plot patterns | Support either direction, many groups, per-trace E and pathlength; no assay-specific labels. |
| `io.py` | CSV/JSON writers and run metadata | Implement labeled xarray output; avoid plate-well collisions, serialize object metadata safely, avoid publishing private paths. |
| `initial_rate.py` | Early-time slope estimation and saturating regression | Convert slopes with sign and per-well optical scale; normalize rates by E before sharing kcat. Free epsilon is unidentifiable from early slopes alone. |
| `time_window.py` | Time-window comparison machinery | Respect global reaction origin; update model/history factories and optional parameters. Defer automated windows in MVP. |
| `trace_pruning.py` | Residual-loss knee and iterative warm starts | Opt-in only; preserve masks and all rejection reasons; do not silently erase bad fits. |
| `trace_selection.py` | Selection audit/reporting patterns | Replace spreadsheet-column assumptions with labeled xarray selection. |
| `data.py`, `config.py`, `plate_map.py` | Trace containers and legacy loader references | Write a new xarray adapter. Do not reuse XLSX layout, dilution inference, independent time reset, or nonnumeric-value skipping for saturation. |
| `simulation.py` | Synthetic first-order coupled reporter generator | Rename pools and enzyme, support both signs and per-well E, use purely synthetic public examples. |
| `ast_clearance.py`, `ast_second_order.py` | Optional reporter kinetic calibration/reference models | Not needed for the generic Schnell-Mendoza MVP; do not expose AST-specific names. |
| `cli.py`, `ast_cli.py`, `ast_second_order_cli.py`, `sim_cli.py` | Examples of argument parsing, output handling and existing workflows | Reference only. Build a new public CLI; do not copy ASN-specific defaults/help. |
| `__init__.py` | Version/export pattern | Replace package name and workbook-centric public exports. |

## High-risk semantic changes

- Donor parameters have one `vmax_mM_s` for all traces. Merely dividing that
  result by one enzyme concentration is wrong for varying per-well enzyme.
  Change prediction to `vmax_i = kcat * enzyme_i_uM / 1000` before optimization.
- Donor optical scale `alpha_AU_mM` is positive and signal is hard-coded
  negative. The public model needs `direction * alpha_i * reporter_product`.
- Do not conflate molar epsilon with per-millimolar epsilon. Legacy diagnostic
  epsilon is per mM per cm; the proposed CLI fixed epsilon is per M per cm.
- The current coupled solver initializes the intermediate pool at the first
  element of the supplied time vector. If the first retained observation is
  not at reaction time zero, propagate from time zero before taking observations.
- The donor fixed quadrature is a useful approximation, not an exact solution
  for arbitrarily large sampling intervals or clearance rates. Validate and
  subdivide as needed in an optional generic coupled implementation.
- A saturated instrument is not finite reporter capacity. Instrument saturation
  requires truncation; a capacity-limited reporter needs an explicit model.
- Donor instantaneous initial impurity contributes a constant signal that can
  be absorbed by the free baseline. Do not promise impurity identification
  without time-dependent evidence or an external baseline constraint.
- Source tests assert legacy behavior. Keep them as a reference baseline;
  change/add public tests deliberately rather than mechanically renaming every
  assertion and preserving inappropriate defaults.

## Dependency boundary

The runnable donor uses Python >=3.10, NumPy, PyTorch, matplotlib and openpyxl.
The public core should use NumPy, PyTorch and xarray, matplotlib for plots, and
NetCDF/Zarr backends appropriate to its documented file formats. openpyxl is
needed only for legacy workbook references/tests, not for an xarray-only public
interface. Record tested versions and choose bounds based on actual testing;
do not assume the donor's unbounded minimum requirements imply compatibility
with every future version.
