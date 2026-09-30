# Validation record

Local validation performed on 2026-09-30, using Python 3.12.13 on macOS,
float64 CPU. Versions used:

| Component | Version |
|---|---|
| progress-kinetics | 0.1.0 |
| NumPy | 2.5.3 |
| PyTorch | 2.10.0 |
| xarray | 2026.9.0 |
| SciPy | 1.17.1 |
| matplotlib | 3.10.9 |
| h5netcdf | 1.8.1 |
| h5py | 3.16.0 |
| Zarr | 3.4.0 |
| pytest | 9.1.1 |

`constraints-py312.txt` records this tested dependency combination. The package
supports Python >=3.10 syntax and declares dependency compatibility ranges;
the local validation establishes this Python 3.12 combination, not every
possible version combination. CI exercises Python 3.12. CUDA is optional and was
unavailable locally; the CPU/CUDA float64 parity test was skipped explicitly.

## Checks performed

- Original bundle SHA256 integrity: all 44 listed files matched.
- Bundled donor regression baseline: **84 passed** in 22.58 seconds. This validates
  the legacy donor snapshot only, not the new API.
- Public acceptance/regression suite: **75 passed, 1 skipped** in 16.10 seconds.
  The skip is CUDA parity on a machine without CUDA.
- Ruff lint and formatting, Python byte compilation, editable installation, CLI
  invocation and wheel/source distribution builds were verified.
- Package archives contain the generic implementation and provenance notice;
  the original donor ZIP, legacy package, local data and run artifacts are excluded.
- A complete CLI run produced all four groups, CSV/JSON/NetCDF and 13 PNG plots.
  Progress/residual plot layout was visually inspected.

## Acceptance coverage

| Spec area | Evidence |
|---|---|
| 1. Parameter recovery | `test_fit.py`: favorable curves from independent DOP853 ODE integration; <=1% fixed optical, <=2% fitted optical targets |
| 2. Per-well E/units | `test_model.py`, `test_data.py`: exact uM/mM conversion, wrong constant-E prediction comparison, independent epsilon/length/stoichiometry conversion |
| 3. Direction | `test_fit.py`: both signs, equal trace votes, majority/conflict, flat/noisy controls, low evidence and override provenance |
| 4. Saturation | `test_data.py`, `test_io_cli.py`: question marks, boolean masks, onset times, numeric readings after saturation, isolated missing values, all-saturated and insufficient data |
| 5. Alignment/schema | `test_data.py`: reversed dimensions, shuffled wells, scalars, custom dimensions, duplicate/missing coordinates, unit overrides, incompatible dimensions and metadata made missing by labeled alignment |
| 6. Groups/identity | `test_data.py`, `test_fit.py`: enzymes/conditions, repeated well labels, explicit merge/group homogeneity, no concentration-derived grouping |
| 7. Optical conventions | `test_data.py`, `test_fit.py`: corrected/reference vs measured length, conditional fallback, raw varying length, fixed epsilon/alpha, mutual exclusion, stoichiometry |
| 8. Timing | `test_data.py`, `test_fit.py`: known age, missing early reads, plate-specific times, fixed and fitted lag, no per-trace reset |
| 9. Identification/optimization | `test_fit.py`: low depletion/narrow range flags, favorable identification, bounded parameters, restored finite state, history/loss consistency, reoptimized profiles, drift and fixed kinetics |
| 10. Interfaces/outputs | `test_io_cli.py`: API/CLI parity, NetCDF/Zarr round trips, safe metadata, nonmutation, seeded reproducibility, statuses 0/2/3/4, configuration overrides and plots |

Key recovery tests use an independently integrated ODE, not the fitted Newton
solver as their sole truth source. The solver also passes the integrated equation
and finite-gradient checks ported from the donor.

## End-to-end synthetic run

Command: `progress-fit examples/synthetic.nc --outdir results/example`.
The example includes 2 plates, 2 constructs, a 0.05–3.0 mM dilution series,
0.25–0.8 uM enzyme, both signs, zero-concentration controls, 0.0005 AU noise,
missing reads and saturation followed by deliberately invalid finite values.

| Group | True Km / fitted Km (mM) | True kcat / fitted kcat (s^-1) | True epsilon / fitted epsilon (M^-1 cm^-1) |
|---|---|---|---|
| plate_1 / SYNTH_A | 0.3 / 0.29982 | 12 / 11.9983 | 6000 / 5999.91 |
| plate_1 / SYNTH_B | 0.6 / 0.60048 | 20 / 20.0068 | 4200 / 4200.40 |
| plate_2 / SYNTH_A | 0.3 / 0.29979 | 12 / 11.9981 | 6000 / 6000.30 |
| plate_2 / SYNTH_B | 0.6 / 0.60040 | 20 / 20.0048 | 4200 / 4200.00 |

All three parameters recover within 0.1% in this favorable synthetic run. These
numbers do not imply the same performance for real assays. Two starts' selected
fits terminated without meeting the strict gradient criterion, despite accurate
synthetic recovery; optimizer completion and convergence remain separate output
fields. Broad substrate coverage/depletion gave sensitivity conditions around
6.5–7.2. Reported precision is numerical evidence here, not experimental uncertainty.

## Interpretation and remaining release decisions

No experimental validation, calibrated intervals, active-fraction estimation,
coupled-reporter validation, or GPU performance claim is made. Drift can make
post-depletion sign evidence conflict, requiring a direction override when fitting
drift. Optical profiles and sensitivity diagnostics are evidence/heuristics, not
confidence intervals. Pathlength reference, enzyme basis, reporter stoichiometry
and known reaction age need external assay confirmation.

Optional coupled reporter, capacity/impurity, initial-rate comparison and automated
pruning/window scanning are deferred according to the spec. Their donor sources
remain in the supplied local bundle rather than the public package. The owner
selected the MIT License; see `LICENSE` and the source attribution in `NOTICE.md`.
No public hosting, push or release was performed.
