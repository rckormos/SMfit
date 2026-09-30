# Progress kinetics

Fit enzyme progress curves directly from an `xarray.Dataset`, NetCDF file, or
Zarr store. Within each enzyme/condition group the tool estimates shared
**Km (mM)**, **kcat (s⁻¹)** and a positive **optical coefficient**. Each well uses
its own known final enzyme concentration. Increasing and decreasing signals,
multiple plates, arbitrary well labels, missing readings, and saturation masks
are supported.

This is the Schnell–Mendoza MVP specified in [the implementation handoff](docs/IMPLEMENTATION_SPEC.md).
It assumes stable enzyme, irreversible single-substrate Michaelis–Menten
kinetics, and an instantaneous unlimited reporter. Fit quality and optimizer
convergence do not establish that those assumptions hold, or that parameters
are identifiable. The numerical log-substrate solver was adapted from the
supplied donor; see [attribution](NOTICE.md) and [migration map](docs/MIGRATION_MAP.md).

## Install and run a complete example

Use Python 3.10 or newer (Python 3.12 is the validated reference), with a CPU
PyTorch installation. The repository has no dependency on Excel, proprietary
reader importers, or collaborator sample files.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

# Create two plates with two synthetic enzymes and both signal directions.
progress-fit example examples/synthetic.nc

# Inspect schema, group sizes, S0/E ranges, direction evidence and assumptions.
progress-fit validate examples/synthetic.nc --report-json examples/validation.json

# Fit and write tables, labeled predictions, provenance and PNG plots.
progress-fit examples/synthetic.nc --outdir results/example

# Fix a known optical coefficient rather than estimate it.
# 6000 is only a synthetic demonstration; it is not a reporter recommendation.
progress-fit examples/synthetic.nc --extinction-coefficient-M-cm 6000 \
  --outdir results/fixed

# Zarr input and output are also supported.
progress-fit example examples/synthetic.zarr
progress-fit examples/synthetic.zarr --output-format zarr --outdir results/zarr

python -m pytest
ruff check src tests examples
```

The fixed-6000 example deliberately fixes one value for both synthetic enzymes;
one has a different true value, so its recovered kinetics/diagnostics will
reflect miscalibration. To calibrate only one, add `--select PLASMIDID=SYNTH_A`.
`python -m progress_kinetics` is equivalent to `progress-fit`.
`python examples/generate_synthetic.py` provides the same generator as a script.
No private measurements or identifiers are distributed.
For the exact validated Python 3.12 dependency combination, install with
`python -m pip install -c constraints-py312.txt -e '.[dev]'`.

## Python and notebook use

The package has a public importable API: `FitConfig`, `fit_dataset`,
`validate_dataset`, `FitResults`, and `SchemaError`. You can pass an existing
`xarray.Dataset` directly to the fitter from a Jupyter notebook; no CLI invocation
or intermediate file is required.

Install into the notebook's active kernel with this cell when the notebook's
working directory is the repository root:

```python
%pip install -e .
```

If the notebook is elsewhere, replace `.` with the checkout's absolute path.
`%pip` installs into the current kernel's environment, which may differ from the
terminal environment. Restart the kernel if an already imported dependency was
updated during installation.

```python
import xarray as xr
from progress_kinetics import FitConfig, fit_dataset, validate_dataset
from progress_kinetics.synthetic import generate_example

ds = generate_example()
config = FitConfig(
    reference_path_length_cm=1.0,  # confirm against the reader normalization
    reporter_stoichiometry=1.0,
    multistart=3,
    seed=42,
)
summary = validate_dataset(ds, config)  # no optimization
results = fit_dataset(ds, config)       # input ds is not mutated
print(results.parameters)
print(results.diagnostics)
results.write("results/notebook", plots=False)  # export without changing the plotting backend

# Results retain original coordinates and unrelated input variables.
predicted = results.predictions["prediction_AU"]
mask = results.predictions["used_for_fit"]
```

For an existing NetCDF file, load it independently of the mathematical API:

```python
with xr.open_dataset("examples/synthetic.nc") as source:
    experiment = source.load()
results = fit_dataset(experiment, config)
```

`FitResults` contains labeled `parameters` (group dimension), `predictions`
(original plate/time/well dimensions), and `diagnostics` (plate/well dimensions),
plus `parameter_rows`, `trace_rows`, and JSON-compatible `run` metadata.
`write(outdir, format="netcdf"|"zarr", plots=False)` controls export.
`FitConfig.to_json(path)` and `FitConfig.from_json(path)` serialize settings.

To inspect a fitted curve inline in Jupyter, use the labeled result arrays:

```python
%matplotlib inline
import matplotlib.pyplot as plt

# Choose one trace from the synthetic example above.
curve = results.predictions.isel(plate_read=0, well_384=6)
curve["observed_AU"].plot.line(x="time", marker=".", linestyle="none", label="observed")
curve["prediction_AU"].plot.line(x="time", label="fitted")
plt.ylabel("Absorbance (AU)")
plt.legend()
plt.show()
```

Use `results.parameters.to_dataframe()` and `results.diagnostics.to_dataframe()`
for familiar notebook tables. `results.run` contains group statuses, assumptions,
warnings, optimizer history, and identification evidence for inspection. Disk
export is optional; call `results.write(...)` only when you want saved artifacts.

## Prepare an input dataset

Default mappings are:

| Meaning | Variable/dimension | Units and dimensions |
|---|---|---|
| Plate identity | `plate_read` | unique coordinate |
| Acquisition sample | `time` | numeric seconds; unique coordinate |
| Well/trace identity | `well_384` | any number of unique labels; no plate-layout assumption |
| Absorbance | `absorbance_corrected` | AU; plate, time, well in any order |
| Initial substrate S0 | `cofactor_concentration_mM` | mM; scalar, plate, well, or plate/well |
| Final enzyme E | `expected_conc_um_NORM1` | uM; scalar, plate, well, or plate/well |
| Enzyme construct | `PLASMIDID` | strings; broadcastable plate/well metadata |
| Conditions | `cofactor_name`, `mastermix_id`, `wavelength_nm`, `NaPO3_concentration_mM` | optional grouping metadata |
| Saturation evidence | `saturated` | optional boolean; same dimensions as signal |
| Saturation onset | `saturation_time_s` | optional per trace; seconds after first acquisition |
| Measured optical length | `pathlength` | positive cm; required in raw AU mode unless overridden |

A minimal dataset can be built as follows:

```python
import numpy as np
import xarray as xr

# Replace these arrays with your joined kinetics and metadata.
# Arrays must refer to the SAME labeled plate/well coordinates.
times = np.arange(0, 601, 10, dtype=float)
wells = ["A1", "A2", "A3", "A4", "A5", "A6"]
signal_AU = np.full((1, len(times), len(wells)), np.nan)

ds = xr.Dataset(
    {
        "absorbance_corrected": (("plate_read", "time", "well_384"), signal_AU),
        "cofactor_concentration_mM": ("well_384", [.05, .1, .3, .7, 1.5, 3]),
        "expected_conc_um_NORM1": ("well_384", [.3, .4, .5, .6, .4, .5]),
        "PLASMIDID": "MY_ENZYME",
        "mastermix_id": "one_buffer",
        "cofactor_name": "one_substrate",
    },
    coords={"plate_read": ["my_plate"], "time": times, "well_384": wells},
)
ds.time.attrs["units"] = "s"
ds.absorbance_corrected.attrs["units"] = "AU"
ds.cofactor_concentration_mM.attrs["units"] = "mM"
ds.expected_conc_um_NORM1.attrs["units"] = "uM"
# Fill signal_AU with observations before fitting this template.
```

Named dimensions are explicitly aligned/transposed. Reversed metadata dimensions
and shuffled wells are supported. Missing/duplicated coordinates and incompatible
dimensions raise `SchemaError`. Missing/nonfinite concentrations yield
trace-specific QC exclusions; negative concentrations are input errors. Zero S0
or zero E remains in QC as a control and does not influence kinetics or direction.
By default a fitting trace needs five distinct usable readings, with a configurable
minimum of three. Do not substitute source-plate metadata coordinates for the
actual observed wells.

Known time units `ms`, `min` and `s` convert to seconds. Missing time units assume
seconds with a warning. Known concentration units `M`, `mM` and `uM` convert
explicitly. Default unit-encoding substrate/enzyme names and corrected signal
mapping supply recorded fallback units; custom mappings require units. Unknown
units always require an explicit override. `EXT_COEFF` is ignored; it may describe
protein quantitation rather than the reporter. Enzyme concentration is already
final in-well uM: no transfer-volume/dilution correction is applied.

```sh
progress-fit experiment.nc --plate-dim plate --well-dim trace --time-dim sample \
  --signal-var absorbance --time-var elapsed --substrate-var S0 --enzyme-var E \
  --identity-var construct --units absorbance=AU --units elapsed=min \
  --units S0=mM --units E=uM --outdir results/mapped
```

For plate-specific acquisition times, supply a numeric `elapsed_s(plate,time)`
variable and `--time-var elapsed_s`. The sample coordinate still identifies original
readings. Nonmonotonic acquisition times are rejected unless `--sort-time` is
explicitly selected; processing then uses chronological order and keeps original
output indices. `--duplicate-time mean` averages valid observations at each
identical time and uses one aggregated point, with a duplicate mask/audit warning.
Otherwise duplicate times are errors. `observed_AU` holds the aggregated value
at the retained index; the original input signal remains in the result dataset.

## Grouping and selection

Default groups are `(plate_read, PLASMIDID, cofactor_name, mastermix_id)` plus
available wavelength and buffer metadata. Substrate/enzyme concentrations and
well IDs do not split groups. Different plates' A1 wells retain separate identities.
`sampletype` is not interpreted automatically.

```sh
# Merge equivalent enzyme/condition replicates across plates explicitly.
progress-fit experiment.nc --merge-plates --outdir results/merged

# Declare which metadata define matching conditions (enzyme identity is retained).
# Useful when mastermix names encode substrate dilution rather than buffer.
progress-fit experiment.nc --group-by PLASMIDID,cofactor_name,NaPO3_concentration_mM \
  --merge-plates --outdir results/conditions

# Supply an explicit group variable; declared conditions and enzyme must be homogeneous.
progress-fit experiment.nc --group-var fit_group --merge-plates --outdir results/groups

# Select only specified metadata values; repeat --select to intersect filters.
progress-fit experiment.nc --select PLASMIDID=ENZYME_A,ENZYME_B \
  --select normalization_pass=True --outdir results/selected

# Explicitly map assay roles instead of interpreting arbitrary sample strings.
progress-fit experiment.nc --sample-role blank=control --sample-role failed=exclude \
  --outdir results/roles
```

An existing `fit_group` variable is preferred automatically. `--group-var` selects
another name. Explicit groups are partitioned by plate unless `--merge-plates`
is supplied. Group homogeneity checks use enzyme identity and the declared
conditions (`group_by`, or the available default conditions). Missing identity
requires `--group-var` or `--single-enzyme`; the latter is an explicit assertion.
Many single-substrate groups trigger a warning; they are never merged automatically.
Time-varying `temperature_C` is recorded and changes exceeding 2°C are flagged.
To group by temperature, provide an explicit scalar or plate/well representative
variable; time-resolved metadata cannot be a group key. Selection can filter any
scalar or plate/well normalization/QC annotation without changing units or active
fraction assumptions.

## Saturation, missing readings and masks

The first trimmed string consisting solely of at least two question marks
(e.g. ` ???? `), a true `saturated` element, or a supplied `saturation_time_s`
excludes that reading and **all later readings in that trace**. Later numeric
readings do not restore eligibility. `saturation_time_s` means seconds since the
plate's first acquisition, regardless of stored acquisition units or reaction age.

Ordinary NaN/Inf readings are excluded pointwise; later finite readings remain.
If an importer replaced saturation markers with NaN and discarded the mask,
saturation meaning cannot be recovered reliably. Preserve a mask during import.
`--trim-at-first-missing` is an explicit conservative fallback, disabled by default.

Masks distinguish `saturation_trimmed`, `missing`, `selected_out`,
`time_window_excluded`, `control`, `insufficient_data`, `invalid_metadata`, and
`duplicate_aggregated`. `used_for_fit` is false for skipped/failed groups.
Predictions after saturation can be plotted but are marked out of fit. Excluded
observations/residuals are never filled with zero in exported results.

```sh
progress-fit experiment.nc --saturation-var clipped --saturation-time-var clip_onset \
  --time-window 30 900 --minimum-points 8 --outdir results/windowed
```

## Optical scale, direction, baseline and timing

The model is

```text
E_mM = E_uM / 1000
Vmax_i = kcat * E_i_mM
S_i + Km*log(S_i) = S0_i + Km*log(S0_i) - Vmax_i*kinetic_time
P_i = S0_i - S_i
alpha_i = epsilon_per_M_cm * effective_length_cm * stoichiometry / 1000
A_i = baseline_i + direction * alpha_i * P_i + drift * acquisition_time
```

`epsilon` is the positive magnitude of the effective reaction extinction change
in M⁻¹ cm⁻¹. It need not equal the extinction coefficient of an isolated molecule.
`alpha` is the directly observed scale in AU/mM and is always exported. With
varying raw pathlengths, group CSV reports mean/min/max alpha; per-trace alpha is
in QC. Stoichiometry is fixed and positive, default 1 reporter equivalent per
substrate, recorded as an assumption. Free stoichiometry and free epsilon cannot
be recovered separately.

For corrected AU, use the **normalized reference length**, never reapply measured
per-well pathlength. Precedence is CLI/config `reference_path_length_cm`, then
signal/dataset attribute of that name, then the explicit 1 cm fallback. The
fallback yields `assumed_1cm_reference` and
`epsilon_is_reference_conditional=true`. Supply a reference only after verifying
the reader convention; a physical epsilon claim depends on that external confirmation.

```sh
progress-fit experiment.nc --reference-path-length-cm 1 \
  --reporter-stoichiometry 2 --outdir results/corrected
progress-fit raw.nc --signal-mode raw --signal-var raw_AU --units raw_AU=AU \
  --pathlength-var pathlength --outdir results/raw
progress-fit raw.nc --signal-mode raw --fixed-path-length-cm 0.4 --outdir results/raw_fixed
progress-fit experiment.nc --fix-alpha-AU-per-mM 6 --outdir results/alpha_fixed
# No raw pathlength calibration available: estimate an effective scale only.
progress-fit raw.nc --signal-mode raw --effective-alpha-only --outdir results/effective
```

Fixed epsilon and fixed alpha are mutually exclusive. Fixed alpha in raw mode
requires `--effective-alpha-only`: it applies to the chosen signal scale without
pretending to correct varying pathlengths. Effective-alpha-only fits have no
physical epsilon estimate. The generic epsilon start of 6000 is an optimizer
starting point, not an assay constant; effective-alpha-only uses 6 AU/mM by default.

Direction defaults to one robust vote per positive-S0/positive-E kinetic trace.
Votes use a median pairwise slope and median early/late window difference. A
trace must exceed both 0.002 AU and three times a detrended-difference MAD noise
estimate. The default needs at least three informative traces and 75% agreement;
flat/noisy traces do not vote. Ambiguous groups are reported and skipped while
other groups continue. Thresholds and every vote are saved. One/two-trace groups
need an explicit direction or a deliberately relaxed minimum-evidence setting.
`--signal-direction increasing|decreasing` records an override. Signals are
never individually flipped to force one group to agree.

One baseline per trace is profiled as the mean observed-minus-model signal over
used points. It is not fixed at the first reading. Equal within-trace weights
make this the weighted least-squares mean. Shared group drift is off by default;
`--fit-drift` enables it and adds it to sensitivity diagnostics, including
parameter sensitivity column cosines to expose near-parallel responses. Late
drift after depletion may create conflicting sign votes; use a known direction
override when the assay warrants it.

Acquisition time zero is the plate's first acquisition, including reads absent
from an individual well. Default reaction start at that point is a recorded
assumption. Missing early reads or time-window selection never reset the reaction.

```text
reaction_time = acquisition_time + known_reaction_age_at_first_read
kinetic_time = max(reaction_time - lag_s, 0)
```

```sh
progress-fit experiment.nc --reaction-age-at-first-read-s 25 --outdir results/aged
progress-fit experiment.nc --lag-s 10 --outdir results/lagged
progress-fit experiment.nc --fit-lag --bound lag_s 0 60 --outdir results/lag_fit
```

Reaction age can be fixed globally or per plate in a config file; it is never
fitted. Fitted lag is shared per group and opt-in. Age and lag are confounded,
so only fit lag with externally constrained age. kcat is apparent on the supplied
enzyme basis: enzyme concentration uncertainty and active fraction propagate into
its interpretation. Set `--enzyme-basis molecules` or `active_sites` to record
which concentration you supplied; the tool does not infer that distinction.

## Optimization and identifiability

Float64 CPU is the reference. CUDA is explicit and optional; MPS float64 is
rejected. Float32 is available explicitly, with reduced accuracy near depletion;
CPU/CUDA agreement is tested when CUDA exists (relative tolerance 1e-8 for
float64 forward predictions). No accelerator is chosen automatically.

Positive kinetic/optical parameters use bounded log-space sigmoid transforms;
lag and drift use bounded linear transforms. Seeded multistart Adam → L-BFGS
restores the best finite evaluated state, including after nonfinite failures.
Saved loss, parameter values and gradient refer to the same evaluation.
Convergence flags describe gradient convergence; budget/change-tolerance stops
are separate. A successful group means a finite fit was produced, not that it
converged or that the science is established.

Default objective is equal-trace mean squared residual normalized by each
trace's dynamic range, floored at 0.01 AU. Control traces are excluded. QC reports
scales and trace/point weights. `--weighting raw_mse` uses equal-trace raw MSE.

```sh
progress-fit experiment.nc --multistart 5 --seed 13 --adam-steps 300 --lbfgs-steps 200 \
  --start Km_mM=0.2 --start kcat_per_s=15 --start optical=5000 \
  --bound Km_mM 0.001 10 --bound kcat_per_s 0.01 1000 \
  --outdir results/custom

# Optional fixed kinetic parameters.
progress-fit experiment.nc --fix-parameter Km_mM=0.3 --outdir results/fixed_Km

# Reoptimize kinetic parameters over fixed optical values around the optimum.
progress-fit experiment.nc --identifiability profile --profile-factors 0.5 0.75 1 1.5 2 \
  --outdir results/profile
```

Diagnostics include profiled parameter sensitivity singular values/rank/condition,
bound hits, competing multistart parameters/losses, S0/E coverage, predicted
depletion, per-trace residual RMSE/autocorrelation, and optional optical profiles.
Weak identification is flagged by sensitivity condition/rank, low median depletion
(<10%), narrow substrate range (<3-fold), near-equivalent starts with >2-fold
parameter differences, bound hits, or a flat reoptimized profile. These are
transparent heuristic warnings with raw evidence, not calibrated probabilities.

With little depletion, slopes primarily constrain optical scale × kcat. With all
S0 far below Km, kcat/Km is also confounded. Broad S0 coverage and substantial
depletion help but cannot guarantee identification. Fix an independently known
optical coefficient when free-scale estimates are weak. Profiles/Hessians are
not confidence intervals; no intervals or precision claims are manufactured.
Inspect the curves and residuals for model mismatch even when sensitivity is good.

## JSON configuration

CLI arguments override config fields only when explicitly supplied; all other
fields persist. Repeated `--start`, `--bound`, `--select` and `--units` merge with
existing dictionaries. Generate a complete editable template:

```python
from progress_kinetics import FitConfig
FitConfig().to_json("fit_config.json")
```

A partial config example:

```json
{
  "merge_plates": true,
  "group_by": ["PLASMIDID", "cofactor_name", "NaPO3_concentration_mM"],
  "reference_path_length_cm": 1.0,
  "reaction_age_at_first_read_s": {"plate_1": 12.0, "plate_2": 18.0},
  "sample_roles": {"blank": "control", "bad": "exclude"},
  "selections": {"normalization_pass": [true]},
  "multistart": 4,
  "seed": 123
}
```

```sh
progress-fit experiment.nc --config fit_config.json --outdir results/configured
progress-fit --help
```

Per-plate age mappings must include every plate. When overriding `bounds` in
JSON, supply the complete five-parameter dictionary from the template; CLI
`--bound` can replace individual entries. `starts` must include all five named
parameters; CLI `--start` replaces one. Configuration rejects unknown keys,
nonfinite values, invalid units/policies and conflicting optical settings.

## Outputs and exit codes

| File | Contents |
|---|---|
| `fit_parameters.csv` | All groups including failures; kinetics, optical scale, direction, assumptions, fixed flags, loss, convergence and identification |
| `trace_diagnostics.csv` | Every plate/well identity, selection/trim reason, masks/counts, S0/E/Vmax, baseline, weights, residual/depletion evidence |
| `fit_results.nc` or `fit_results.zarr` | Original coordinates/input variables plus AU predictions/residuals, substrate/product mM, masks, group parameters and trace diagnostics |
| `run.json` | Config, versions, seed, input provenance, conventions, warnings, statuses, full optimization history, multistart and profile summaries |
| `config.json` | Reusable settings |
| `plots/` | Paginated per-trace progress/residual plots with exclusions, optimizer histories, optional profiles and parameter summary |

Opaque object metadata is converted to stable strings for NetCDF/Zarr. No pickle
is accepted or written. Result exports include supplied input data/provenance;
handle your own results according to your data policies. Nothing is uploaded.
Existing output files in the requested output directory are replaced on a repeat
run; choose a new directory for each experiment/configuration if retaining history.

CLI prints machine-readable JSON:

| Exit | Meaning |
|---|---|
| 0 | All requested groups produced finite fits (warnings allowed); validation has usable traces |
| 2 | Invalid input/schema/configuration or I/O error; JSON includes the reason |
| 3 | Partial success: some groups failed, or some non-control requested traces were unusable |
| 4 | No valid fit; validation found no usable kinetic traces |

Selection-only/control-only groups remain visible with `no_kinetic_traces`; if
other groups fit, this results in partial success. Auto-direction ambiguity is
always visible. `--report-json path` saves validation/error/fit summaries.
`--validate-only` and `progress-fit validate` inspect data without optimizing;
validation reports direction ambiguity but does not require direction success
for exit 0.

## Development, validation and scope

```sh
python -m pytest -q
ruff check src tests examples
ruff format --check src tests examples
python -m compileall -q src
```

[Validation notes](docs/VALIDATION.md) describe the tested runtime, independent
ODE recovery tests, acceptance coverage, donor baseline and practical limits.
The synthetic generator deliberately includes reversed metadata dimensions,
repeated well names across plates, variable E, both signs, isolated missing
points, and a saturation mask followed by invalid finite readings. Tests cover
unit conversions, grouping, timing, identifiability, CLI/API parity and exports.
The original ZIP/extraction is ignored by Git and excluded from package builds.

Optional donor concepts (coupled first-order reporter, capacity/impurity models,
initial-rate comparison, automated trace pruning/window scans) remain in the
supplied local donor reference. They are **not enabled or exposed by this MVP**.
Adding them requires the independent scientific validation described in the spec.
This project is licensed under the [MIT License](LICENSE). Source attribution
is preserved in [NOTICE.md](NOTICE.md). No push, hosting, or publication is part
of this implementation.
