# Generic xarray progress-curve fitter: implementation specification

Status: implementation handoff, 2026-09-30. User-requested MVP is a public-facing,
enzyme-agnostic Schnell-Mendoza progress-curve tool. This bundle does not yet
implement that tool. `progress_kinetics` and `progress-fit` are proposed names.

## 1. Scope and confirmed assay contract

Implement global progress-curve fitting within enzyme/assay groups, with known
initial substrate and known, possibly different, enzyme concentrations in each
well. Estimate shared `Km_mM` and `kcat_per_s` plus an optical coefficient by
default. Do not fit one global Vmax and convert it with one average enzyme
concentration. Expose a reusable Python API and a CLI for serialized xarray.

Confirmed by the user and collaborator:

- `cofactor_concentration_mM` is the initial concentration of the substrate
  used for this assay. Despite the field name, treat it as substrate here.
- `expected_conc_um_NORM1` is the final enzyme concentration in the well, in
  micromolar; do not adjust again for transfer volumes or dilution factors.
- `absorbance_corrected` is still AU, scaled by the reader's pathlength
  measurement. It is not AU/mM, and the measured pathlength must not be applied
  again to an already normalized signal.
- One dataset can contain multiple plate reads with up to 96 unique enzymes
  per plate. `PLASMIDID` identifies enzyme constructs; condition metadata also
  matters. Do not fit all wells together by default.
- Instrument `????`-style values mean saturation. At the first saturated
  observation, exclude it and all later observations from that trace.
- Infer increasing/decreasing direction by default, but allow an override.
- Fit the optical/extinction coefficient by default, with a CLI option to
  supply a known fixed coefficient.

Unconfirmed conventions must be configurable and appear in output: reference
pathlength of the corrected signal, reporter stoichiometry, reaction start
relative to the first read, which metadata define equivalent assay conditions,
and whether supplied enzyme concentration counts molecules or active sites.
These do not require more sample data to implement the tool. They do constrain
interpretation of fitted values and must never be hidden assumptions.

MVP assumes stable enzyme, irreversible single-substrate Michaelis-Menten
kinetics, no product inhibition or substrate inhibition, and instantaneous,
unlimited reporter. Goodness of fit cannot prove those assumptions. A finite
reporter response is a separately requested/selected extension, not implicit
in an ordinary Schnell-Mendoza fit.

## 2. Input adapter and labeled data

Primary API: `fit_dataset(dataset: xarray.Dataset, config: FitConfig) -> FitResults`.
Keep file I/O and plotting outside the mathematical core. CLI accepts NetCDF
and Zarr; avoid untrusted pickle and automatic proprietary TXT/CSV heuristics.

Default dimensions are `plate_read`, `time`, `well_384`; allow dimension/variable
mapping overrides. The well dimension may describe any plate size or arbitrary
trace IDs; never hard-code 384 or a 96-to-384 layout. If metadata only refer to
96-well sources, they do not replace the 384-well observation coordinates.

| Meaning | Default variable | Units / dimensions |
| --- | --- | --- |
| Signal | `absorbance_corrected` | AU; plate, time, well |
| Time | `time` | seconds; time (support plate-specific time through explicit mapping) |
| Initial substrate | `cofactor_concentration_mM` | mM; plate, well, possibly stored in reverse order |
| Final enzyme | `expected_conc_um_NORM1` | uM; plate, well |
| Enzyme identity | `PLASMIDID` | strings; plate, well |
| Assay annotations | `mastermix_id`, `sampletype`, `cofactor_name` | strings; either plate/well order |
| Measured optical length | `pathlength` | cm for raw AU mode; plate, well |
| Optical wavelength | `wavelength_nm` | nm; scalar or explicitly mapped condition |
| Saturation evidence | optional `saturated` | bool; plate, time, well |

`temperature_C`, `NaPO3_concentration_mM`, worklist IDs, source files, source wells,
QC/normalization annotations and other concentration columns are provenance or
optional grouping/selection variables, not alternative substrate/enzyme inputs.
`EXT_COEFF` may describe a protein's concentration measurement rather than the
reporter: ignore it unless the user explicitly maps and gives its meaning/units.

Implementation requirements:

1. Use xarray named dimensions, alignment and explicit transposition. Reject
   missing or duplicated coordinates and incompatible dimensions; never flatten
   positional arrays that silently pair metadata with the wrong observations.
2. Preserve a unique trace identity `(plate_read, well)` and original sample
   indices. Two plates' A1 wells are distinct even if fitted in one group.
3. Allow scalar or plate-only metadata to broadcast by names. Reject missing
   required metadata for fitting wells, with trace-specific reasons. Unknown
   units require an explicit override, not silent conversion guesses.
4. Require finite nonnegative S0 and enzyme concentrations. E=0 or S0=0 are
   possible controls; default retain them in QC output but exclude from kinetic
   and direction estimation. Optional control-informed drift/baseline handling
   must be explicit. Negative concentrations are errors, not controls.
5. Time must be finite, ordered and have at least three distinct usable points
   for an MVP fitting trace (configurable higher minimum recommended). Reject
   duplicates by default; offer an explicit mean-aggregation policy. Reject
   nonmonotonic data or sort only as an explicit logged preprocessing operation.
6. Convert known ms/min units to seconds; missing time units default to seconds
   only with a recorded warning. Keep original coordinates in predictions.
7. Do not change input dataset in place; preserve unrelated variables.

### Saturation and missing values

If the input signal still contains strings, trim at the first whitespace-trimmed
value consisting only of at least two question marks, or at a true saturation
mask element. A numeric signal plus explicit `saturated` mask is preferred.
Also support per-trace `saturation_time_s` for already cleaned numeric datasets.
Retain the first cause/time and counts in output.

Ordinary isolated NaN/Inf values are excluded pointwise, not assumed to be
saturation. If all markers were replaced with NaN without a saturation mask,
the fitter cannot recover saturation semantics reliably. Document this and
warn that later finite readings will not automatically be discarded. An
explicit opt-in `--trim-at-first-missing` may provide a conservative fallback;
it must not be the default.

Masks in output distinguish saturation-trimmed, missing, selected-out, time-
window excluded, control, and insufficient-data cases. Do not fill excluded
observations with zero. Simulated predictions may be evaluated after saturation
for display, but must be marked out-of-fit.

## 3. Grouping, conditions and QC

Prefer an explicitly supplied `fit_group` variable, after checking that enzyme
identity and declared conditions are homogeneous within each group. Otherwise
default to separate groups by `(plate_read, PLASMIDID, cofactor_name,
mastermix_id)` plus available scalar/well-level wavelength and buffer-condition
metadata such as `NaPO3_concentration_mM`. Normalize object values to deterministic
strings and handle missing values explicitly, not as one accidental NaN group.

Substrate concentration, enzyme concentration, well ID, and replicate number
are NOT grouping keys. `sampletype` is not a reliable grouping key: it may
encode concentration, replicate, assay role, or controls. Offer sample-role
mapping and explicit selections; do not parse arbitrary strings as scientific
truth. By default fit concentration-valid non-control wells and report what
was selected. Where normalization flags exist, expose filtering and record
policy; do not infer units or active enzyme fraction from a pass flag.

Offer `--group-by` and `--group-var`, and an explicit `--merge-plates` option for
replicates with matching enzyme/conditions. No unrequested parameter sharing
between different enzyme constructs or assay conditions. Temperature traces
are recorded and significant changes flagged; they should not split every time
point into a new group. A representative temperature grouping policy can be
explicitly configured.

Because mastermix names may encode a substrate dilution, warn if default
grouping produces many single-concentration groups and suggest user-defined
grouping. Do not automatically merge those groups. A missing enzyme identifier
requires a user-supplied group or explicit single-enzyme assertion.

## 4. Mathematical model and units

For well i with known initial substrate `S0_i` and enzyme `E_i`:

```text
E_i_mM = E_i_uM / 1000
Vmax_i_mM_s = kcat_per_s * E_i_mM
dS_i/dt = -Vmax_i * S_i / (Km + S_i)
S_i(0) = S0_i
P_i(t) = S0_i - S_i(t)
S_i + Km*log(S_i) = S0_i + Km*log(S0_i) - Vmax_i*t
```

Reuse the donor differentiable log-substrate solver. Handle exact S0=0 or E=0
without sending them to a positive-only logarithm solver. Validate all parameter
and observation tensors. Float64 CPU is the reference; GPU is optional and must
give matching results within documented tolerance. Do not select unsupported
float64 MPS acceleration automatically.

Observation model, independent of reporter direction:

```text
A_i(t) = b_i + d * alpha_i * P_i(t) + drift_g*t
d = +1 for increasing, -1 for decreasing
alpha_i_AU_per_mM = epsilon_g_per_M_cm * l_eff_i_cm * stoichiometry / 1000
```

Here epsilon is the positive magnitude of the effective reaction extinction
change, not necessarily the extinction coefficient of one isolated molecule.
For one consumed absorbing reporter and transparent product it is the reporter
epsilon magnitude. Stoichiometry counts reporter equivalents per substrate.
Document that enzyme concentration uncertainty/active fraction propagates into
apparent kcat, which is on the supplied enzyme-concentration basis.

For corrected AU mode, `l_eff` is the reader's normalized reference length;
never the per-well measured length again. Proposed fallback: 1.0 cm if neither
metadata nor CLI supplies a reference, with an explicit `assumed_1cm_reference`
warning and `epsilon_is_reference_conditional=true`. This is an implementation
convention, not a collaborator-confirmed fact. Always output alpha (the directly
fitted effective scale) as well. Suppress claims of calibrated physical epsilon
unless the reference convention is confirmed. CLI/metadata reference length
override takes precedence and is recorded.

For raw AU mode use per-well measured positive lengths, or a supplied fixed
length. Missing raw pathlength is an error unless the user explicitly requests
effective-alpha-only fitting. Offer `--fix-alpha-AU-per-mM` as an alternative
calibration route; mutually exclusive with fixed epsilon. Fixed alpha applies
directly to the chosen signal scale, not to raw varying-pathlength epsilon.

Default stoichiometry is 1.0 and is recorded as an assumption. A user can
configure another positive value. Free epsilon and free stoichiometry cannot
be estimated separately; stoichiometry stays fixed. Reference length and epsilon
are similarly inseparable without an optical convention.

### Baseline and timing

Profile one baseline per trace: for equal-weight points, `b_i = mean(A_obs -
signal_i)`. For weighted residuals use the corresponding weighted mean. Baselines
are not forced to be the first reading. Drift is off by default, with an opt-in
shared per-group term and diagnostics of correlation with kinetic parameters.

Define acquisition time zero from the plate time coordinate's first acquisition,
not independently from each well's first nonmissing value. Default assumes
reaction started then; retain a warning/convention in provenance. Allow known
nonnegative `reaction_age_at_first_read_s` (global or per plate) to represent
reaction before acquisition. With optional nonnegative lag after reaction start:

```text
reaction_time = acquisition_time + reaction_age_at_first_read_s
kinetic_time = max(reaction_time - lag_s, 0)
```

Default lag is fixed zero, not automatically fitted. Optional fitted lag is
shared within a group; preserve actual plate timing. Reaction age and lag are
confounded, so never fit both freely from the same data without external
constraints. Missing early readings do not reset reaction age or pool state.

## 5. Direction auto-detection

Run per kinetic group after saturation trimming, selection and missing-point
masking, before fitting. Vote using valid positive-S0, positive-E kinetic traces.
Use a robust elapsed-time slope or median pairwise slope plus a robust comparison
of early/late windows. Exclude nearly flat/noisy traces using a documented noise/
amplitude threshold (e.g. detrended residual MAD and a minimum AU change). Each
informative trace gets one vote, regardless of amplitude or number of points.

Proposed configurable default: at least 3 informative traces and >=75% agreeing
signs. One/two-trace groups can use explicit override or a clearly reported
low-evidence policy, not silent certainty. If insufficient evidence or conflicting
signals, return `ambiguous_direction` and skip that group without fabricating
a fit. Other groups may proceed. `--signal-direction increasing|decreasing`
overrides auto and records override source. Log informative/flat/conflicting
counts, agreement, inferred direction and thresholds. Do not flip signs trace
by trace to force incompatible wells into one fit.

## 6. Fitting and identifiability

Fit shared positive `Km`, `kcat`, and optical coefficient within each group by
default. One optical coefficient per kinetic group is the safe MVP scope; never
one free scale per well. A later explicit reporter/optical-group variable can
share one epsilon across enzymes with the same reporter, wavelength and optical
convention. Such sharing requires true joint optimization or an externally
calibrated value, not sequential fits pretending to be joint.

Support positive bounded transforms, fixed parameters, reproducible multi-start,
and saved optimizer history. Reuse Adam -> L-BFGS best-state restoration; ensure
recorded losses and parameter values refer to the same evaluation. Default
objective can retain equal-trace normalized MSE with a documented 0.01 AU scale
floor. Expose raw-MSE vs normalized-MSE and floor options. Keep control/noise
traces from dominating normalization; report scales and actual trace weights.
Expose configurable starts/bounds without hard-coded ASN/NADH constants.

Required diagnostic distinctions:

- Optimizer converged does not mean parameters are identifiable or model is good.
- With little substrate depletion, initial slopes identify primarily epsilon *
  kcat, and often only kcat/Km when all S0 << Km. Free epsilon can yield many
  similar fits. Substantial depletion and an appropriate substrate range can
  help separate parameters but do not guarantee it.
- Provide bound-hit flags, competing multi-start parameter/loss summaries,
  substrate/enzyme coverage, predicted depletion, and residual diagnostics.
  For each free optical coefficient, provide a configurable profile over fixed
  epsilon with reoptimized kinetic parameters; use it or sensitivity rank/
  conditioning to flag practical non-identifiability. Report raw diagnostic
  evidence rather than unexplained confidence scores.
- A dataset with a single concentration, no signal or no curvature may produce
  an estimate with an explicit warning, but must not be advertised as a
  trustworthy Km/kcat recovery. Flat direction-auto groups are skipped.
- Profile objectives and optimizer Hessians are not automatically calibrated
  confidence intervals. If intervals are included, document assumptions and
  use trace/replicate-level bootstrap where appropriate; absent intervals are
  better than unjustified precision.

Recommend fixing a known optical coefficient when free-scale fits are weakly
identified. The default remains fitted optical scale as the user requested.

## 7. Proposed Python and CLI interfaces

Public objects: `FitConfig`, `fit_dataset`, `FitResults`; a results object exposes
labeled parameter/prediction/diagnostic datasets and `write(outdir)`. Configuration
is serializable to JSON. Avoid requiring CLI use from a notebook.

Illustrative CLI, not an already-installed command:

```sh
progress-fit experiment.nc --outdir results
progress-fit experiment.zarr --extinction-coefficient-M-cm 6000 --outdir calibrated
progress-fit experiment.nc --signal-direction decreasing --reference-path-length-cm 1
progress-fit experiment.nc --group-var fit_group --merge-plates --outdir joint
```

The value 6000 is only a synthetic example, not a NADH/reporter recommendation.

Minimum arguments: input, outdir, input format auto/netcdf/zarr, absorbance/time/
substrate/enzyme/identity dimension and variable mappings, group-var/group-by,
merge-plates, sample selection/roles, signal-direction, corrected/raw signal
mode, reference-path-length-cm, pathlength-var or fixed raw pathlength,
reporter-stoichiometry, extinction-coefficient-M-cm (fixed when supplied),
fix-alpha-AU-per-mM, saturation-var/saturation-time-var, trim-at-first-missing,
reaction-age-at-first-read-s, optional fitted lag, minimum-points/time-window,
starts/bounds, optimizer steps/tolerances, weighting/floor, multi-start/seed,
identifiability profile policy, and device/dtype. Implement readable help with
units and default meaning; use a config file for complex per-plate mappings.

`progress-fit validate` or `--validate-only` should report schema/selection/group
and timing/optical assumptions without optimizing. A dataset inspection summary
should show group sizes, S0/E ranges and missing metadata. Machine-readable exit
policy: 0 for all requested groups successfully processed (warnings permitted),
nonzero for invalid input or no valid fit; distinguish partial success in JSON
and a documented CLI exit status. Never let one bad group silently disappear.

## 8. Output contract

Produce parameter CSV + JSON/config/provenance, per-trace QC CSV, and labeled
NetCDF (or Zarr) predictions/results. Include all requested groups, even failed
ones, with status/reason. Suggested outputs:

- `fit_parameters.csv`: group IDs, Km_mM, kcat_per_s, epsilon_per_M_cm,
  alpha_AU_per_mM (or per-trace values when lengths vary), direction, optical
  assumptions, fixed/fitted flags, loss, convergence and identifiability flags.
- `trace_diagnostics.csv`: plate/well/group IDs, S0/E/Vmax, point counts/masks,
  baseline, noise/dynamic range, residual metrics, trim causes/times.
- `fit_results.nc`: prediction AU, residual AU, substrate/product mM, selected
  masks aligned to original plate/time/well coordinates; labeled group parameters.
- `run.json`: config, software versions/seed, optical and timing conventions,
  selected variables, input provenance, warnings, group statuses, optimization
  settings and multistart/profile summaries.
- Progress/residual plots per group, a compact parameter summary, and optimizer/
  profile plots where enabled. Display both observed and predicted data with
  exclusion markers, both directions, actual enzyme/initial substrate labels.

Serialize object coordinates/metadata to stable strings or appropriately typed
arrays for NetCDF; no opaque Python object pickle. Preserve input IDs but do not
include private collaborator data in public examples. User-run provenance can
include their input path; public docs/tests may not contain personal paths.

## 9. Required acceptance tests and synthetic example

Port donor regression tests appropriate to the generic model; do not claim donor
tests validate the new xarray utility before it is built. Add independently
generated synthetic fixtures (ODE integration or an independent integrated
solver for key recovery tests, not only the identical prediction function).

1. Recover known Km, kcat and fitted optical scale from high-information synthetic
   progress curves spanning below/near/above Km, with varying E and substantial
   depletion. Fixed-scale, noise-free recovery target ~1% relative error; fitted-
   scale target ~2% under favorable data, bounded by numerical precision. Choose
   test generation/settings so failure indicates a regression, not poor design.
2. Recover kcat with per-well E; prove that wrong constant E gives a different
   prediction. Check uM -> mM conversion exactly; raw optical epsilon -> alpha
   conversion and normalized reference length must have independent unit tests.
3. Increasing/decreasing data, majority detection with flat/noisy controls,
   conflicting signs, too few informative traces, override and recorded votes.
4. First saturation trims all later data even when later values are numeric;
   ordinary isolated missing values do not. Both question-mark strings and
   masks/per-trace times, insufficient points, all-saturated traces.
5. Reverse `well_384,plate_read` metadata dimensions, shuffled well coordinates,
   broadcasting scalars, duplicate IDs/times, invalid/missing metadata, unknown
   units, non-384 well dimension names. Explicit alignment error cases.
6. Multiple enzymes and conditions, repeated A1 IDs across plates, shared groups
   only when requested; substrate and E variation do not accidentally split fits.
7. Corrected AU does not apply measured length twice. Default 1-cm assumption
   warning, user-supplied reference, raw varying lengths, fixed epsilon and fixed
   alpha mutual exclusion, reporter stoichiometry scaling.
8. Reaction age and missing early observations preserve kinetics; no per-trace
   re-zero. Lag vs reaction-age configurations are distinct and recorded.
9. Low-depletion/free-scale and narrow substrate-range fixtures trigger weak-
   identification flags; broad information resolves them. Multi-start finite
   failures and bounds handled transparently; optimizer returns best finite state.
10. CLI/API parity and NetCDF/Zarr round trips, serializable metadata, invalid
    input/partial-success exit policy, nonmutation of input, seeded reproducibility.

Synthetic example generator should build the collaborator-compatible schema
with dummy enzyme IDs, 2 plates, at least 2 enzymes, repeated well names,
S0 dilution series, variable E, both signal directions across separate groups,
ordinary missing points and saturation masks. Optional full 384-well schema
can include explicit empty/control wells. No sample private data is required.

## 10. Extensions, separate from MVP

These are available donor concepts to preserve for later opt-in interfaces,
not unrequested extra work required to call the generic MVP complete.

### First-order coupled reporter

Rename the ASNase/ASP/AST concepts as primary enzyme, intermediate Q and reporter:

```text
v_A(t) = kcat * E_mM * S(t)/(Km + S(t))
dQ/dt = v_A(t) - k_B*Q(t), Q(0)=Q0
v_reporter(t) = k_B*Q0*exp(-k_B*t)
              + k_B*integral_0^t exp(-k_B*(t-u))*v_A(u) du
R(t) = integral_0^t v_reporter(u)du = Q0 + P(t) - Q(t)
A(t) = baseline + direction*alpha*R(t) + drift*t
```

Fit absorbance to cumulative R, not to instantaneous v_reporter. A supplied k_B
in s^-1 is an effective first-order rate incorporating reporter enzyme conditions;
do not independently demand AST and MDH concentrations to resolve this already-
parameterized model. If reporter concentrations change across wells, map the
appropriate effective k_B per well or separate conditions. State the assumption
that any remaining reporter steps are fast/unlimited. Propagate from the actual
reaction origin even when the first observation is later; test quadrature against
an independent ODE solver at large/small k_B and irregular sampling intervals.

### Initial intermediate contamination and reporter capacity

Optional fixed/fitted contamination fraction f in [0,1] uses a **nominal total
stock** definition: actual S0=(1-f)*S0_nominal, Q0=f*S0_nominal. If the input S0
is known pure substrate concentration, instead use Q0=f*S0 or an independent
Q0 mapping. These definitions are not interchangeable; require a named policy.
Default f=0. A directly supplied per-well Q0 is also possible. Fit a shared
fraction per stock/group, not independent per-well impurity unless justified.
For instantaneous reporters, initial impurity is mostly baseline-confounded;
do not imply automatic recovery.

Default capacity is infinite. Optional simple demand clipping `R_observed =
min(R, C)` preserves the donor observation approximation; state that it is not
a mechanistic substrate-limited reporter enzyme model. C is positive in reporter-
equivalent mM; divide by stoichiometry if the model stores substrate equivalents.
Support fixed per-well C or fitted shared C with explicit precedence/conflict
handling. Capacity, epsilon and impurity can be correlated. Distinguish modeled
reporter exhaustion from instrument saturation flags at every interface.

### Initial-rate comparison

Fit an early slope per trace on an explicit time/point window. Convert to primary
rate using `v0_i = slope_i/(direction*alpha_i)`, then regress
`v0_i = kcat*E_i_mM*S0_i/(Km+S0_i)`. Equivalently normalize by E_i. Report window,
slope uncertainty and exclusions. Free epsilon cannot be separately recovered
from initial-rate slopes alone: use an externally fixed coefficient, inherit one
from progress fitting with an explicit dependency, or report only apparent
optically scaled parameters. A lagged/coupled reporter early slope is not
automatically the primary enzyme's initial rate; warn or use the coupled model.

Automated trace pruning/time-window scans remain opt-in, audited extensions.

## 11. Implementation sequence and release gate

1. Verify donor baseline; create generic package/tests and schema/unit adapter.
2. Implement per-well E, direction and optical conventions with synthetic recovery.
3. Implement grouping/saturation/QC and Python API, then file CLI/exports/plots.
4. Add multi-start and identifiability diagnostics; pass acceptance suite and
   document practical limitations and all assumed conventions.
5. Choose a license with the owner, review source/docs for private metadata and
   residual ASNase-specific defaults, then hand off locally. Publishing/pushing
   requires a separate request. Do not overwrite the original donor repository.
