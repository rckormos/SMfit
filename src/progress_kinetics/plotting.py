"""Headless progress, residual, parameter and optimization plots."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_results(results, outdir):
    root = Path(outdir) / "plots"
    root.mkdir(exist_ok=True)
    c = results.run["config"]
    ds = results.predictions
    for number, row in enumerate(results.parameter_rows):
        if row["status"] not in ("success", "partial_success"):
            continue
        traces = [
            t
            for t in results.trace_rows
            if t["group_id"] == row["group_id"] and t["status"] == "fitted"
        ]
        # Paginate to keep large groups readable, never silently omit traces.
        for page, offset in enumerate(range(0, len(traces), 8)):
            chosen = traces[offset : offset + 8]
            fig, axes = plt.subplots(
                len(chosen),
                2,
                figsize=(12, max(3, 2.2 * len(chosen))),
                squeeze=False,
                constrained_layout=True,
            )
            for j, trace in enumerate(chosen):
                # Index through original labels, including numeric identifiers.
                pi = [str(v) for v in ds[c["plate_dim"]].values].index(trace["plate"])
                wi = [str(v) for v in ds[c["well_dim"]].values].index(trace["well"])
                curve = ds.isel({c["plate_dim"]: pi, c["well_dim"]: wi})
                t = curve.acquisition_time_s.values
                y = curve.observed_AU.values
                used = curve.used_for_fit.values.astype(bool)
                ax, rx = axes[j]
                ax.plot(t[used], y[used], ".", ms=4, label="used")
                excluded = ~used & np.isfinite(y)
                ax.scatter(
                    t[excluded], y[excluded], marker="x", color="gray", s=15, label="excluded"
                )
                order = np.argsort(t)
                ax.plot(t[order], curve.prediction_AU.values[order], "-", lw=1, label="model")
                if trace["trim_time_s"] is not None:
                    ax.axvline(trace["trim_time_s"], color="orange", ls=":", label="trim onset")
                ax.set_title(
                    f"{trace['plate']}/{trace['well']}: S0={trace['S0_mM']:g} mM, "
                    f"E={trace['enzyme_uM']:g} uM",
                    fontsize=9,
                )
                ax.set_ylabel("Absorbance (AU)")
                rx.plot(t[used], curve.residual_AU.values[used], ".", ms=4)
                rx.axhline(0, color="gray", lw=0.6)
                rx.set_ylabel("Residual (AU)")
                for a in (ax, rx):
                    a.set_xlabel("Acquisition time (s)")
                if j == 0:
                    ax.legend(fontsize=7)
            fig.suptitle(
                f"Group {number + 1}: {row['direction']}; "
                f"Km={row['Km_mM']:.4g} mM, kcat={row['kcat_per_s']:.4g} s⁻¹"
            )
            fig.savefig(root / f"group_{number:03d}_progress_{page:02d}.png", dpi=140)
            plt.close(fig)
        report = results.run["group_reports"][row["group_id"]]
        fig, ax = plt.subplots(figsize=(7, 3), constrained_layout=True)
        for attempt in report["multistart"]:
            hist = attempt["history"]
            ax.semilogy(
                [h["evaluation"] for h in hist],
                [max(h["loss"], 1e-30) for h in hist],
                label=f"Start {attempt['start_index']}",
            )
        ax.set(xlabel="Objective evaluation", ylabel="Loss", title="Optimizer histories")
        ax.legend()
        fig.savefig(root / f"group_{number:03d}_optimizer.png", dpi=140)
        plt.close(fig)
        profile = report["identifiability"]["optical_profile"]
        if profile:
            fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)
            usable = [r for r in profile if "loss" in r]
            ax.plot([r["optical"] for r in usable], [r["loss"] for r in usable], "o-")
            ax.set(
                xlabel="Fixed optical scale (epsilon or alpha)",
                ylabel="Reoptimized loss",
                title="Optical profile (not a confidence interval)",
            )
            fig.savefig(root / f"group_{number:03d}_profile.png", dpi=140)
            plt.close(fig)
    fitted = [r for r in results.parameter_rows if r["status"] in ("success", "partial_success")]
    if fitted:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3), constrained_layout=True)
        for ax, name, unit in zip(axes, ("Km_mM", "kcat_per_s"), ("mM", "s⁻¹")):
            ax.plot(range(1, len(fitted) + 1), [r[name] for r in fitted], "o")
            ax.set(xlabel="Successful group index (CSV order)", ylabel=f"{name} ({unit})")
            ax.set_yscale("log")
        fig.savefig(root / "parameter_summary.png", dpi=140)
        plt.close(fig)
