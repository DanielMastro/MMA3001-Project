"""
run_all.py - one command reproduces every result in the project.

    python Scripts/run_all.py          (run from the repo folder)

Where this fits in the project
------------------------------
This is the only file you need to run. It calls the other modules in order and
writes every table and figure used in the report into Output_Data/. All
settings for both motors are in the SETTINGS block below; nothing else in the
code needs editing, and there are no absolute paths.

Steps
  1. Create Input_Data/Zenodo_IPM/zenodo_input_data.csv if it is missing
     (create_zenodo_input_data.py).
  2. Preprocess both motors into flux maps (preprocess.py).
  3. Choose the polynomial degree and network size by Cobalt cross-validation
     (evaluate.py). The settings are frozen after this step.
  4. Compare the four flux models on Cobalt and pick the final one
     (lowest cross-validated flux error).
  5. Fit the final model to all Cobalt points and write the output maps for
     the inverter team: flux, inductances and torque (derive.py).
  6. Validate the frozen script on the independent Zenodo motor.
  7. Formula checks: flux torque vs 07-07's matrix torque, and whether the
     matrix inductances reproduce the FEA flux.
  8. Performance: 07-07 loop vs vectorised, profiling, and the cost of each model.

Output_Data/
  Cobalt/       cobalt_flux_map.csv/.png, cobalt_output_map.csv,
                cobalt_output_at_fea_points.csv, cobalt_07-07_format.csv
  Zenodo_IPM/   zenodo_ipm_flux_map.csv/.png
  Validation/   every comparison, validation and performance table
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

import derive as dv
import evaluate as ev
import models as md
import preprocess as pp
from create_zenodo_input_data import create_zenodo_input_data

# ============================================================================
# SETTINGS
# ============================================================================
ROOT = Path(__file__).resolve().parent.parent            # the repo folder
INPUT = ROOT / "Input_Data"
OUTPUT = ROOT / "Output_Data"
VALIDATION = OUTPUT / "Validation"

MOTORS = {
    # Team FEA: 2-turn 2D Maxwell model -> 33-turn machine, 0.9 stack factor, 5 pole pairs
    "Cobalt": {"input": INPUT / "Cobalt" / "cobalt_input_data.csv",
               "turns_ratio": 33 / 2, "stack_factor": 0.9, "pole_pairs": 5},
    # Open Zenodo IPM (24 slots, 16 poles): already full-machine values, so 1 and 1
    "Zenodo_IPM": {"input": INPUT / "Zenodo_IPM" / "zenodo_input_data.csv",
                   "fea_torque": INPUT / "Zenodo_IPM" / "zenodo_input_data_operating_points.csv",
                   "turns_ratio": 1.0, "stack_factor": 1.0, "pole_pairs": 8},
}

POLY_DEGREES = (2, 3, 4, 5, 6)     # polynomial degrees tried by cross-validation
NN_SIZES = (8, 16, 32)             # neurons per hidden layer tried by cross-validation
CV_FOLDS, CV_REPEATS = 5, 3        # 5-fold cross-validation, repeated with 3 shuffles
HELD_OUT_RINGS = 2                 # outer current rings held out for extrapolation
OUTPUT_GRID_STEP = {"dI": 1.0, "dbeta": 1.0}   # output map: 1 A x 1 deg
TIME_ZENODO_LOOP = True            # time 07-07's row loop on Zenodo's 140,000 rows (about 10-60 s)


def heading(text):
    """Print a section heading so the console output is easy to follow."""
    print("\n" + "=" * 72 + f"\n{text}\n" + "=" * 72)


def save(df, path):
    """Write a table and say where it went."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  saved {path.relative_to(ROOT)}")


def main():
    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    heading("1. Zenodo input data")
    zen_cfg = MOTORS["Zenodo_IPM"]
    if zen_cfg["input"].exists() and zen_cfg["fea_torque"].exists():
        print(f"  found {zen_cfg['input'].relative_to(ROOT)}")
    else:
        src = INPUT / "Zenodo_IPM"
        create_zenodo_input_data(src / "Psi_d.csv", src / "Psi_q.csv", src / "Torque.csv",
                                 zen_cfg["input"], pole_pairs=zen_cfg["pole_pairs"])

    # ------------------------------------------------------------------
    maps = {}
    for name, cfg in MOTORS.items():
        heading(f"2. Preprocess {name}")
        maps[name] = pp.build_flux_map(cfg["input"], cfg["turns_ratio"], cfg["stack_factor"],
                                       cfg["pole_pairs"], fea_torque_csv=cfg.get("fea_torque"))
        save(maps[name], OUTPUT / name / f"{name.lower()}_flux_map.csv")
        pp.plot_flux_map(maps[name], f"{name} FEA flux map",
                         OUTPUT / name / f"{name.lower()}_flux_map.png")
    cobalt, zenodo = maps["Cobalt"], maps["Zenodo_IPM"]
    p_cob, p_zen = MOTORS["Cobalt"]["pole_pairs"], MOTORS["Zenodo_IPM"]["pole_pairs"]
    save(pp.export_07_07_format(cobalt), OUTPUT / "Cobalt" / "cobalt_07-07_format.csv")

    # ------------------------------------------------------------------
    heading("3. Choose model settings by Cobalt cross-validation")
    settings_cv, degree, size = ev.select_settings(cobalt, p_cob, POLY_DEGREES, NN_SIZES,
                                                   CV_FOLDS, CV_REPEATS)
    print(settings_cv.round(3).to_string(index=False))
    print(f"  chosen: polynomial degree {degree}, network size {size}  (now frozen)")
    save(settings_cv, VALIDATION / "cobalt_settings_cv.csv")

    # The four models with their frozen settings, used for everything below
    factories = {
        "constant_inductance": md.constant_inductance,
        "lookup_table": md.lookup_table,
        f"polynomial_regression({degree})": lambda: md.polynomial_regression(degree),
        f"neural_network({size})": lambda: md.neural_network(size),
    }

    # ------------------------------------------------------------------
    heading("4. Compare the models on Cobalt (alternatives)")
    comparison = ev.compare_models(factories, cobalt, p_cob, CV_FOLDS, CV_REPEATS, HELD_OUT_RINGS)
    print(comparison.round(3).to_string(index=False))
    save(comparison, VALIDATION / "cobalt_model_comparison.csv")
    final_name = comparison.loc[comparison["cv_flux_rmse_pct"].idxmin(), "model"]
    print(f"  final model (lowest cross-validated flux error): {final_name}")

    # ------------------------------------------------------------------
    heading("5. Output maps for the inverter team (Cobalt)")
    pts = pp.unique_operating_points(cobalt)
    final = factories[final_name]().fit(pts["Id"], pts["Iq"], pts["Psi_d"], pts["Psi_q"])
    lo, hi = final.beta_range
    Id, Iq = dv.polar_grid(final.I_base, round(lo), round(hi), **OUTPUT_GRID_STEP)
    grid_map = dv.output_map(final, Id, Iq, p_cob)
    save(grid_map, OUTPUT / "Cobalt" / "cobalt_output_map.csv")
    save(dv.output_map(final, pts["Id"], pts["Iq"], p_cob),
         OUTPUT / "Cobalt" / "cobalt_output_at_fea_points.csv")
    print(f"  {len(grid_map)} operating points, |I| 0-{final.I_base:.1f} A, beta "
          f"{round(lo)}-{round(hi)} deg; peak torque {grid_map['Torque_Nm'].max():.2f} Nm")

    # ------------------------------------------------------------------
    heading("6. Validate the frozen script on the Zenodo motor")
    validation = ev.zenodo_validation(factories, zenodo, p_zen)
    print(validation.round(3).to_string(index=False))
    save(validation, VALIDATION / "zenodo_validation.csv")

    # ------------------------------------------------------------------
    heading("7. Formula checks")
    torque_check = pd.concat([ev.torque_formula_comparison(cobalt, p_cob).assign(motor="Cobalt"),
                              ev.torque_formula_comparison(zenodo, p_zen).assign(motor="Zenodo_IPM")])
    print(torque_check.round(3).to_string(index=False))
    save(torque_check, VALIDATION / "torque_formula_comparison.csv")
    consistency = ev.matrix_inductance_consistency(cobalt)
    print(consistency.round(3).to_string(index=False))
    save(consistency, VALIDATION / "cobalt_matrix_inductance_consistency.csv")

    # ------------------------------------------------------------------
    heading("8. Performance")
    timing = []
    for name, cfg in MOTORS.items():
        if name == "Zenodo_IPM" and not TIME_ZENODO_LOOP:
            continue
        settings = {k: cfg[k] for k in ("turns_ratio", "stack_factor", "pole_pairs")}
        repeats = 3 if name == "Cobalt" else 1
        timing.append(ev.time_preprocessing(cfg["input"], settings, repeats).assign(motor=name))
    timing = pd.concat(timing)
    print(timing.round(4).to_string(index=False))
    save(timing, VALIDATION / "performance_loop_vs_vectorised.csv")

    for name, cfg in MOTORS.items():
        settings = {k: cfg[k] for k in ("turns_ratio", "stack_factor", "pole_pairs")}
        save(ev.profile_pipeline(cfg["input"], settings),
             VALIDATION / f"performance_profile_{name.lower()}.csv")

    costs = ev.model_costs(factories, cobalt)
    print(costs.round(3).to_string(index=False))
    save(costs, VALIDATION / "performance_model_costs.csv")

    heading(f"Done in {time.perf_counter() - t_start:.0f} s")


if __name__ == "__main__":
    main()
