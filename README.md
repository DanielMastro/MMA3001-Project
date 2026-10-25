# MMA3001 Project: FEA flux maps to inductance and torque for permanent-magnet motors

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

Daniel Mastromanno, Monash University, MMA3001 Project (2026)

## What this tool is for

Finite-element analysis (FEA) of a permanent-magnet synchronous motor (PMSM) is
slow, so a design is usually simulated at only a coarse grid of currents. This
tool turns that coarse FEA output into a continuous model of the motor's magnetic
behaviour, so torque and inductance can be read at **any** current inside the
simulated range.

It reads Ansys Maxwell 2D "winding table" exports and builds a flux map, the dq
flux linkages ψd and ψq at each simulated current. It then fits a flux model
ψd, ψq = f(Id, Iq). From that model it derives torque and inductances:

* apparent (secant) inductances
* incremental (differential) inductances
* the dq cross-coupling term

The intended use is **comparing motor geometries before manufacture**, and
giving the motor controller dense torque, flux and inductance maps. Every design
goes through the same script, so their predicted maps can be compared like for
like.

The tool was developed on the FEA of the Cobalt Formula Student motor. It was
then validated, **unchanged**, on an independent open-access interior PM motor
dataset [3] (the "Zenodo IPM").

## Inputs and outputs

**Input.** An Ansys Maxwell 2D winding-table export (CSV), as in
`Input_Data/Cobalt/cobalt_input_data.csv`. It has one block of rows per operating
point (a current magnitude J and current angle TimingAdjustment), covering the
rotor sweep, with the phase currents [mA], phase flux linkages [Wb] and 3×3
phase inductance matrix [µH, nH]. Three motor settings go with it: the turns
ratio (real turns / model turns), the stack factor (2D-to-3D correction) and the
number of pole pairs.

**Outputs.** `run_all.py` writes:

| File | Contents |
|---|---|
| `Output_Data/Cobalt/cobalt_flux_map.csv` and `.png` | The FEA flux map: Id, Iq, ψd, ψq, inductances and torque at each FEA operating point |
| `Output_Data/Cobalt/cobalt_output_map.csv` | The final model's predictions on a 1 A × 1° grid (0–113 A, 0–90°): torque, flux, apparent and incremental Ld, Lq and Ldq, saliency and an `Extrapolated` flag |
| `Output_Data/Cobalt/cobalt_output_at_fea_points.csv` | The same outputs at the FEA points, for checking against the FEA |
| `Output_Data/Zenodo_IPM/zenodo_ipm_flux_map.csv` and `.png` | The flux map of the validation motor |
| `Output_Data/Validation/*.csv` | Every comparison, validation and performance table in the report |
| `Docs/test_report.txt` | The results of the 22 tests |

Units: current in A (peak, amplitude-invariant dq), flux linkage in Wb,
inductance in H (µH in the output maps), torque in Nm and current angle β in
degrees (0° is pure +q, 90° is pure −d).

## Features

* **Input checks.** Reads Maxwell winding-table CSVs and stops with a clear message if the file or a column is missing, values are NaN, operating-point blocks are incomplete, there are no no-load rows, or the phase currents are unbalanced.
* **dq conversion.**
  * amplitude-invariant Park transform [1, 2]
  * rotor angle found from the direction of the no-load flux
  * averaging over whole electrical cycles
  * scaling from the 2D model to the real machine (turns ratio, stack factor)
* **Flux map outputs.**
  * magnet flux ψpm and the cross-saturated ψpm,eff(Iq)
  * Maxwell matrix inductances, plus apparent and incremental inductances on the FEA grid [5]
  * torque from T = 1.5·p·(ψd·Iq − ψq·Id)
  * colour-map plots of the map
* **Four flux models with one interface.** `constant_inductance()`, `lookup_table()`, `polynomial_regression(degree)` and `neural_network(size)` (a scikit-learn MLP [6]). Each provides `fit`, `predict_flux`, `flux_gradient` and an extrapolation flag.
* **Derived outputs at any (Id, Iq).**
  * torque
  * apparent and incremental Ld, Lq and Ldq, from analytic gradients or 3-point and 5-point finite differences
  * an `Extrapolated` column that marks points outside the FEA range
* **Validation.**
  * model settings chosen by repeated 5-fold cross-validation
  * the outer current rings held out to test extrapolation
  * a frozen-script test on the independent Zenodo motor (trained on a coarse subset, tested against its dense FEA points and FEA torque)
  * a seed study: the network's tests repeated at 10 random seeds, because its accuracy depends on its random starting weights
* **Zenodo converter.** `create_zenodo_input_data.py` rebuilds the Zenodo dq maps into the same winding-table format, so both motors go through identical code.
* **Performance tools.** Loop vs vectorised timing, `cProfile` profiling, and the parameter count, memory, FLOPs and run time of each model.
* **Tests.** 22 `pytest` tests, including a known-answer analytic motor whose exact inductances the pipeline must recover.

## Repository layout

```
MMA3001-Project/
├── Scripts/
│   ├── run_all.py                    reproduces every result and the test report (the only file you need to run)
│   ├── run_tests.py                  runs the pytest tests and saves Docs/test_report.txt
│   ├── preprocess.py                 winding table -> flux map
│   ├── models.py                     the four flux models
│   ├── derive.py                     torque and inductances from a flux model
│   ├── evaluate.py                   cross-validation, Zenodo validation, formula checks, performance
│   ├── create_zenodo_input_data.py   Zenodo dq maps -> winding-table CSV
│   └── Tests/                        pytest tests and the known-answer motor (conftest.py)
├── Input_Data/
│   ├── Cobalt/                       Cobalt Maxwell 2D winding table
│   └── Zenodo_IPM/                   original Zenodo files, the converted winding table and its provenance record
├── Output_Data/
│   ├── Cobalt/                       flux map, output maps, plot
│   ├── Zenodo_IPM/                   flux map and plot
│   └── Validation/                   every comparison, validation and performance table
├── Docs/                             pdoc HTML API documentation and the pytest report
├── History/                          earlier versions of the original inductance script (preprocess_v3.py was refactored into preprocess.py)
├── requirements.txt                  Python packages needed
├── pytest.ini                        test settings
└── LICENSE
```

## Environment and dependencies

* **Python 3.10 or newer**, and Git. No GPU is needed: a full run takes 1–2
  minutes on a laptop, and the cloned repository needs about 100 MB of disk.
* **Tested** on Windows with Python 3.14, which produced the results in this
  README, and on Linux with Python 3.13, a full re-run from a fresh clone that
  reached the same conclusions.
* **Packages.** `requirements.txt` lists them without fixed versions, so pip
  installs the latest ones. These versions were used for the Linux re-run:

| Package | Used for | Version tested |
|---|---|---|
| numpy | arrays, Park transforms, least squares | 2.5.3 |
| pandas | reading the winding tables, results tables | 3.0.5 |
| scipy | lookup-table interpolation | 1.18.1 |
| scikit-learn | the neural network (`MLPRegressor`) | 1.9.1 |
| matplotlib | flux-map plots | 3.11.2 |
| pytest | the 22 tests | 9.1.1 |
| pdoc | the HTML documentation in `Docs/` | 16.0.0 |

## Installation

```bash
git clone https://github.com/DanielMastro/MMA3001-Project.git
cd MMA3001-Project
python -m pip install -r requirements.txt
```

On Windows, use `py` instead of `python` if `python` is not found.

Optionally, install into a virtual environment so the packages stay separate
from other projects:

```bash
python -m venv .venv
.venv\Scripts\activate             # Windows; on Linux or macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Quick start

Run everything from the repository folder.

**Reproduce all results**

```bash
python Scripts/run_all.py
```

This takes 1–2 minutes. It writes every table and figure into `Output_Data/`
and the test report into `Docs/test_report.txt`.
The console shows each step:

1. convert the Zenodo data, if needed
2. build both flux maps
3. choose the model settings by cross-validation
4. compare the four models
5. write the Cobalt output maps
6. validate on Zenodo, and repeat the network's tests at 10 random seeds
7. run the formula checks
8. run the performance tests
9. run the pytest tests and save the report

All settings are in the `SETTINGS` block at the top of `run_all.py`, and no
file paths need editing. Setting `TIME_ZENODO_LOOP = False` skips the slowest
timing test, and `NN_SEEDS` sets the seeds used by the seed study.

**Run only the tests**

```bash
python Scripts/run_tests.py
```

This runs `pytest -v` and saves the full output to `Docs/test_report.txt`. You
should see `22 passed`. Plain `python -m pytest -v` also works, but it only
prints to the screen.

## Tutorial: torque and inductance at any current

This example builds the Cobalt flux map, fits a degree-6 polynomial flux model
and asks for three operating points. The third point is deliberately outside the
FEA current range (113 A).

```python
import sys
sys.path.insert(0, "Scripts")          # the modules live in Scripts/

import preprocess as pp
import models as md
import derive as dv

# 1. Winding table -> flux map (Cobalt: 2-turn 2D model scaled to 33 turns, 0.9 stack, 5 pole pairs)
flux_map = pp.build_flux_map("Input_Data/Cobalt/cobalt_input_data.csv",
                             turns_ratio=33 / 2, stack_factor=0.9, pole_pairs=5, verbose=False)
pts = pp.unique_operating_points(flux_map)

# 2. Fit a flux model psi_d, psi_q = f(Id, Iq)
model = md.polynomial_regression(6).fit(pts["Id"], pts["Iq"], pts["Psi_d"], pts["Psi_q"])

# 3. Torque and inductances at any current
out = dv.output_map(model, Id=[-20.0, -30.0, -50.0], Iq=[60.0, 100.0, 150.0], pole_pairs=5)
print(out[["Id", "Iq", "Torque_Nm", "Ld_app_uH", "Lq_app_uH", "Ld_inc_uH", "Lq_inc_uH",
           "Extrapolated"]].round(1).to_string(index=False))
```

Expected output:

```
   Id    Iq  Torque_Nm  Ld_app_uH  Lq_app_uH  Ld_inc_uH  Lq_inc_uH  Extrapolated
-20.0  60.0       20.6      393.0      454.9      446.5      413.9         False
-30.0 100.0       31.4      349.4      428.7      408.7      300.5         False
-50.0 150.0       42.1      288.3      329.3      272.0     -160.8          True
```

The third row shows why the `Extrapolated` flag exists. Outside the data, the
polynomial predicts a negative incremental inductance, which is physically
impossible. Treat any flagged row with care.

`neural_network(8)`, the model selected by `run_all.py`, gives 20.5, 31.4 and
40.6 Nm at the same points. Its results vary slightly between library versions.

**Using your own motor:** export the winding table from Maxwell with the same
columns as `Input_Data/Cobalt/cobalt_input_data.csv`. The file needs:

* one block of `NStepsPerElectricalCycle + 1` rows per operating point
* at least one no-load (J = 0) block

Then change the file path, `turns_ratio`, `stack_factor` and `pole_pairs` in the
example above.

## Test case

`Scripts/Tests/conftest.py` defines a known-answer motor with analytic flux
linkages. It has a magnet flux, a saturating d-axis, a constant q-axis and
cross-coupling. Its winding table is generated in the same format as the FEA
exports.

The tests check that:

* the pipeline recovers its exact flux and inductances (within 1 %)
* the 3-point and 5-point finite differences converge at their theoretical orders (h² and h⁴)
* the models reproduce cases they should fit exactly, and the network's analytic gradient matches finite differences
* the Park transforms round-trip
* bad input files are rejected

## Results at a glance

These numbers are from the author's run of `run_all.py`. Errors are RMS values:
flux as a percentage of ψpm, and torque as a percentage of the base torque
1.5·p·ψpm·I_max.

| Check | Result |
|---|---|
| Cobalt, 5-fold cross-validation, final model `neural_network(8)` | flux 0.21 %, torque 0.19 % |
| Cobalt, two outer current rings held out (extrapolation) | flux 0.53 % |
| Zenodo, frozen script, 540 points inside the training range | flux 0.25 %, torque 0.17 % (vs FEA torque) |
| Zenodo, 240 points outside the training range | flux 1.3 % (polynomial: 12.7 %) |
| Network trained from 10 random seeds | Cobalt cross-validation flux 0.21–0.49 % (polynomial: 0.21 %); Zenodo outside 1.3–2.3 % (polynomial: 12.7 %) |
| Torque from flux vs Zenodo FEA torque | 0.05 Nm RMS (peak 331 Nm) |
| Vectorised vs loop preprocessing | 34× (Cobalt), 68× (Zenodo), identical results |

Full tables are in `Output_Data/Validation/`.

## Reproducibility

* `run_all.py` regenerates every output from the input data with one command.
* The preprocessing, the constant-inductance, lookup-table and polynomial models
  and the formula checks are deterministic: a re-run gives the same numbers, to
  within floating-point rounding, on any machine.
* The neural network uses a fixed random seed (0), so a re-run on the same
  machine repeats exactly. On another platform or library version its numbers
  can differ slightly (Cobalt cross-validation flux error 0.205 % on Windows,
  0.207 % on Linux), but every conclusion is unchanged.
* The timings in `Output_Data/Validation/performance_*.csv` are wall-clock
  measurements, so they change from run to run.

## Limitations

* **Use inside the FEA range.** Results are reliable only there. Outside it, the
  `Extrapolated` flag is set and the error grows. The polynomial model can become
  non-physical (see the tutorial).
* **Inductances are least accurate at the edges of the data.** This applies to the
  outer current ring and the Iq = 0 and Id = 0 edges, because gradients there are
  one-sided.
* **Apparent inductance is sensitive at low current.** It divides a small flux
  difference by a small current.
* **The Zenodo winding table is a reconstruction.** It was rebuilt from published
  dq maps, not exported from Maxwell, so it tests the models and the dq pipeline
  but not raw Maxwell output.
* **The FEA is 2D.** End effects are represented only through the stack factor.
* **The network depends on its random seed.** Over 10 seeds its cross-validated
  error ranges from 0.21 % to 0.49 %. The delivered model uses seed 0, the best
  of the ten; inside the data the polynomial is equally accurate.

## Documentation

* **API documentation.** `Docs/index.html`, generated by pdoc from the docstrings
  in every module. The docstrings follow the NumPy style: each function lists its
  parameters with units, what it returns, the errors it raises and, for the main
  functions, a worked example. Open `Docs/index.html` in a browser to read it. To
  regenerate it:
  `python -m pdoc Scripts/preprocess.py Scripts/derive.py Scripts/models.py Scripts/evaluate.py Scripts/create_zenodo_input_data.py Scripts/run_all.py Scripts/run_tests.py -o Docs`
* **Test report.** `Docs/test_report.txt`, regenerated by `run_all.py` or `run_tests.py`.
* **Module docstrings.** Each script begins with a docstring explaining what it
  does and where it fits in the pipeline.
* **Zenodo dataset.** Original files and description are in
  `Input_Data/Zenodo_IPM/ReadMeData.txt` [3], and the converter's record of what
  it changed is in `zenodo_input_data_provenance.json`.

## Data and licence

Copyright (C) 2026 Daniel Mastromanno

The **code** in this repository is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your option)
any later version. It is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See [LICENSE](LICENSE) for details.

The **data** is not covered by the GPL:

* `Input_Data/Cobalt/` is Ansys Maxwell 2D FEA output for the Cobalt Formula
  Student motor (77 operating points, 0–113 A). Contact the author before
  reusing it.
* `Input_Data/Zenodo_IPM/` contains data by Mai et al. [3], licensed under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): `Psi_d.csv`,
  `Psi_q.csv`, `Torque.csv`, `ReadMeData.txt` and `GammaDefinition.pdf`.
  `create_zenodo_input_data.py` makes three files from them:
  `zenodo_input_data.csv` (an adapted version, with phase currents and flux
  linkages reconstructed from the published dq maps and inductance columns
  derived from them), `zenodo_input_data_operating_points.csv` (the FEA torque
  per operating point) and `zenodo_input_data_provenance.json`, which lists
  every change. `run_all.py` recreates these files if they are missing.

## How to cite

If you use this tool, please cite it:

> Mastromanno, D. (2026). *MMA3001 Project: FEA flux maps to inductance and torque for permanent-magnet motors* [Computer software]. GitHub. https://github.com/DanielMastro/MMA3001-Project

If you use the Zenodo motor data, also cite [3] and [4].

## Use of AI

Tools Used: Claude (Anthropic, Claude Opus 5.5, via claude.ai), September - October 2026.

AI was used in this project for the purpose of:

* Analysing the brief and my original script and suggesting the surrogate-model framing.
* Finding the Zenodo dataset and writing its converter so it can be used as input data.
* Refactoring my original script into preprocess.py and drafting the other modules and tests with docstrings and the comments.
* Providing templates for the report and readme.
* Explaining concepts that I then applied: secant and tangent inductance, cross-validation, stencil convergence, git and pytest.
* Finding relevant references which I validated myself to ensure reliability.

Level of contribution: High for the new code, roughly 70%. Low for the preprocessing method, which is largely my original script (preprocess_v3.py). None for the FEA data or the decisions below.

Why AI was used: Using several tools that were new to me (scikit-learn, pytest, pdoc). AI made a complete, tested and documented pipeline feasible, and it gave a second opinion on methods. Use of AI significantly increased my proficiency in coding as well as developed my knowledge for the numerical solving methods used throughout this unit, showing me how they can be practically applied to a real engineering example.

How AI outputs were checked:

1. Known-answer tests, rather than trusting code by reading it (Section 5).
2. Comparison with my original script's output during the refactor, which was identical.
3. An independent check against Zenodo's FEA torque, which is real and validated.
4. Running everything myself (Windows, Python 3.14) and reviewing every table and plot.
5. Checking the dataset licence and the references against their sources.

My decisions:

* The engineering problem, the model’s purpose and how the problem should be solved.
* Developing on the team's FEA and validating the frozen script on a different motor, training on Cobalt rather than the larger Zenodo set to better suit the team’s needs.
* Keeping my original script's preprocessing methods to avoid losing accuracy of the input data.
* The repository structure and naming to maintain readability and easy understanding of the repository.
* Interpreting the results, including keeping the network after the seed study and deciding which model should be primarily used by the model compared to the others that were tested.
* The engineering decisions, the data and the validation of outputs were completed by myself, and every number comes from my own run of the code rather than depending on AI outputs.

AI errors detected:

* An extrapolation flag that flagged the FEA's own edge points, whose angles sit 0.04° off-axis.
* A convergence test at h = 40 A, outside the asymptotic range.
* A test demanding 1 % accuracy where the derivative is one-sided.
* Code that broke on my newer scikit-learn, and on paths from another folder.
* An over-scoped initial plan for the project that was not representative of the project requirements
* Selecting the network on a single seed, which overstated its accuracy until the seed study.

Effect on understanding and workflow: My work shifted from writing code to specifying, reviewing and testing it, as well as determining the most suitable numerical method for the engineering problem and why. AI output looked plausible even when it was wrong, so I had to understand each method well enough to test it and identify when it was wrong which I actually found quite challenging. That is how I learned the convergence orders, why secant and tangent inductances differ, and why the matrix torque formula fails.


## References

1.	R. H. Park, "Two-reaction theory of synchronous machines: generalized method of analysis – Part I," Transactions of the AIEE, vol. 48, no. 3, pp. 716–727, 1929. doi:10.1109/T-AIEE.1929.5055275
2.	P. C. Krause, O. Wasynczuk, S. D. Sudhoff and S. Pekarek, Analysis of Electric Machinery and Drive Systems, 3rd ed. Hoboken, NJ: Wiley-IEEE Press, 2013.
3.	A. Mai, X. Liu, B. Wagner and M. Hofmann, "Current Harmonics Minimization of PMSM Based on Iterative Learning Control and Neural Networks: Motor Data," Zenodo, v1, 2023, CC BY 4.0. doi:10.5281/zenodo.7866449
4.	A. Mai, X. Liu, B. Wagner and M. Hofmann, "Current Harmonics Minimization of Permanent Magnet Synchronous Machine Based on Iterative Learning Control and Neural Networks," Machines, vol. 11, no. 8, p. 784, 2023. doi:10.3390/machines11080784
5.	B. Štumberger, G. Štumberger, D. Dolinar, A. Hamler and M. Trlep, "Evaluation of saturation and cross-magnetization effects in interior permanent-magnet synchronous motor," IEEE Transactions on Industry Applications, vol. 39, no. 5, pp. 1264–1271, 2003.
6.	F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.
7.	P. Virtanen et al., "SciPy 1.0: fundamental algorithms for scientific computing in Python," Nature Methods, vol. 17, pp. 261–272, 2020.
8.	C. R. Harris et al., "Array programming with NumPy," Nature, vol. 585, pp. 357–362, 2020.
9.	D. C. Liu and J. Nocedal, "On the limited memory BFGS method for large scale optimization," Mathematical Programming, vol. 45, pp. 503–528, 1989.
10.	R. L. Burden, J. D. Faires and A. M. Burden, Numerical Analysis, 10th ed. Boston, MA: Cengage Learning, 2016.
11.	T. Hastie, R. Tibshirani and J. Friedman, The Elements of Statistical Learning, 2nd ed. New York: Springer, 2009.
