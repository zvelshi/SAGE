# SAGE (Suspension Analysis & Geometry Evaluator)

SAGE is a Python-based simulation suite for designing, analyzing, and optimizing off-road vehicle suspension geometry. This tool provides kinematic analysis, quasi-static dynamic terrain simulation, and hardpoint optimization for **Double A-Arm** (Front) and **Semi-Trailing Link** (Rear) suspension types.

## Key Capabilities

* **Kinematic Analysis (`kin`)**:
    * **Sweep Suspension**: Travel (Bump/Droop), Steering, and combined interactions (Droop/Steer, Jounce/Steer).
    * **Calculated Metrics**: Camber, Caster, Toe, and CV Joint angles.
    * **Front Steer (Ackermann) Geometry**: Analyze steering geometry percentages and curves across full rack travel.
    * **Axle Travel (Heave/Roll)**: Sweep both corners of an axle together — same-direction (heave) or opposite (roll) — for camber/caster/toe/motion-ratio, track change, and roll center.
    * **Extreme Points**: Resolve global bounds of outboard points through full travel/steer and export directly to XLSX for use in CAD.
* **Dynamic Simulation (`dyn`)**:
    * **Static/Drop Simulation**: Solves for chassis equilibrium (Heave, Pitch, Roll) and simulates the vehicle response to a full-vehicle drop using ODE integrators. 
    * **Shock Dyno**: Simulates damper performance with harmonic excitations to evaluate wheel rates and damping curves.
* **Optimization (`opt`)**:
    * Genetic algorithm (NSGA-II) optimizer to refine hardpoint locations.
    * Minimizes specific kinematic objectives like Bump Steer or Camber Gain.
    * Generates Pareto-optimal solutions and plots performance trade-offs.

---

## Installation

### Prerequisites
* **Python 3.11** installed (see `.python-version`). Other versions may work but are not tested. (Ensure "Add Python to PATH" is checked during installation).

### Setup Steps
1.  **Clone or Download** this repository.
2.  Open a terminal in the project folder (`baja-suspension`).
3.  **Create and Activate a Virtual Environment**:
    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate

    # Mac/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```
4.  **Install Dependencies**:

    For general use:
    ```bash
    pip install -e .
    ```

    For the environment that exactly matches the devs:
    ```bash
    pip install -r requirements-lock.txt
    pip install -e . --no-deps
    ```

    If you add, remove, or intentionally upgrade a dependency, update it in `pyproject.toml` first, then regenerate the lockfile so everyone else picks up the same versions:
    ```bash
    pip install -e . --upgrade
    pip freeze | grep -v "^-e " > requirements-lock.txt
    ```
    Commit the updated `requirements-lock.txt` alongside your `pyproject.toml` change.

---

## Main Entrypoints

SAGE has two interfaces: an interactive Web UI (`app.py`) and a CLI (`main.py`). **Both must be launched from the repo root** — config paths are hardcoded as relative paths (`config/...`).

### 1. Interactive Web UI (`app.py`)
The recommended way to use SAGE, and the only way to run dynamic simulations. Provides a live 3D viewer, 2D charts, dynamic-sim animation, a Pareto-front viewer for optimization runs, and in-browser YAML editing (with save-to-disk) for all three config files plus the hardpoints file.

The 3D viewer carries a **parts tree** (top-right overlay): a checkbox per suspension component — per corner in the full-vehicle sims — to show/hide it. Hidden parts are skipped when the animation re-poses the scene, so hiding what you don't need also speeds up scrubbing.

```bash
python app.py
```
- Opens automatically at `http://localhost:8080` (port is hardcoded, no CLI flags).
- Each tab (Kinematics / Dynamics / Optimization) edits and saves its own config file directly (`config/kin_config.yml`, `config/dyn_config.yml`, `config/opt_config.yml`).

### 2. Command Line Interface (`main.py`)
Useful for batch processing and scripting. Outputs interactive matplotlib plots and a `run.log`/config snapshot under `out/`.

#### Kinematics
Runs a geometric sweep or front-steer/axle-travel/extreme-points analysis (see [Kinematic Simulations](#kinematic-simulations-kin) below).
```bash
python main.py kin --config config/kin_config.yml
```

#### Dynamics — not currently supported via CLI
```bash
python main.py dyn
```

#### Optimization
Runs the NSGA-II multi-objective optimizer (see [Optimization](#optimization-opt) below).
```bash
python main.py opt --kin_config config/kin_config.yml --opt_config config/opt_config.yml
```

All three subcommands accept `--config`/`--kin_config`/`--opt_config`/`--dyn_config` to point at alternate YAML files, defaulting to the files in `config/`.

---

## Kinematic Simulations (`kin`)

Set `SIMULATION` in `config/kin_config.yml` to one of:

| `SIMULATION` value | What it does |
|---|---|
| `travel` | Sweeps bump/droop travel (`TRAVEL.MIN..MAX`) at zero steer, on a single corner. |
| `steer` | Sweeps rack travel (`STEER.MIN..MAX`) at zero bump/droop, on a single corner. |
| `droop_steer` | Sweeps steer while the corner is held at full droop (`TRAVEL.MIN`). |
| `jounce_steer` | Sweeps steer while the corner is held at full bump (`TRAVEL.MAX`). |
| `left_travel` / `right_travel` | Sweeps travel while steer is held at `STEER.MIN` / `STEER.MAX`. |
| `sweep_space` | Full 2D sweep: `TRAVEL` (outer) × `STEER` (inner), `SIM_STEPS × SIM_STEPS` result rows. |
| `extreme` | Resolves the extreme (max jounce/max droop) outboard-point positions at neutral/full-left/full-right steer, for **all four corners** at once, and exports them to `out/kin_sim/<timestamp>/HARDPOINTS_<name>.xlsx` (via the template at `utils/HARDPOINTS_TEMPLATE.xlsx`) for use in CAD. |
| `front_steer` | Sweeps both front corners together across `STEER.MIN..MAX` and computes toe angle / Ackermann percentage, track change, and mechanical trail (ignores `HALF`/`SIDE`). |
| `heave` | Sweeps all four corners together across `TRAVEL.MAX..MIN` (jounce to droop, ride motion), computing per-corner camber/caster/toe/motion-ratio, front/rear track change, wheelbase change, pitch angle, front/rear roll angle, (front, double-A-arm only) roll center Y/Z, and front/rear ground clearance between the chassis-bottom plane (horizontal, 1in below the lowest inboard front lower-A-arm pickup) and the ground plane (through the front- and rear-axle contact-patch centres, parallel to the lateral axis) with the chassis–ground plane angle through travel. The ground/chassis planes, contact patches and front/rear clearance gauges render in the 3D view as a "Ground Clearance" branch of the parts tree (hidden by default). |
| `roll` | Sweeps all four corners: the left side (front-left + rear-left) across `TRAVEL.MIN..MAX` while the right side sweeps `TRAVEL.MAX..MIN` (opposite) — a true full-vehicle roll, same metrics as `heave`. |

`HALF`/`SIDE` (`'front'`/`'rear'`, `'left'`/`'right'`) select which single corner is simulated for the single-corner sweep types (`travel`, `steer`, `droop_steer`, `jounce_steer`, `left_travel`, `right_travel`, `sweep_space`). They're ignored by `front_steer` (always both fronts) and `extreme`/`heave`/`roll` (always all four corners).

`SIM_STEPS` controls sweep resolution (number of samples across the travel/steer range).

`HARDPOINTS` is the filename (without `.yml`) of the vehicle definition under `config/hardpoints/`, e.g. `'2026'` → `config/hardpoints/2026.yml`.

```yaml
# config/kin_config.yml
HARDPOINTS: '2026'
SIM_STEPS:  XXX
SIMULATION: 'travel'      # travel | steer | droop_steer | jounce_steer | left_travel | right_travel | sweep_space | extreme | front_steer | heave | roll

HALF: 'front'             # 'front' or 'rear'
SIDE: 'right'             # 'left' or 'right'

TRAVEL:
  MIN: -XX.X              # [mm] max droop (extension)
  MAX:  XXX.X             # [mm] max bump (compression)
STEER:
  MIN: -XX                # [mm] rack travel
  MAX:  XX                # [mm] rack travel
```

Calculated per-step metrics include camber, caster, toe, kingpin/steering-axis geometry, CV joint angle, motion ratio, and (for `front_steer`) Ackermann percentage, track change, and mechanical trail, and (for `heave`/`roll`) front/rear track width and wheelbase change, pitch/roll angle, and roll center Y/Z.

---

## Dynamic Simulations (`dyn`)

Run from the Web UI's Dynamics tab only (see the CLI warning above). `kin_config.yml` is merged in underneath `dyn_config.yml` — `shock_dyno` in particular needs `HALF`/`SIDE` from the kin config to pick which corner's shock to test.

```yaml
# config/dyn_config.yml
SIMULATION: 'shock_dyno'   # 'static' or 'shock_dyno' — sets the Web UI dropdown's initial value only

SOL_DT: 0.001              # [s] ODE integrator timestep
VIZ_DT: 0.01               # [s] visualization frame interval

# Static / Drop parameters
HOIST_DURATION: 0.5        # [s] time held at hoist height before release
HOIST_HEIGHT: 0.5          # [m] height above static ride height to hoist the CoG to
MAX_SIM_TIME: 60.0         # [s] hard stop if the vehicle never settles

# Shock Dyno parameters
DYNO_STROKE: 50            # [mm] — capped automatically to the shock's physical travel if it exceeds it
DYNO_FREQUENCY: 1.63       # [Hz] — dyno runs exactly one sinusoidal cycle at this frequency, sampled at 200 points
```

- **`static`** (`StaticDrop`): two-phase simulation — the chassis is held ("hoisted") `HOIST_HEIGHT` above static ride height for `HOIST_DURATION` seconds (suspension free-drops under gravity), then released into a full 14-state Euler integration of heave/pitch/roll until the CoG settles (within 2% of static height, sustained) or `MAX_SIM_TIME` elapses. On completion the Web UI exports `out/dyn_sim/<timestamp>/<HARDPOINTS>_NEW_STATIC.yml` — the settled hardpoints (including updated CoG), useful for seeding a more accurate baseline into future kin/opt runs.
- **`shock_dyno`** (`ShockDyno`): drives the selected corner's shock through one full sinusoidal compression/rebound cycle and computes spring force, damper force, and total force vs. displacement/velocity. Exports `out/dyn_sim/<timestamp>/shock_dyno_results.csv`.

---

## Optimization (`opt`)

The optimizer (`SuspensionOptimizer`, NSGA-II via `pymoo`) searches over a subset of hardpoint coordinates to minimize one or more objectives simultaneously, producing a Pareto-optimal set of designs.

```yaml
# config/opt_config.yml

# GLOBAL OPTIMIZER
POP_SIZE: 75             # NSGA-II population size
N_OFFSPRINGS: 75          # Offspring generated per generation
MAX_GEN: 40                # Number of generations to run
M_PROB: 1.0                 # Polynomial mutation probability
M_ETA: 15                    # Polynomial mutation eta (distribution index)
# Note: the optimizer's random seed is hardcoded to 1 in optimization/engine.py — not configurable via YAML.

# OBJECTIVES TO OPTIMIZE — class names from optimization/objectives.py
OBJECTIVES:
  - MinimumBumpSteer
  - ParallelSteer
  - PointToPointCollision

# PARAMETERS TO OPTIMIZE — hardpoint names from the HALF ('front'/'rear', from kin_config.yml)
# section of the active hardpoints file. Ranges are OFFSETS from each point's
# current coordinate, not absolute bounds — e.g. x: [-50.0, 10.0] means the
# optimizer may move that point from (current_x - 50) to (current_x + 10).
FREE_POINTS:
  "tie_rod_inboard":
    x: [-50.0, 10.0]        # [mm] offset range
    y: [-50.0, 50.0]        # [mm] offset range
    z: [-40.0, 45.0]        # [mm] offset range
  "tie_rod_outboard":
    x: [-20.0, 15.0]
    y: [-50.0, 50.0]
    z: [-50.0, 50.0]

# Scenario used to evaluate PointToPointCollision (default: droop_steer if omitted)
COLLISION_SCENARIO: "droop_steer"

# KEEPOUT ZONES: capsule/box volumes extruded along the segment point_a -> point_b,
# checked for interference at every step of COLLISION_SCENARIO's sweep.
#   shape: "cylinder" -> dim1 = radius (mm)
#   shape: "box"      -> dim1, dim2 = side lengths (mm)
# point_a/point_b use the short hardpoint codes from models/hardpoints.py's
# _YAML_MAP (e.g. "tr_ib"=tie_rod_inboard, "tr_ob"=tie_rod_outboard, "uf"=upper_a_arm_front,
# "ur"=upper_a_arm_rear, "ubj"=upper_ball_joint, "lf"/"lr"/"lbj", "s_ib"/"s_ob", "piv_ib"/"piv_ob", "wc").
# At least 2 zones are required for any collision to be detected.
KEEPOUT_ZONES:
  - name: "tierod"
    point_a: "tr_ib"
    point_b: "tr_ob"
    shape: "cylinder"
    dim1: 12.0
  - name: "upper_arm_f"
    point_a: "uf"
    point_b: "ubj"
    shape: "cylinder"
    dim1: 12.0
  - name: "upper_arm_r"
    point_a: "ur"
    point_b: "ubj"
    shape: "cylinder"
    dim1: 12.0

# COLLISION GROUPS (optional): zones in the SAME group are allowed to overlap
# (e.g. two arms that share a ball joint and would otherwise always "collide").
# Every other pair — different groups, or an ungrouped zone vs anything — IS
# checked. Max 10 zones per group. Also controls visualizer coloring.
COLLISION_GROUPS:
  group1:
    - "tierod"
  group2:
    - "upper_arm_f"
    - "upper_arm_r"
```

### Available objectives (`optimization/objectives.py`)
| Objective | Scenario evaluated | Cost |
|---|---|---|
| `MinimumBumpSteer` | `travel` (fixed) | Penalizes peak toe angle and total toe swing across the bump/droop sweep. |
| `ParallelSteer` | `front_steer` (fixed) | RMSE of Ackermann percentage across the steer sweep — pushes toward 100% (parallel) or a custom target curve. |
| `PointToPointCollision` | `COLLISION_SCENARIO` (default `droop_steer`) | Average penetration depth summed across all non-exempt `KEEPOUT_ZONES` pairs over the sweep. |

Every objective receives the full merged `kin_config.yml` + `opt_config.yml` dict, so scenario-specific keys (e.g. `TRAVEL`, `STEER`, `SIM_STEPS`) still apply during optimization.

Run it:
```bash
python main.py opt --kin_config config/kin_config.yml --opt_config config/opt_config.yml
```
or via the Web UI's Optimization tab, which also renders the Pareto front and all evaluated designs live.

---

## Vehicle / Hardpoints File (`config/hardpoints/<name>.yml`)

One file fully defines a vehicle's **left-side** geometry (right side is mirrored automatically) plus mass/wheel/shock properties. Referenced by `HARDPOINTS` in `kin_config.yml`. Top-level key is the vehicle nickname:

```yaml
baja_2026:
  shock_min: XXX.XXX    # [mm] global shock travel limits (front & rear, unless overridden)
  shock_max: XXX.XXX    # [mm]

  wheel_properties:
    radius: X.X          # [mm]
    width: X.X           # [mm]
    stiffness: X.X       # [N/mm]
    damping: X.X         # [N*s/mm]

  mass_properties:
    sprung_mass: XXX.XXX # [kg]
    unsprung_mass: {fl: XX, fr: XX, rl: XX, rr: XX}  # [kg]
    cog: [x, y, z]              # [mm], body frame
    inertia: [[...], [...], [...]]  # [kg*m^2], 3x3 tensor

  front:
    _type: 'DoubleAArm'         # selects the front corner solver
    shock_setup:
      spring_rate: XX.XX        # [N/mm]
      ls_comp: X.X              # [N*s/mm] low-speed compression damping
      hs_comp: X.X              # [N*s/mm] high-speed compression damping
      ls_rebound: X.X           # [N*s/mm] low-speed rebound damping
      hs_rebound: X.X           # [N*s/mm] high-speed rebound damping
      split_vel: X.X            # [mm/s] shaft velocity where low/high-speed damping switches
      preload: X.X              # [mm]
      free_length: XXX          # [mm]
    # hardpoints, all [x, y, z] in mm, body frame (X: longitudinal, Y: lateral toward wheel, Z: up)
    upper_a_arm_front: [...]
    upper_a_arm_rear:  [...]
    lower_a_arm_front: [...]
    lower_a_arm_rear:  [...]
    upper_ball_joint:  [...]
    lower_ball_joint:  [...]
    tie_rod_inboard:   [...]
    tie_rod_outboard:  [...]
    shock_location: "lower"     # or "upper" — where the shock mounts relative to the a-arm
    shock_inboard:  [...]
    shock_outboard: [...]
    pivot_inboard:  [...]       # axle/CV pivot at the upright
    pivot_outboard: [...]
    wheel_center:   [...]

  rear:
    _type: 'SemiTrailingLink'       # selects the rear corner solver
    shock_setup: {...}               # same keys as front
    trailing_link_front: [...]
    upper_camber_link_inboard:  [...]
    upper_camber_link_outboard: [...]
    lower_camber_link_inboard:  [...]
    lower_camber_link_outboard: [...]
    shock_location: upper"
    shock_inboard:  [...]
    shock_outboard: [...]
    pivot_inboard:  [...]
    pivot_outboard: [...]
    wheel_center:   [...]
```

The two corner `_type`s (`DoubleAArm` front, `SemiTrailingLink` rear) are the only ones currently implemented (`models/corners/`); each expects the exact hardpoint key set shown above for that type.

---

## Output / Run Directories

Every `kin`/`dyn`/`opt` run (CLI or Web UI) creates a timestamped directory under `out/<mode>/<YYYYMMDD_HHMMSS>/`, where `<mode>` is `kin_sim`, `dyn_sim`, or `opt`. Each contains at minimum:
- `run.log` — full console output plus debug-only lines, all timestamped.
- A copy (or live-edited snapshot, from the Web UI) of the config file(s) used, and a copy of the hardpoints file.

Plus, mode-specific outputs:
- `kin`, `SIMULATION: extreme` → `HARDPOINTS_<name>.xlsx` (requires `utils/HARDPOINTS_TEMPLATE.xlsx` to exist; skipped with a warning otherwise).
- `dyn`, `static` (Web UI only) → `<HARDPOINTS>_NEW_STATIC.yml`, the settled hardpoints/CoG from the drop.
- `dyn`, `shock_dyno` (Web UI only) → `shock_dyno_results.csv`.

`out/` is gitignored — treat it as scratch output, not something to commit.