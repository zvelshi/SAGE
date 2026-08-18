# SAGE (Suspension Analysis & Geometry Evaluator)

SAGE is a Python-based simulation suite for designing, analyzing, and optimizing off-road vehicle suspension geometry. This tool provides kinematic analysis, quasi-static dynamic terrain simulation, and hardpoint optimization for **Double A-Arm** (Front) and **Semi-Trailing Link** (Rear) suspension types.

## Key Capabilities

* **Kinematic Analysis (`kin`)**:
    * **Sweep Suspension**: Travel (Bump/Droop), Steering, and combined interactions (Droop/Steer, Jounce/Steer).
    * **Calculated Metrics**: Camber, Caster, Toe, and CV Joint angles.
    * **Ackermann Geometry**: Analyze steering geometry percentages and curves across full rack travel.
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
* **Python 3.8+** installed. (Ensure "Add Python to PATH" is checked during installation).

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
    ```bash
    pip install -e .
    ```

---

## Main Entrypoints

SAGE has two main interfaces for interacting with the simulation engine: an interactive Web UI and a CLI.

### 1. Interactive Web UI (`app.py`)
The recommended way to use SAGE is through the local web application. It provides real-time 3D visualization, interactive charts, and on-the-fly configuration editing.

```bash
python app.py
```
- Automatically opens in your browser at `http://localhost:8080`.
- Includes a live 3D viewer, 2D kinematic charts, and dynamic simulation animations.
- Allows live editing of YAML configurations (hardpoints and simulation parameters) directly in the browser.

### 2. Command Line Interface (`main.py`)
The CLI is useful for batch processing, scripting, and headless execution. It outputs 3D/2D matplotlib plots and console logs.

#### Kinematics
Runs a geometric sweep of the suspension.
```bash
python main.py kin
```

#### Dynamics
Runs a dynamic simulation (Drop or Shock Dyno).
```bash
python main.py dyn
```

#### Optimization
Runs the genetic optimizer.
```bash
python main.py opt
```

---

## Configuration Guide

Simulation parameters are controlled via YAML files located in the `config/` directory. The vehicle's hardpoints are defined in `config/hardpoints/`.

### Kinematics (`config/kin_config.yml`)
Controls the kinematic sweep parameters.
```yaml
HARDPOINTS: '2026'              # Filename in config/hardpoints/ (e.g. 2026.yml)
SIM_STEPS:  330                 # Resolution of the sweep
SIMULATION: 'travel'            # 'steer', 'travel', 'droop_steer', 'jounce_steer', "left_travel", "right_travel", 'sweep_space', 'extreme', or 'ackermann'

HALF: 'front'                   # 'front' or 'rear'
SIDE: 'right'                   # 'left' or 'right'

TRAVEL:
  MIN: -90                      # [mm] Max Droop (Extension)
  MAX:  240                     # [mm] Max Bump (Compression)
```

### Dynamics (`config/dyn_config.yml`)
Controls the dynamic terrain simulation parameters.
```yaml
SIMULATION: 'shock_dyno' # 'static' (drop), 'shock_dyno'

SOL_DT: 0.001            # [s] ODE integrator timestep
VIZ_DT: 0.01             # [s] visualization frame interval

# Static/Drop Parameters
HOIST_DURATION: 0.5      # [s] time to hold CoG at hoist height before release
HOIST_HEIGHT: 0.5        # [m] height above static CoG to hoist to
MAX_SIM_TIME: 3.0        # [s] hard stop for the drop phase

# Shock Dyno Parameters
DYNO_STROKE: 50          # [mm]
DYNO_FREQUENCY: 1.63     # [Hz]
```

### Optimizer (`config/opt_config.yml`)
Controls the optimization parameters.
```yaml
# GLOBAL OPTIMIZER
POP_SIZE: 35             # Size of the population
N_OFFSPRINGS: 25         # Number of off
MAX_GEN: 25              # Maximum number of generations
M_PROB: 1.0              # Mutation probability
M_ETA: 15                # Mutation eta (polynomial mutation)

# OPTIMIZATION TO RUN
OBJECTIVES:              # Objectives to optimize
  - MinimumBumpSteer
  - ParallelSteer
  - NoCollision

# PARAMETERS TO OPTIMIZE
FREE_POINTS:             # Parameters to optimize
  "point_name_1":        # Name of the hardpoint to optimize (must match the hardpoint name in the hardpoint YAML file)
    x: [MIN, MAX]        # [mm] Limits for x-coordinate
    y: [MIN, MAX]        # [mm] Limits for y-coordinate
    z: [MIN, MAX]        # [mm] Limits for z-coordinate
  "point_name_2":          # Name of the hardpoint to optimize (must match the hardpoint name in the hardpoint YAML file)
    x: [MIN, MAX]        # [mm] Limits for x-coordinate
    y: [MIN, MAX]        # [mm] Limits for y-coordinate
    z: [MIN, MAX]        # [mm] Limits for z-coordinate
```