# SAGE (Suspension Analysis & Geometry Evaluator)

SAGE is a Python-based simulation suite for designing, analyzing, and optimizing off-road vehicle suspension geometry. This tool provides kinematic analysis, quasi-static dynamic terrain simulation, and hardpoint optimization for **Double A-Arm** (Front) and **Semi-Trailing Link** (Rear) suspension types.

## Key Features

* **Kinematic Analysis (`kin`)**:
    * Sweep suspension through Travel (Bump/Droop) and Steering.
    * Calculate key metrics: Camber, Caster, Toe, and CV Joint angles.
    * Resolve global positions of outboard points through travel/steer and export to SOLIDWORKS.
    * **Ackermann Geometry**: Analyze steering geometry percentages and curves across the full rack travel.
* **Dynamic Simulation (`dyn`)**:
    * **Live 3D Visualization**: Watch the vehicle drive over procedural terrain in real-time.
    * **Quasi-Static Solver**: Solves for chassis equilibrium (Heave, Pitch, Roll) at every step to ensure wheels track the terrain accurately.
    * **Terrain Generation**: Configurable sine-wave profiles to simulate "whoops" or rough terrain.
* **Optimization (`opt`)**:
    * Optimize hardpoint locations to minimize specific objectives like Bump Steer.
    * Support for multi-objective optimization using genetic algorithms.

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

## Usage Modes

The tool is run via `main.py` with a specific command mode: `kin`, `dyn`, or `opt`.

### 1. Kinematic Analysis
Runs a geometric sweep of the suspension. Best for validating hardpoints and checking kinematic curves.

```bash
python main.py kin
```
- **Config File**: `config/kin_config.yml`
- **Output**: Interactive 3D plots and 2D charts for Camber, Toe, and Caster curves

### 2. Dynamic Simulation
Runs a "Live" simulation of the vehicle driving over terrain.

```bash
python main.py dyn
```
- **Config File**: `config/dyn_config.yml` (Controls speed and terrain shape).
- **Config File**: `config/kin_config.yml` (Selects the vehicle hardpoints).
- **Output**: A real-time animated dashboard showing the vehicle geometry and shock travel logs.

### 3. Optimization
Runs the genetic optimizer to refine hardpoint locations.

```bash
python main.py opt
```
- **Config File**: `config/opt_config.yml`.
- **Output**: Pareto-optimal solutions logged to the terminal and visualized via Pareto plots.

## Configuration Guide
`config/kin_config.yml`

Controls the kinematic sweep parameters.

```yml
HARDPOINTS: '2026'      # Filename in config/hardpoints/ (e.g. 2026.yml)
SIM_STEPS:  330         # Resolution of the sweep
SIMULATION: 'travel'    # 'steer', 'travel', 'steer_travel', or 'ackermann'

HALF: 'front'           # 'front' or 'rear'
SIDE: 'right'           # 'left' or 'right'

TRAVEL:
  MIN: -90              # [mm] Max Droop (Extension)
  MAX:  240             # [mm] Max Bump (Compression)
```

`config/dyn_config.yml`

Controls the dynamic terrain simulation parameters.

```yml
VIZ_DT: 0.05            # [s] Time between visual frames
DURATION: 60.0          # [s] Total simulation length
VELOCITY: 10.0          # [m/s] Forward speed

TERRAIN:
  AMPLITUDE: 70.0       # [mm] Height of the bumps
  FREQUENCY: 0.5        # [Hz] Spacing of the bumps
```
