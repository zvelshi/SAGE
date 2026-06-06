# default
import os
from typing import List, Dict, Tuple, Optional

# third-party
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ours
from models.hardpoints import DoubleAArm, SemiTrailingLink
from models.vehicle import Vehicle
from utils.geometry import get_wheel_attitude

def _compute_wheel_cylinder(wc, step, radius, width, n_theta=40):
    """
    Computes the visual mesh for the wheel cylinder.
    """
    wc = np.asarray(wc, dtype=float)
    axis = None

    # Try to get axis from step
    if "wheel_axis" in step:
        a = np.asarray(step["wheel_axis"], dtype=float)
        if np.linalg.norm(a) > 1e-9:
            axis = a / np.linalg.norm(a)

    # Default to Global Y
    if axis is None:
        axis = np.array([0.0, 1.0, 0.0])

    # Basis vectors
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, axis)) > 0.9:
        tmp = np.array([0.0, 0.0, 1.0])
    
    u = tmp - np.dot(tmp, axis) * axis
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)

    c_front = wc - 0.5 * width * axis
    c_back  = wc + 0.5 * width * axis

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    cf = c_front[:, None] + radius * (u[:, None] * cos_t + v[:, None] * sin_t)
    cb = c_back[:, None]  + radius * (u[:, None] * cos_t + v[:, None] * sin_t)
    return cf, cb

class Plotter:
    def __init__(self, title="Simulation Results", save_dir=None):
        self.title = title
        self.figures = []
        self.save_dir = save_dir

    def show(self):
        if self.save_dir:
            for i, fig in enumerate(self.figures):
                # Attempt to generate a clean filename from the plot title
                try:
                    # Get title from first axes
                    raw_title = fig.axes[0].get_title()
                    # Clean string (remove special chars)
                    clean_title = "".join([c if c.isalnum() else "_" for c in raw_title])
                    clean_title = clean_title.strip("_").lower()
                except (IndexError, AttributeError):
                    clean_title = f"figure_{i}"

                # Fallback if title is empty
                if not clean_title:
                    clean_title = f"figure_{i}"

                fname = f"{clean_title}.png"
                save_path = os.path.join(self.save_dir, fname)
                
                try:
                    fig.savefig(save_path, dpi=150)
                    print(f"-> Saved plot: {save_path}")
                except Exception as e:
                    print(f"Error saving plot {fname}: {e}")

        plt.show()

    def _set_axes_equal(self, ax):
        """Forces 3D axes to be equal scale."""
        limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
        origin = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
        ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
        ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

    def plot_3d_corner(self, steps: List[Dict], hp):
        """
        Plots a single corner (Double A-Arm or Semi-Trailing Link).
        Requires the Hardpoints object (hp) to draw static chassis points.
        """
        if not steps: return
        step = steps[-1] # Plot the last state

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f"{self.title} - 3D Geometry")

        if isinstance(hp, DoubleAArm):
            self._render_double_a_arm(ax, step, hp)
        elif isinstance(hp, SemiTrailingLink):
            self._render_semi_trailing(ax, step, hp)
        
        self._set_axes_equal(ax)
        self.figures.append(fig)

    def plot_3d_ackermann(self, steps: List[Dict], vehicle: Vehicle):
        """
        Plots the full front end (Left and Right Double A-Arms).
        """
        if not steps: return
        step = steps[-1]

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Ackermann Geometry (Full Front)")

        # Left Corner
        if 'left' in step:
            self._render_double_a_arm(ax, step['left'], vehicle.front_left.hardpoints, color_main='blue')
        
        # Right Corner
        if 'right' in step:
            self._render_double_a_arm(ax, step['right'], vehicle.front_right.hardpoints, color_main='red')

        self._set_axes_equal(ax)
        self.figures.append(fig)

    def _render_double_a_arm(self, ax, step, hp: DoubleAArm, color_main='black'):
        # Helper to plot line
        def line(p1, p2, c=color_main, lw=1.5):
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=c, lw=lw)

        # A-Arms (Chassis -> Upright)
        line(hp.uf, step['ubj']) # Upper Front
        line(hp.ur, step['ubj']) # Upper Rear
        line(hp.lf, step['lbj']) # Lower Front
        line(hp.lr, step['lbj']) # Lower Rear

        # Upright
        line(step['lbj'], step['ubj'], c=color_main, lw=2)

        # Steering (Tie Rod)
        line(step['tr_ib'], step['tr_ob'], c='green', lw=2)

        # Shock
        line(hp.s_ib, step['s_ob'], c='black', lw=2)

        # Axle
        if 'piv_ob' in step:
            line(hp.piv_ib, step['piv_ob'], c='red', lw=2)
            line(step['piv_ob'], step['wc'], c='red', lw=2)

        # Wheel
        try:
            cf, cb = _compute_wheel_cylinder(step['wc'], step, hp.wr, hp.ww)
            ax.plot(cf[0], cf[1], cf[2], color='grey', alpha=0.3)
            ax.plot(cb[0], cb[1], cb[2], color='grey', alpha=0.3)
        except:
            pass

        # Points
        all_pts = [hp.uf, hp.ur, hp.lf, hp.lr, hp.s_ib, step['ubj'], step['lbj'], step['wc']]
        pts = np.array(all_pts)
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], c='k', s=20)

    def _render_semi_trailing(self, ax, step, hp: SemiTrailingLink, color_main='black'):
        def line(p1, p2, c=color_main, lw=1.5):
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=c, lw=lw)

        # Trailing Link (Front Pivot -> Camber Links Outboard)
        line(hp.tl_f, step['ucl_ob']) 
        line(hp.tl_f, step['lcl_ob'])

        # Camber Links
        line(hp.ucl_ib, step['ucl_ob'])
        line(hp.lcl_ib, step['lcl_ob'])

        # Shock
        line(hp.s_ib, step['s_ob'])

        # Axle
        if 'piv_ob' in step:
            line(hp.piv_ib, step['piv_ob'], c='red', lw=2) # Axle Shaft
            line(step['piv_ob'], step['wc'], c='red', lw=2) # Stub Axle

        # Wheel
        try:
            cf, cb = _compute_wheel_cylinder(step['wc'], step, hp.wr, hp.ww)
            ax.plot(cf[0], cf[1], cf[2], color='grey', alpha=0.3)
            ax.plot(cb[0], cb[1], cb[2], color='grey', alpha=0.3)
        except: pass
        
        # Points
        pts = [hp.tl_f, hp.ucl_ib, hp.lcl_ib, hp.s_ib, step['wc']]
        pts = np.array(pts)
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], c='k', s=20)

    def plot_kinematics_curves(self, steps: List[Dict]):
        """
        Plots the 5 curves: Camber, Caster, Toe, Plunge, Axle Angles.
        """
        if not steps: return

        # Prepare Data
        attitudes = [get_wheel_attitude(s) for s in steps]
        
        # Extract Axle Data
        plunge = []
        angle_ib = []
        angle_ob = []
        for s in steps:
            if 'axle_data' in s:
                plunge.append(s['axle_data'].get('plunge_mm', 0))
                angle_ib.append(s['axle_data'].get('angle_ib_deg', 0))
                angle_ob.append(s['axle_data'].get('angle_ob_deg', 0))
            else:
                plunge.append(0); angle_ib.append(0); angle_ob.append(0)

        # X-Axis Definition
        if 'x_val' in steps[0]:
            xs = [s['x_val'] for s in steps]
            xlabel = steps[0].get('x_label', "Input")
        elif 'travel_mm' in steps[0]:
            xs = [s['travel_mm'] for s in steps]
            xlabel = "Shock Travel [mm]"
        elif 'steer_mm' in steps[0]:
            xs = [s['steer_mm'] for s in steps]
            xlabel = "Rack Travel [mm]"
        else:
            xs = range(len(steps))
            xlabel = "Step"

        # Setup Figure (2 Rows, 3 Cols)
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(f"{self.title} - Kinematics Dashboard")
        
        # Row 1: Wheel Attitude
        ax_camber = fig.add_subplot(2, 3, 1)
        ax_caster = fig.add_subplot(2, 3, 2)
        ax_toe    = fig.add_subplot(2, 3, 3)
        
        # Row 2: Axle Stats
        ax_plunge = fig.add_subplot(2, 3, 4)
        ax_angles = fig.add_subplot(2, 3, 5)

        # Plot Camber
        ax_camber.plot(xs, [a['camber'] for a in attitudes], '-o', markersize=3)
        ax_camber.set_title("Camber [deg]")
        ax_camber.set_xlabel(xlabel)
        ax_camber.grid(True)

        # Plot Caster
        ax_caster.plot(xs, [a['caster'] for a in attitudes], '-o', markersize=3)
        ax_caster.set_title("Caster [deg]")
        ax_caster.set_xlabel(xlabel)
        ax_caster.grid(True)

        # Plot Toe
        ax_toe.plot(xs, [a['toe'] for a in attitudes], '-o', markersize=3)
        ax_toe.set_title("Toe [deg]")
        ax_toe.set_xlabel(xlabel)
        ax_toe.grid(True)

        # Plot Plunge
        ax_plunge.plot(xs, plunge, '-o', color='orange', markersize=3)
        ax_plunge.set_title("Axle Plunge [mm]")
        ax_plunge.set_xlabel(xlabel)
        ax_plunge.grid(True)

        # Plot Axle Angles (IB vs OB)
        ax_angles.plot(xs, angle_ib, label="Inboard", color='red')
        ax_angles.plot(xs, angle_ob, label="Outboard", color='blue', linestyle='--')
        ax_angles.set_title("CV Joint Angles [deg]")
        ax_angles.set_xlabel(xlabel)
        ax_angles.legend()
        ax_angles.grid(True)

        self.figures.append(fig)

    def plot_ackermann_curve(self, steps: List[Dict]):
        """Plot for Ackermann % vs Steer"""
        if not steps: return
        
        xs = [s['input'] for s in steps]
        ys = [s['ackermann_pct'] for s in steps]

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(xs, ys, color='purple', lw=2)
        ax.set_title("Ackermann Percentage")
        ax.set_xlabel("Steering Rack [mm]")
        ax.set_ylabel("Ackermann [%]")
        ax.grid(True)
        self.figures.append(fig)    

class ParetoPlotter:
    """
    Visualizes Optimization Results.
    """
    def __init__(self, optimizer, save_dir=None):
        self.opt = optimizer
        self.obj_names = [o.name for o in optimizer.objectives]
        self.save_dir = save_dir # Path to save images

    def _save_plot(self, name_suffix):
        """Helper to save the current active figure"""
        if self.save_dir:
            fname = f"pareto_{name_suffix}.png"
            path = os.path.join(self.save_dir, fname)
            plt.savefig(path, dpi=200)
            print(f"-> Saved plot: {path}")

    def plot_front(self, res, history: Optional[List] = None):
        """
        Plots the Pareto Front and optionally the full search history.
        
        Args:
            res: The pymoo Result object.
            history: Optional list of history objects (if save_history=True was used).
        """
        F = res.F
        if F is None:
            print("No results to plot.")
            return

        if F.ndim == 1:
            F = F.reshape(-1, 1)
            
        valid_mask = np.all(F <= 1e2, axis=1)
        F = F[valid_mask]

        if len(F) == 0:
            print("No solutions found with cost <= 1e2.")
            return

        F_cloud = None
        if history:
            cloud_list = []
            for algo in history:
                if hasattr(algo, 'pop'):
                    pop_F = algo.pop.get("F")
                    if pop_F is not None:
                        if pop_F.ndim == 1:
                            pop_F = pop_F.reshape(-1, 1)
                        mask = np.all(pop_F <= 1e2, axis=1)
                        valid_pop = pop_F[mask]
                        
                        if len(valid_pop) > 0:
                            cloud_list.append(valid_pop)
            
            if cloud_list:
                F_cloud = np.vstack(cloud_list)

        n_obj = F.shape[1]

        if n_obj == 1:
            self._plot_1d_front(F, F_cloud)
        elif n_obj == 2:
            self._plot_2d_front(F, F_cloud)
        elif n_obj >= 3:
            self._plot_multidim_front(F, F_cloud)
        self._save_plot("front")

    def _plot_1d_front(self, F, F_cloud=None):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if F_cloud is not None:
            costs_cloud = F_cloud.flatten()
            ax.hist(costs_cloud, bins=30, color='lightgray', alpha=0.5, label='All Designs')

        costs = F.flatten()
        ax.hist(costs, bins=20, color='skyblue', edgecolor='black', alpha=0.7, label='Final Gen')
        
        best_val = np.min(costs)
        ax.axvline(best_val, color='red', linestyle='--', linewidth=2, label=f'Best: {best_val:.4f}')
        
        ax.set_title(f"Objective Distribution: {self.obj_names[0]}")
        ax.set_xlabel(f"Cost ({self.obj_names[0]})")
        ax.set_ylabel("Count")
        ax.legend()
        plt.show()

    def _plot_2d_front(self, F, F_cloud=None):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if F_cloud is not None:
            ax.scatter(F_cloud[:, 0], F_cloud[:, 1], s=10, c='lightgray', alpha=0.3, label="Tested Designs")
        ax.scatter(F[:, 0], F[:, 1], s=40, facecolors='none', edgecolors='blue', lw=1.5, label="Pareto Front")
        
        F_norm = (F - F.min(axis=0)) / (F.ptp(axis=0) + 1e-9)
        dist = np.linalg.norm(F_norm, axis=1)
        best_idx = np.argmin(dist)
        
        ax.scatter(F[best_idx, 0], F[best_idx, 1], c='red', s=100, marker='*', label="Best Balance")

        ax.set_title("Pareto Front vs. Search Space")
        ax.set_xlabel(self.obj_names[0])
        ax.set_ylabel(self.obj_names[1])
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        plt.show()

    def _plot_multidim_front(self, F, F_cloud=None):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        x_idx, y_idx, z_idx = 0, 1, 2

        if F_cloud is not None:
             ax.scatter(F_cloud[:, x_idx], F_cloud[:, y_idx], F_cloud[:, z_idx], s=5, c='lightgray', alpha=0.2)
        sc = ax.scatter(F[:, x_idx], F[:, y_idx], F[:, z_idx], c=F[:, z_idx], cmap='viridis', s=40, depthshade=False)
        
        ax.set_title("Pareto Front (3D)")
        ax.set_xlabel(self.obj_names[x_idx])
        ax.set_ylabel(self.obj_names[y_idx])
        ax.set_zlabel(self.obj_names[z_idx])
        fig.colorbar(sc, label=self.obj_names[z_idx])
        plt.show()

class DynamicPlotter(Plotter):
    """
    Creates an animated dashboard showing the suspension geometry, kinematics, 
    and chassis velocity tracking over time.
    """
    def animate(self, steps: List[Dict], vehicle, config):
        if not steps: return

        interval = config['VIZ_DT'] * 1000 # Convert to ms for matplotlib FuncAnimation
        
        fig = plt.figure(figsize=(18, 14)) 
        # Increased to 4 rows to include the velocity controller plot
        gs = fig.add_gridspec(4, 2, width_ratios=[2, 1]) 
        
        ax3d = fig.add_subplot(gs[:, 0], projection='3d')
        ax_camber = fig.add_subplot(gs[0, 1])
        ax_caster = fig.add_subplot(gs[1, 1])
        ax_toe    = fig.add_subplot(gs[2, 1])
        ax_vel    = fig.add_subplot(gs[3, 1])

        corners = ['fl', 'fr', 'rl', 'rr']
        styles  = {'fl': 'b-', 'fr': 'r-', 'rl': 'b--', 'rr': 'r--'}
        labels  = {'fl': 'FL', 'fr': 'FR', 'rl': 'RL', 'rr': 'RR'}
        data_store = {c: {'camber': [], 'caster': [], 'toe': [], 'time': []} for c in corners}
        
        # Velocity Tracking Data
        times = []
        actual_vels = []
        target_vels = []
        v_max = config.get('VELOCITY', 5.0)
        t_ramp = config.get('RAMP_DURATION', 1.0)

        for s in steps:
            t = s['time']
            times.append(t)
            
            # Record Velocity (taking absolute value so it plots positively for easier reading)
            vx_ms = abs(s.get('chassis_vx', 0.0)) / 1000.0
            actual_vels.append(vx_ms)
            target_vels.append((v_max / t_ramp) * t if t < t_ramp else v_max)

            for c_name in corners:
                if c_name in s['corners']:
                    att = get_wheel_attitude(s['corners'][c_name])
                    data_store[c_name]['camber'].append(att['camber'])
                    data_store[c_name]['caster'].append(att['caster'])
                    data_store[c_name]['toe'].append(att['toe'])
                    data_store[c_name]['time'].append(t)
                else:
                    if len(data_store[c_name]['time']) > 0:
                        data_store[c_name]['camber'].append(np.nan)
                        data_store[c_name]['caster'].append(np.nan)
                        data_store[c_name]['toe'].append(np.nan)
                        data_store[c_name]['time'].append(t)

        lines_camber = {}
        lines_caster = {}
        lines_toe = {}

        for c_name in corners:
            ln_cam, = ax_camber.plot([], [], styles[c_name], lw=1.5, label=labels[c_name])
            ln_toe, = ax_toe.plot([], [], styles[c_name], lw=1.5)
            lines_camber[c_name] = ln_cam
            lines_toe[c_name] = ln_toe

            if 'f' in c_name:
                ln_cas, = ax_caster.plot([], [], styles[c_name], lw=1.5)
                lines_caster[c_name] = ln_cas

        # Setup Velocity Plot Lines
        ln_v_targ, = ax_vel.plot([], [], 'k--', lw=2, label="Target Vel")
        ln_v_act,  = ax_vel.plot([], [], 'g-', lw=2, label="Actual Vel")

        ax_camber.set_title("Camber [deg]"); ax_camber.grid(True); ax_camber.legend(loc='upper right', ncol=2, fontsize='small')
        ax_caster.set_title("Front Caster [deg]"); ax_caster.grid(True)
        ax_toe.set_title("Toe [deg]"); ax_toe.grid(True)
        ax_vel.set_title("Chassis Velocity Tracking [m/s]"); ax_vel.set_xlabel("Time [s]"); ax_vel.grid(True); ax_vel.legend(loc='lower right')

        # Set Limits for Kinematics
        for ax, key, subset in zip([ax_camber, ax_caster, ax_toe], ['camber', 'caster', 'toe'], [corners, ['fl','fr'], corners]):
            all_vals = []
            for c in subset: all_vals.extend(data_store[c][key])
            all_vals = [v for v in all_vals if not np.isnan(v)]
            if all_vals:
                ax.set_ylim(min(all_vals)-1, max(all_vals)+1)
                ax.set_xlim(0, steps[-1]['time'])
                
        # Set Limits for Velocity
        ax_vel.set_xlim(0, steps[-1]['time'])
        ax_vel.set_ylim(0, v_max + 1.0)

        def draw_terrain(ax, current_x_mm):
            """Draws a 3D terrain surface blending from Left (0 phase) to Right (pi phase)"""
            x_range_mm = np.linspace(current_x_mm - 3000, current_x_mm + 3000, 60)
            y_range_mm = np.linspace(-1200, 1200, 20)
            X, Y = np.meshgrid(x_range_mm, y_range_mm)
            
            # Smoothly interpolate phase shift across the vehicle track width 
            # Assuming ~1500mm track width. Left side (+750Y) gets 0 phase, Right side (-750Y) gets Pi phase.
            phase_shift = np.clip((750 - Y) / 1500.0 * np.pi, 0, np.pi)
            
            t_cfg = config['TERRAIN']
            x_m = X / 1000.0
            Z = t_cfg['AMPLITUDE'] * np.sin((2 * np.pi * t_cfg['FREQUENCY'] * x_m) + phase_shift)
            
            # Using plot_surface instead of wireframe makes the whoops much easier to visualize
            ax.plot_surface(X, Y, Z, color='green', alpha=0.3, edgecolor='none')

        # def draw_terrain(ax, current_x_mm):
        #     """Draws discrete, high-resolution staggered whoops without surface tearing"""
        #     x_range_mm = np.linspace(current_x_mm - 2000, current_x_mm + 4000, 600)
            
        #     whoop_height_mm = 11.0 * 25.4
        #     whoop_spacing_mm = 7.5 * 12.0 * 25.4
        #     whoop_width_mm = 18.0 * 25.4

        #     # --- THE MANUAL OFFSET DIAL ---
        #     # Tweak this number (in mm) to slide the visual terrain forward/backward
        #     visual_offset_mm = 1000.0 

        #     # --- LEFT SIDE TERRAIN ---
        #     y_left = np.linspace(0, 1200, 15)
        #     X_L, Y_L = np.meshgrid(x_range_mm, y_left)
            
        #     # Apply the offset before the modulo
        #     local_x_L = (X_L + visual_offset_mm) % whoop_spacing_mm
            
        #     Z_L = np.where(
        #         local_x_L < whoop_width_mm,
        #         whoop_height_mm * np.sin((local_x_L / whoop_width_mm) * np.pi),
        #         0.0
        #     )
        #     ax.plot_surface(X_L, Y_L, Z_L, color='sandybrown', alpha=0.6, edgecolor='none')

        #     # --- RIGHT SIDE TERRAIN ---
        #     y_right = np.linspace(-1200, 0, 15)
        #     X_R, Y_R = np.meshgrid(x_range_mm, y_right)
            
        #     # Apply the half-spacing stagger AND the manual offset
        #     local_x_R = (X_R + (whoop_spacing_mm / 2.0) + visual_offset_mm) % whoop_spacing_mm
            
        #     Z_R = np.where(
        #         local_x_R < whoop_width_mm,
        #         whoop_height_mm * np.sin((local_x_R / whoop_width_mm) * np.pi),
        #         0.0
        #     )
        #     ax.plot_surface(X_R, Y_R, Z_R, color='peru', alpha=0.6, edgecolor='none')

        def update(frame):
            step = steps[frame]
            t = step['time']
            car_x = step['x_pos']

            # Update 2D Plot Lines
            for c_name in corners:
                ts = data_store[c_name]['time']
                if frame < len(ts):
                    lines_camber[c_name].set_data(ts[:frame], data_store[c_name]['camber'][:frame])
                    lines_toe[c_name].set_data(ts[:frame], data_store[c_name]['toe'][:frame])
                    if 'f' in c_name:
                        lines_caster[c_name].set_data(ts[:frame], data_store[c_name]['caster'][:frame])

            # Update Velocity Plot
            ln_v_targ.set_data(times[:frame], target_vels[:frame])
            ln_v_act.set_data(times[:frame], actual_vels[:frame])

            ax3d.cla()
            draw_terrain(ax3d, car_x)
            
            # --- 6-DOF KINEMATICS ---
            chassis_z = step['chassis_z']
            pitch_angle = step['pitch_angle']
            roll_angle = step.get('roll_angle', 0.0)
            yaw_angle = step.get('yaw_angle', 0.0)
            
            cp, sp = np.cos(pitch_angle), np.sin(pitch_angle)
            cr, sr = np.cos(roll_angle), np.sin(roll_angle)
            
            R_pitch = np.array([
                [cp,  0, sp],
                [0,   1,  0],
                [-sp, 0, cp]
            ])
            R_roll = np.array([
                [1,  0,   0],
                [0, cr, -sr],
                [0, sr,  cr]
            ])
            R_rot = R_pitch @ R_roll 

            # --- CALCULATE TRUE GLOBAL CoG ---
            static_cog = np.array(vehicle.cog)
            global_cog = np.array([
                car_x + static_cog[0], 
                step.get('y_pos', 0.0) + static_cog[1], 
                chassis_z + static_cog[2]
            ])

            # Draw the Center of Gravity
            ax3d.scatter(
                [global_cog[0]], [global_cog[1]], [global_cog[2]], 
                color='black', marker='D', s=180, label="CoG", zorder=100
            )

            # --- LIVE TELEMETRY OVERLAY ---
            r_deg, p_deg, y_deg = np.degrees(roll_angle), np.degrees(pitch_angle), np.degrees(yaw_angle)
            title_text = (f"Sim Time: {t:.2f}s | Target Vel: {target_vels[frame]:.2f} m/s\n"
                          f"XYZ: [{global_cog[0]:.1f}, {global_cog[1]:.1f}, {global_cog[2]:.1f}] mm\n"
                          f"RPY: [{r_deg:.2f}°, {p_deg:.2f}°, {y_deg:.2f}°]")
            ax3d.set_title(title_text, fontsize=10, loc='left')

            # --- APPLY ROTATION AROUND CoG ---
            for corner_name, corner_data in step['corners'].items():
                is_front = 'f' in corner_name
                is_left = 'l' in corner_name
                
                hp_static = vehicle.front_left.hardpoints if is_front else vehicle.rear_left.hardpoints
                if not is_left:
                    hp_static = vehicle.front_right.hardpoints if is_front else vehicle.rear_right.hardpoints

                shifted_step = {}
                for k, v in corner_data.items():
                    if k == 'wheel_axis':
                        shifted_step[k] = R_rot @ v
                    elif isinstance(v, np.ndarray) and v.shape == (3,):
                        # Shift to CoG, rotate, shift to global
                        delta = v - static_cog
                        shifted_step[k] = (R_rot @ delta) + global_cog
                    else:
                        shifted_step[k] = v

                class ShiftedHP: pass
                shp = ShiftedHP()
                for k, v in hp_static.__dict__.items():
                    if isinstance(v, np.ndarray) and v.shape == (3,):
                        # Shift to CoG, rotate, shift to global
                        delta = v - static_cog
                        setattr(shp, k, (R_rot @ delta) + global_cog)
                    else:
                        setattr(shp, k, v)

                if is_front:
                    self._render_double_a_arm(ax3d, shifted_step, shp, color_main='blue' if is_left else 'red')
                else:
                    self._render_semi_trailing(ax3d, shifted_step, shp, color_main='blue' if is_left else 'red')

            # Shifted camera limits: Because we drive in -X, we want more camera room "behind" the car
            ax3d.set_xlim(car_x - 750, car_x + 2250)
            ax3d.set_ylim(-1000, 1000)
            ax3d.set_zlim(0, 1500) 
            ax3d.set_box_aspect([3, 2, 1.5])
        
        anim = FuncAnimation(fig, update, frames=len(steps), interval=interval)
        plt.show()