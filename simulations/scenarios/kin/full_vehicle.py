# default
from typing import List, Dict
from math import atan2, degrees

# third-party
import numpy as np

# ours
from simulations.scenarios.base import Scenario
from simulations.solvers import SingleCornerSolver
from utils.config import SweepConfig
from utils.logging_setup import get_logger
from utils.geometry import roll_center_yz, get_contact_patch
from utils.spatial import Point, Line, Plane

log = get_logger(__name__)

FULL_VEHICLE_TYPES = {"roll", "heave"}

class FullVehicleScenario(Scenario):
    """Sweeps all four corners together -- 'heave' moves all four the same direction
    (jounce -> droop), 'roll' moves the left and right sides oppositely (each side's
    front/rear corners moving together). Driven by wheel-center vertical travel
    (bump_z), not shock travel: front and rear have different motion ratios, so an
    equal shock travel_mm on every corner does NOT move the wheels by equal amounts
    and the 3D view looks unsynced. bump_z shifts each wheel center by the same
    absolute Z regardless of that corner's motion ratio, keeping the visualization
    genuinely synced. The bump_z range is calibrated from how far the front-left wheel
    actually travels across TRAVEL.MIN..MAX, so it stays in the same ballpark as the
    other kin scenarios' travel range."""

    def __init__(self, vehicle, config: SweepConfig, mode: str, roll_center: bool = True,
                 bump_z_limits: dict | None = None):
        self.config = config
        self.mode = mode
        self.vehicle = vehicle
        # Roll centers need 8 extra perturbed solves per step; the optimizer
        # turns this off when no objective reads a roll-center field.
        self.roll_center = roll_center
        self.fl_solver = SingleCornerSolver(vehicle, corner_id=[0, 0])
        self.fr_solver = SingleCornerSolver(vehicle, corner_id=[1, 0])
        self.rl_solver = SingleCornerSolver(vehicle, corner_id=[0, 1])
        self.rr_solver = SingleCornerSolver(vehicle, corner_id=[1, 1])

        fl_hp, fr_hp = vehicle.front_left.hardpoints, vehicle.front_right.hardpoints
        rl_hp, rr_hp = vehicle.rear_left.hardpoints, vehicle.rear_right.hardpoints
        self.wr_front, self.wr_rear = fl_hp.wr, rl_hp.wr
        self.static_front_track = abs(fl_hp.wc[1] - fr_hp.wc[1])
        self.static_rear_track  = abs(rl_hp.wc[1] - rr_hp.wc[1])
        self.static_wheelbase = (fl_hp.wc[0] + fr_hp.wc[0]) / 2.0 - (rl_hp.wc[0] + rr_hp.wc[0]) / 2.0

        # bump_z sweep envelope: start from the mechanically reachable range each
        # axle has between its shock limits (precomputed on the Vehicle), intersect
        # the two axles so the synced whole-car motion never bottoms a shock, then
        # clamp by the config TRAVEL range (shock-mm, converted to wheel-mm per axle
        # with a quick solve -- if TRAVEL asks for more than is mechanically
        # possible the solve returns None and the mechanical limit stands).
        # per-run calibration (from the base geometry) when the optimizer supplies
        # it, otherwise this vehicle's own
        limits = bump_z_limits if bump_z_limits is not None else vehicle.bump_z_limits
        f_lo, f_hi = limits["front"]
        r_lo, r_hi = limits["rear"]
        lo_candidates = [max(f_lo, r_lo)]
        hi_candidates = [min(f_hi, r_hi)]
        for scs, corner, hp in ((self.fl_solver, vehicle.front_left, fl_hp),
                                (self.rl_solver, vehicle.rear_left, rl_hp)):
            at_min = scs.solve(steer_mm=0.0, travel_mm=config.travel.min)
            at_max = scs.solve(steer_mm=0.0, travel_mm=config.travel.max)
            if at_min:
                lo_candidates.append(at_min['wc'][2] - hp.wc[2])
            if at_max:
                hi_candidates.append(at_max['wc'][2] - hp.wc[2])
            corner.solver.reset()
        self.bump_min = max(lo_candidates)
        self.bump_max = min(hi_candidates)
        log.debug("%s bump_z envelope: %.2f .. %.2f mm", mode, self.bump_min, self.bump_max)

        # Roll center height is conventionally quoted above the ground (the tire
        # contact patch at static ride height), not above the hardpoints' raw Z=0 --
        # those don't coincide here (Z=0 sits ~15mm below the static contact patch).
        fl0 = self.fl_solver.solve(steer_mm=0.0, bump_z=0.0)
        rl0 = self.rl_solver.solve(steer_mm=0.0, bump_z=0.0)
        self.front_ground_z = get_contact_patch(fl0, self.wr_front)[2] if fl0 else 0.0
        self.rear_ground_z  = get_contact_patch(rl0, self.wr_rear)[2] if rl0 else 0.0

        # Chassis bottom plane: a whole-vehicle property (Plane) built once when the
        # Vehicle is created -- horizontal, 1in below the lowest inboard front
        # lower a arm point
        self.chassis_plane = vehicle.chassis_bottom_plane
        self.cog_point = vehicle.cog_point

    def _build_step(self, t, fl, fr, rl, rr, perturbed) -> Dict:
        front_track = abs(fl['wc'][1] - fr['wc'][1])
        rear_track  = abs(rl['wc'][1] - rr['wc'][1])
        front_x = (fl['wc'][0] + fr['wc'][0]) / 2.0
        rear_x  = (rl['wc'][0] + rr['wc'][0]) / 2.0
        wheelbase = front_x - rear_x
        front_avg_z = (fl['wc'][2] + fr['wc'][2]) / 2.0
        rear_avg_z  = (rl['wc'][2] + rr['wc'][2]) / 2.0
        # abs(wheelbase): this repo's X axis runs front->rear (rear corners sit at a
        # larger X than front), so the signed front_x - rear_x is negative here -- using
        # it directly in atan2 would flip the angle into the far quadrant near +-180 deg.
        pitch_angle_deg = degrees(atan2(rear_avg_z - front_avg_z, abs(wheelbase))) if wheelbase else 0.0
        front_roll_angle_deg = degrees(atan2(fl['wc'][2] - fr['wc'][2], front_track)) if front_track else 0.0
        rear_roll_angle_deg  = degrees(atan2(rl['wc'][2] - rr['wc'][2], rear_track)) if rear_track else 0.0
        fl_p, fl_m, fr_p, fr_m, rl_p, rl_m, rr_p, rr_m = perturbed
        front_rc = roll_center_yz(fl, fl_p, fl_m, fr, fr_p, fr_m, self.wr_front)
        rear_rc  = roll_center_yz(rl, rl_p, rl_m, rr, rr_p, rr_m, self.wr_rear)

        gc = self._ground_clearance(fl, fr, rl, rr, front_x, rear_x)

        return {
            "input": t, "fl": fl, "fr": fr, "rl": rl, "rr": rr,
            "front_track_mm": front_track, "rear_track_mm": rear_track,
            "front_track_change_mm": front_track - self.static_front_track,
            "rear_track_change_mm": rear_track - self.static_rear_track,
            "wheelbase_mm": wheelbase, "wheelbase_change_mm": wheelbase - self.static_wheelbase,
            "pitch_angle_deg": pitch_angle_deg,
            "front_roll_angle_deg": front_roll_angle_deg, "rear_roll_angle_deg": rear_roll_angle_deg,
            "front_roll_center_y_mm": front_rc[0] if front_rc is not None else None,
            "front_roll_center_z_mm": (front_rc[1] - self.front_ground_z) if front_rc is not None else None,
            "rear_roll_center_y_mm": rear_rc[0] if rear_rc is not None else None,
            "rear_roll_center_z_mm": (rear_rc[1] - self.rear_ground_z) if rear_rc is not None else None,
            **gc,
        }

    def _ground_clearance(self, fl, fr, rl, rr, front_x, rear_x) -> Dict:
        """Front/rear distance between the fixed chassis-bottom plane and the
        four-contact-patch ground plane, plus the sagittal angle between the two
        planes and the raw geometry needed to draw them in 3D.

        The ground plane is built directly (no fitting): the left/right contact
        patches at each axle share X and Z and differ only in Y, so their
        midpoints give the front- and rear-axle contact centers, and the plane is
        the one through both centers running parallel to the Y (lateral) axis."""
        none = {
            "front_ground_clearance_mm": None, "rear_ground_clearance_mm": None,
            "chassis_ground_angle_deg": None, "gc_viz": None,
        }
        chassis = self.chassis_plane
        if chassis is None:
            return none
        fl_c = Point(get_contact_patch(fl, self.wr_front))
        fr_c = Point(get_contact_patch(fr, self.wr_front))
        rl_c = Point(get_contact_patch(rl, self.wr_rear))
        rr_c = Point(get_contact_patch(rr, self.wr_rear))
        contacts = [fl_c, fr_c, rl_c, rr_c]
        if not all(c.is_finite() for c in contacts):
            return none
        front_c = fl_c.midpoint_to(fr_c)
        rear_c  = rl_c.midpoint_to(rr_c)
        try:
            ground = Plane.from_points_and_direction(front_c, rear_c, Point(0.0, 1.0, 0.0))
        except ValueError:
            return none
        ground_centroid = front_c.midpoint_to(rear_c)

        def clearance_at(x: float) -> float:
            v = Line.vertical_through(x, 0.0)
            return chassis.intersect_line(v).z - ground.intersect_line(v).z

        cf, cr = clearance_at(front_x), clearance_at(rear_x)
        low_point = chassis.point.translated(dz=25.4)  # the inboard lower-A-arm pickup itself
        return {
            "front_ground_clearance_mm": cf,
            "rear_ground_clearance_mm":  cr,
            "chassis_ground_angle_deg":  chassis.sagittal_angle_to(ground),
            "gc_viz": {
                "ground_centroid": ground_centroid.to_list(),
                "ground_normal": ground.normal.to_list(),
                "contacts": [c.to_list() for c in contacts],
                "chassis_bottom_z": chassis.point.z,
                "chassis_low_point": low_point.to_list(),
                "cog_point": self.cog_point.to_list(),
                "front_x": front_x, "rear_x": rear_x,
                "front_ground_z": ground.z_at(front_x, 0.0),
                "rear_ground_z": ground.z_at(rear_x, 0.0),
                "_front_clearance": cf, "_rear_clearance": cr,
            },
        }

    def _nan_step(self, t, fl, fr, rl, rr) -> Dict:
        return {
            "input": t, "fl": fl, "fr": fr, "rl": rl, "rr": rr,
            "front_track_mm": np.nan, "rear_track_mm": np.nan,
            "front_track_change_mm": np.nan, "rear_track_change_mm": np.nan,
            "wheelbase_mm": np.nan, "wheelbase_change_mm": np.nan,
            "pitch_angle_deg": np.nan,
            "front_roll_angle_deg": np.nan, "rear_roll_angle_deg": np.nan,
            "front_roll_center_y_mm": None, "front_roll_center_z_mm": None,
            "rear_roll_center_y_mm": None, "rear_roll_center_z_mm": None,
            "front_ground_clearance_mm": None, "rear_ground_clearance_mm": None,
            "chassis_ground_angle_deg": None, "gc_viz": None,
        }

    # Roll-center construction needs each corner's small-bump-perturbed neighbours
    # (see utils.geometry.roll_center_yz) -- this is the +/- bump_z step used for that.
    _RC_EPS = 1.0

    def run(self) -> List[Dict]:
        steps = []
        log.debug("full-vehicle scenario, %s mode", self.mode)
        bmin, bmax = self.bump_min, self.bump_max
        # heave sweeps jounce -> droop (all four corners together); roll sweeps
        # droop -> jounce since it's paired against a mirrored opposite-side value
        travel_vals = (np.linspace(bmax, bmin, self.config.sim_steps) if self.mode == "heave"
                        else np.linspace(bmin, bmax, self.config.sim_steps))

        for b in travel_vals:
            b_mirror = bmin + bmax - b
            if self.mode == "heave":
                fl_b = fr_b = rl_b = rr_b = b
            else:
                fl_b = rl_b = b
                fr_b = rr_b = b_mirror

            fl = self.fl_solver.solve(steer_mm=0.0, bump_z=fl_b)
            fr = self.fr_solver.solve(steer_mm=0.0, bump_z=fr_b)
            rl = self.rl_solver.solve(steer_mm=0.0, bump_z=rl_b)
            rr = self.rr_solver.solve(steer_mm=0.0, bump_z=rr_b)

            if fl and fr and rl and rr:
                if self.roll_center:
                    eps = self._RC_EPS
                    perturbed = (
                        self.fl_solver.solve(steer_mm=0.0, bump_z=fl_b + eps),
                        self.fl_solver.solve(steer_mm=0.0, bump_z=fl_b - eps),
                        self.fr_solver.solve(steer_mm=0.0, bump_z=fr_b + eps),
                        self.fr_solver.solve(steer_mm=0.0, bump_z=fr_b - eps),
                        self.rl_solver.solve(steer_mm=0.0, bump_z=rl_b + eps),
                        self.rl_solver.solve(steer_mm=0.0, bump_z=rl_b - eps),
                        self.rr_solver.solve(steer_mm=0.0, bump_z=rr_b + eps),
                        self.rr_solver.solve(steer_mm=0.0, bump_z=rr_b - eps),
                    )
                else:
                    perturbed = (None,) * 8
                steps.append(self._build_step(b, fl, fr, rl, rr, perturbed))
            else:
                log.debug("full-vehicle %s step failed at input %.2fmm (FL=%s FR=%s RL=%s RR=%s)",
                          self.mode, b, bool(fl), bool(fr), bool(rl), bool(rr))
                steps.append(self._nan_step(b, fl, fr, rl, rr))

        return steps
