# default
from abc import ABC, abstractmethod
from itertools import combinations

# third-party
import numpy as np

# ours
from utils.geometry import get_toe_angle, segment_segment_distance

class OptimizationObjective(ABC):
    def __init__(self, config: dict = None):
        self.config = config or {}

    @abstractmethod
    def calculate_cost(self, results: list) -> float:
        """Returns a scalar cost for a given simulation run."""
        pass

    @abstractmethod
    def get_scenario_type(self) -> str:
        """Returns the key of the scenario class to run ('steer', 'travel', 'ackermann')."""
        pass

    @property
    def name(self):
        """Helper to get class name for logging/plotting."""
        return self.__class__.__name__

class MinimumBumpSteer(OptimizationObjective):
    def calculate_cost(self, results):
        toes = np.array([get_toe_angle(step) for step in results])
        max_abs_toe = np.max(np.abs(toes))
        toe_range = np.max(toes) - np.min(toes)
        return (max_abs_toe + toe_range)/150.0

    def get_scenario_type(self):
        return 'travel'

class ParallelSteer(OptimizationObjective):
    def calculate_cost(self, results):
        pcts = np.array([step['ackermann_pct'] for step in results])
        rmse = np.sqrt(np.mean(pcts**2))
        if any(np.isnan(pcts)) or np.isnan(rmse):
            return 1e2
        else:
            return rmse/1400.0
    
    def get_scenario_type(self):
        return 'ackermann'

class PointToPointCollision(OptimizationObjective):
    """
    Generic keepout-zone collision objective. Each zone in KEEPOUT_ZONES defines a
    shape (cylinder or box) extruded along the axis between two named hardpoints.
    Cost is the summed penetration between every checked pair of zones, across the sweep.

    If COLLISION_GROUPS is set (group name -> list of zone names), a group marks its
    member zones as ALLOWED to overlap each other — same-group pairs are exempt from
    the cost. Every other pair (different groups, or either zone ungrouped) IS checked
    and penalized if it overlaps. Without COLLISION_GROUPS, every pair of zones is
    checked (requires >=2 zones total). Each group may hold at most MAX_GROUP_SIZE
    zones; excess members past that limit lose their exemption and are checked
    normally against the rest.
    """
    MAX_GROUP_SIZE = 10

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.zones = (self.config or {}).get("KEEPOUT_ZONES", []) or []
        self.groups = (self.config or {}).get("COLLISION_GROUPS", None)
        self._pairs = self._build_pairs()
        if not self._pairs:
            print(f"WARNING: PointToPointCollision has no collidable zone pairs "
                  f"(zones={len(self.zones)}, groups={'set' if self.groups else 'unset'}). "
                  f"Cost will always be 0.")

    def _build_pairs(self):
        if not self.groups:
            return list(combinations(self.zones, 2))

        zones_by_name = {z.get("name"): z for z in self.zones}
        exempt_group = {}  # zone name -> group name, only for zones within the size cap
        for group_name, members in self.groups.items():
            valid_members = []
            for member in members:
                if member not in zones_by_name:
                    print(f"WARNING: COLLISION_GROUPS['{group_name}'] references "
                          f"unknown zone '{member}'.")
                    continue
                valid_members.append(member)
            if len(valid_members) > self.MAX_GROUP_SIZE:
                print(f"WARNING: COLLISION_GROUPS['{group_name}'] has {len(valid_members)} zones, "
                      f"exceeding the max of {self.MAX_GROUP_SIZE}. Only the first "
                      f"{self.MAX_GROUP_SIZE} are exempt from collision with each other; "
                      f"the rest will be checked normally.")
                valid_members = valid_members[:self.MAX_GROUP_SIZE]
            for member in valid_members:
                exempt_group[member] = group_name

        pairs = []
        for za, zb in combinations(self.zones, 2):
            ga = exempt_group.get(za.get("name"))
            gb = exempt_group.get(zb.get("name"))
            if ga is not None and ga == gb:
                continue  # same group -> allowed to overlap, skip
            pairs.append((za, zb))
        return pairs

    @staticmethod
    def _zone_radius(zone: dict) -> float:
        if zone.get("shape") == "box":
            dim1 = float(zone.get("dim1", 0.0))
            dim2 = float(zone.get("dim2", 0.0))
            return 0.5 * float(np.hypot(dim1, dim2))
        return float(zone.get("dim1", 0.0))

    @staticmethod
    def _resolve_point(name: str, step: dict) -> np.ndarray:
        return np.asarray(step[name], dtype=float)

    def calculate_cost(self, results):
        if not self._pairs:
            return 0.0

        pairs = self._pairs
        total_violation = 0.0

        for step in results:
            try:
                endpoints = {
                    id(z): (self._resolve_point(z["point_a"], step),
                            self._resolve_point(z["point_b"], step))
                    for z in self.zones
                }
            except KeyError as e:
                raise KeyError(f"KEEPOUT_ZONES point {e} not found in scenario step output") from e

            for za, zb in pairs:
                pa1, pa2 = endpoints[id(za)]
                pb1, pb2 = endpoints[id(zb)]
                d = segment_segment_distance(pa1, pa2, pb1, pb2)
                r_sum = self._zone_radius(za) + self._zone_radius(zb)
                total_violation += max(0.0, r_sum - d)

        return total_violation / len(results)

    def get_scenario_type(self):
        return self.config.get("COLLISION_SCENARIO", "droop_steer")