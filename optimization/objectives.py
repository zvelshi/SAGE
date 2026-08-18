# default
from abc import ABC, abstractmethod

# third-party
import numpy as np

# ours
from utils.geometry import get_toe_angle

class OptimizationObjective(ABC):
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

class NoCollision(OptimizationObjective):
    def calculate_cost(self, results):
        tr_ib = np.array([step['tr_ib'] for step in results])
        tr_ob = np.array([step['tr_ob'] for step in results])
        ubj = np.array([step['ubj'] for step in results])
        uf  = np.array([step['uf'] for step in results])
        ur  = np.array([step['ur'] for step in results])
        
        # form plane using upper a arm vectors
        v1 = uf - ubj
        v2 = ur - ubj
        normals = np.cross(v1, v2)
        
        # normalize vectors
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / (norms + 1e-9)
        
        # force upper a arm plane normal to point +ve z
        flip_mask = normals[:, 2] < 0
        normals[flip_mask] *= -1

        # calculate signed distance from tie rod points to upper a arm plane
        dist_ib = np.einsum('ij,ij->i', tr_ib - ubj, normals)
        dist_ob = np.einsum('ij,ij->i', tr_ob - ubj, normals)
        
        # allow the tie rod to be no higher than 12.7mm below the a arm plane center
        tol = -12.7
        
        # calculate the gradient cost violation
        # if the tie rod is above the plane (dist > 0), no violation
        # if it's below the plane (dist < 0), the violation is how far below it is
        viol_ib = np.maximum(0, dist_ib - tol)
        viol_ob = np.maximum(0, dist_ob - tol)
        
        # sum of all violations across the sweep, normalized by steps
        total_violation = np.sum(viol_ib) + np.sum(viol_ob)
        return total_violation / len(results)

    def get_scenario_type(self):
        return 'droop_steer'