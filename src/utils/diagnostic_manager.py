"""
diagnostic_manager.py
---------------------
Provides the DiagnosticManager class for tracking and saving diagnostic statistics during numerical simulations.
Supports tracking min, max, mean, mass, and L2 error at each time step, and saving results in YAML or CSV.
"""

import numpy as np
import yaml
import csv
import os

class DiagnosticManager:
    """
    Manages the collection and saving of diagnostic statistics during simulations.
    Tracks quantities such as min, max, mean, mass, and L2 error for each simulation step.
    Diagnostics can be saved in YAML or CSV formats.
    """
    def __init__(self, dx=None, dy=None, u_ref=None, track=("min", "max", "mean", "mass", "l2_error")):
        """
        Initialize the DiagnosticManager.

        Args:
            dx (float): Grid spacing in x-direction. Required for mass and L2 error.
            dy (float, optional): Grid spacing in y-direction. Defaults to dx if not provided.
            u_ref (np.ndarray, optional): Reference solution for L2 error computation.
            track (Iterable[str], optional): Diagnostics to track. Options: 'min', 'max', 'mean', 'mass', 'l2_error'.
        Raises:
            ValueError: If dx is not provided.
        """
        if dx is None:
            raise ValueError("dx must be provided for diagnostics involving spatial integration.")
        self.dx = dx
        self.dy = dy if dy is not None else dx
        self.u_ref = u_ref
        self.track = set(track)
        self.records = []

    def track_step(self, u, t):
        """
        Collects diagnostic statistics at a single time step.

        Args:
            u (np.ndarray): Solution array at current time step.
            t (float): Current simulation time.
        Raises:
            ValueError: If u and u_ref shapes mismatch when computing L2 error.
        """
        u = np.asarray(u)
        
        if "l2_error" in self.track and self.u_ref is not None:
            if u.shape != self.u_ref.shape:
                raise ValueError(f"Shape mismatch: u has shape {u.shape}, but u_ref has shape {self.u_ref.shape}")

        entry = {"time": t}
        if "min" in self.track:
            entry["min"] = float(np.min(u))
        if "max" in self.track:
            entry["max"] = float(np.max(u))
        if "mean" in self.track:
            entry["mean"] = float(np.mean(u))
        if "mass" in self.track:
            entry["mass"] = float(np.sum(u) * self.dx * self.dy)
        if "l2_error" in self.track and self.u_ref is not None:
            diff = u - self.u_ref
            entry["l2_error"] = float(np.sqrt(np.sum(diff**2) * self.dx * self.dy))
        self.records.append(entry)

    def save_yaml(self, path):
        """
        Save all collected diagnostics to a YAML file.

        Args:
            path (str): File path for saving diagnostics.
        """
        with open(path, "w") as f:
            yaml.dump(self.records, f)

    def save_csv(self, path):
        """
        Save all collected diagnostics to a CSV file.

        Args:
            path (str): File path for saving diagnostics.
        """
        if not self.records:
            return
        keys = self.records[0].keys()
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.records)