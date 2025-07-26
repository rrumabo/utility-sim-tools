import numpy as np
import yaml
import csv
import os

class DiagnosticManager:
    def __init__(self, dx=None, dy=None, u_ref=None, track=("min", "max", "mean", "mass", "l2_error")):
        if dx is None:
            raise ValueError("dx must be provided for diagnostics involving spatial integration.")
        self.dx = dx
        self.dy = dy if dy is not None else dx
        self.u_ref = u_ref
        self.track = set(track)
        self.records = []

    def track_step(self, u, t):
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
        with open(path, "w") as f:
            yaml.dump(self.records, f)

    def save_csv(self, path):
        if not self.records:
            return
        keys = self.records[0].keys()
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.records)