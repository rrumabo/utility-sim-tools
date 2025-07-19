import numpy as np

def gaussian_bump_2d(X, Y, center=(0, 0), width=0.5, amplitude=1.0):
    return amplitude * np.exp(-((X - center[0])**2 + (Y - center[1])**2) / (2 * width**2))