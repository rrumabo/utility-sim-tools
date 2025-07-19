import numpy as np

def gaussian_bump(x, center=0.0, width=0.5, amplitude=1.0):
    return amplitude * np.exp(-((x - center) / width)**2)

def square_pulse(x, center=0.0, width=1.0, amplitude=1.0):
    return amplitude * ((x > center - width/2) & (x < center + width/2)).astype(float)

def triangle_wave(x, center=0.0, width=1.0, amplitude=1.0):
    slope = 2 * amplitude / width
    return np.clip(amplitude - slope * np.abs(x - center), 0, amplitude)