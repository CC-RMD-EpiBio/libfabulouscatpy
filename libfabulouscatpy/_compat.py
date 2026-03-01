"""Compatibility shims for numpy/scipy API changes."""

import numpy as np
from scipy import integrate

try:
    trapz = np.trapezoid
except AttributeError:
    trapz = np.trapz

try:
    cumtrapz = integrate.cumulative_trapezoid
except AttributeError:
    cumtrapz = integrate.cumtrapz
