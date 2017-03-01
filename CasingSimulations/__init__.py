from .Model import CasingProperties
from .CasingMesh import CasingMesh
from .CasingPhysics import (
    CasingCurrents, plotCurrentDensity, plot_currents_over_freq,
    plot_currents_over_mu, plot_j_over_mu_z, plot_j_over_freq_z,
    plot_j_over_mu_x
)
import Sources

__version__   = '0.0.1'
__author__    = 'Lindsey Heagy'
__license__   = 'MIT'
__copyright__ = 'Copyright 2017 Lindsey Heagy'
