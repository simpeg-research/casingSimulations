from .model import CasingParameters, PhysicalProperties
from .mesh import (
    CylMeshGenerator, TensorMeshGenerator,
    edge3DthetaSlice, face3DthetaSlice
)
from .physics import (
    CasingCurrents, plotCurrentDensity, plot_currents_over_freq,
    plot_currents_over_mu, plot_j_over_mu_z, plot_j_over_freq_z,
    plot_j_over_mu_x
)
from .view import plotEdge2D, plotFace2D
import sources
import run
from utils import load_properties

__version__   = '0.0.1'
__author__    = 'Lindsey Heagy'
__license__   = 'MIT'
__copyright__ = 'Copyright 2017 Lindsey Heagy'
