from .Model import CasingParameters, PhysicalProperties
from .Mesh import MeshGenerator, edge3DthetaSlice, face3DthetaSlice
from .Physics import (
    CasingCurrents, plotCurrentDensity, plot_currents_over_freq,
    plot_currents_over_mu, plot_j_over_mu_z, plot_j_over_freq_z,
    plot_j_over_mu_x
)
from .View import plotEdge2D, plotFace2D
import Sources
import Run
from Utils import load_properties

__version__   = '0.0.1'
__author__    = 'Lindsey Heagy'
__license__   = 'MIT'
__copyright__ = 'Copyright 2017 Lindsey Heagy'
