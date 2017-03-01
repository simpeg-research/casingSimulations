import numpy as np
import properties
from SimPEG import Utils

##############################################################################
#                                                                            #
#                           Simulation Parameters                            #
#                                                                            #
##############################################################################


# Parameters to set up the model
class CasingProperties(properties.HasProperties):
    """
    Simulation Parameters
    """
    # Conductivities
    sigma_air = properties.Float(
        "conductivity of the air (S/m)",
        default=1e-8
    )

    sigma_back = properties.Float(
        "conductivity of the background (S/m)",
        default=1e-2
    )

    sigma_layer = properties.Float(
        "conductivity of the layer (S/m)",
        default=1e-2
    )

    sigma_casing = properties.Float(
        "conductivity of the casing (S/m)",
        default=5.5e6
    )

    sigma_inside = properties.Float(
        "conductivity of the fluid inside the casing (S/m)",
        default=1.
    )

    # Magnetic Permeability
    mu_casing = properties.Float(
        "permeability of the casing",
        default= 100.
    )

    # Casing Geometry
    casing_top = properties.Float(
        "top of the casing (m)",
        default=0.
    )
    casing_l = properties.Float(
        "length of the casing (m)",
        default=1000
    )

    casing_d = properties.Float(
        "diameter of the casing (m)",
        default=10e-2
    ) # 10cm diameter

    casing_t = properties.Float(
        "thickness of the casing (m)",
        default=1e-2
    ) # 1cm thickness

    # Layer Geometry
    layer_z = properties.Array(
        "z-limits of the layer",
        shape=(2,),
        default=np.r_[-1000., -900.]
    )

    freqs = properties.Array(
        "source frequencies",
        default=np.r_[1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1],
        dtype=float
    )

    src_a = properties.Array(
        "down-hole z-location for the source",
        default=np.r_[0., 0., -975.]
    )

    src_b = properties.Array(
        "B electrode location",
        default=np.r_[1e4, 0., 0.]
    )

    def __init__(self, **kwargs):
        Utils.setKwargs(self, **kwargs)

    # useful quantities to work in
    @property
    def casing_r(self):
        return self.casing_d/2.

    @property
    def casing_a(self):
        return self.casing_r - self.casing_t/2.  # inner radius

    @property
    def casing_b(self):
        return self.casing_r + self.casing_t/2.  # outer radius

    @property
    def casing_z(self):
        return np.r_[-self.casing_l, 0.] + self.casing_top
