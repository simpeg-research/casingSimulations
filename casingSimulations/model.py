import numpy as np
import properties
import json
import os
from SimPEG import Maps, Utils
from scipy.constants import mu_0

import discretize
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from .info import __version__
##############################################################################
#                                                                            #
#                           Simulation Parameters                            #
#                                                                            #
##############################################################################


# Parameters to set up the model
class CasingParameters(properties.HasProperties):
    """
    Simulation Parameters
    """
    version = properties.String(
        "version of the software",
        default = __version__
    )

    # Conductivities
    sigma_air = properties.Float(
        "conductivity of the air (S/m)",
        default=1e-6
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
    mur_casing = properties.Float(
        "relative permeability of the casing",
        default=100.
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
        default=np.r_[0.5],
        dtype=float
    )

    src_a = properties.Array(
        "down-hole z-location for the source",
        default=np.r_[0., 0., -975.]
    )

    src_b = properties.Array(
        "B electrode location",
        default=np.r_[1e3, 0., 0.]
    )

    def __init__(self, filename=None, **kwargs):
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

    def skin_depth(self, sigma=None, mu=None, f=None):
        if sigma is None:
            sigma = self.sigma_back
        if mu is None:
            mu = mu_0
        if f is None:
            f = self.freqs
        return np.sqrt(2./(2.*np.pi*f*mu*sigma))

    def save(self, filename='CasingParameters.json', directory='.'):
        """
        Save the casing properties to json
        :param str file: filename for saving the casing properties
        """
        if not os.path.isdir(directory):  # check if the directory exists
            os.mkdir(directory)  # if not, create it
        f = '/'.join([directory, filename])
        with open(f, 'w') as outfile:
            cp = json.dump(self.serialize(), outfile)

    def copy(self):
        """
        Make a copy of the current CasingParameters object
        """
        return CasingParameters.deserialize(self.serialize())


class PhysicalProperties(object):
    """
    Physical properties on the mesh
    """
    def __init__(self, mesh, cp):
        self.mesh = mesh
        self.cp = cp

    @property
    def casing_xind(self):
        """
        x-indices of the casing
        """
        return (
            (self.mesh.gridCC[:, 0] > self.cp.casing_a) &
            (self.mesh.gridCC[:, 0] < self.cp.casing_b)
        )

    @property
    def casing_zind(self):
        """
        z-indices of the casing
        """
        return (
            (self.mesh.gridCC[:, 2] > self.cp.casing_z[0]) &
            (self.mesh.gridCC[:, 2] < self.cp.casing_z[1])
        )

    @property
    def inside_xind(self):
        return (self.mesh.gridCC[:, 0] < self.cp.casing_a)

    @property
    def casing_ind(self):
        return self.casing_xind & self.casing_zind

    @property
    def inside_ind(self):
        return self.inside_xind & self.casing_zind

    @property
    def air_ind(self):
        return self.mesh.gridCC[:, 2] > 0.

    @property
    def layer_ind(self):
        return (
            (self.mesh.gridCC[:, 2] > self.cp.layer_z[0]) &
            (self.mesh.gridCC[:, 2] < self.cp.layer_z[1])
        )

    @property
    def sigma(self):
        if getattr(self, '_sigma', None) is None:
            sigma = self.cp.sigma_back * np.ones(self.mesh.nC)
            sigma[self.air_ind] = self.cp.sigma_air
            sigma[self.layer_ind] = self.cp.sigma_layer
            sigma[self.casing_ind] = self.cp.sigma_casing
            sigma[self.inside_ind] = self.cp.sigma_inside
            self._sigma = sigma
        return self._sigma

    @property
    def mur(self):
        if getattr(self, '_mur', None) is None:
            mur = np.ones(self.mesh.nC)
            mur[self.casing_ind] = self.cp.mur_casing
            self._mur = mur
        return self._mur

    @property
    def mu(self):
        return mu_0 * self.mur

    @property
    def model(self):
        return np.hstack([self.sigma, self.mu])

    @property
    def wires(self):
        if getattr(self, '_wires', None) is None:
            self._wires = Maps.Wires(
                ('sigma', self.mesh.nC), ('mu', self.mesh.nC)
            )
        return self._wires

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        # generate a 2D mesh for plotting slices
        mesh2D = discretize.CylMesh(
            [self.mesh.hx, 1., self.mesh.hz], x0 = self.mesh.x0
        )

        # plot Sigma
        sigmaplt = self.sigma.reshape(self.mesh.vnC, order='F')
        ax[0].set_title('$\sigma$')
        plt.colorbar(
            mesh2D.plotImage(
                discretize.utils.mkvc(sigmaplt[:, 0, :]), ax=ax[0],
                mirror=True, pcolorOpts={'norm': LogNorm()}
            )[0], ax=ax[0],

        )

        # Plot mu
        murplt = self.mur.reshape(self.mesh.vnC, order='F')
        ax[1].set_title('$\mu_r$')
        plt.colorbar(mesh2D.plotImage(
            discretize.utils.mkvc(murplt[:, 0, :]), ax=ax[1], mirror=True)[0],
            ax=ax[1]
        )

        plt.tight_layout()
        return ax





