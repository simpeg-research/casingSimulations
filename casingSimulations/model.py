import numpy as np
import properties
import json
import os
from SimPEG import Maps, Utils
from scipy.constants import mu_0

import discretize
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from .base import BaseCasing


# Global variables (Filenames)
SIMULATION_PARAMETERS_FILENAME = "ModelParameters.json"


##############################################################################
#                                                                            #
#                           Simulation Parameters                            #
#                                                                            #
##############################################################################

class TimeStepArray(properties.Array):

    class_info = 'an array or list of tuples specifying the mesh tensor'

    def validate(self, instance, value):
        if isinstance(value, list):
            value = discretize.utils.meshTensor(value)
        return super(TimeStepArray, self).validate(instance, value)


class SurveyParametersMixin(properties.HasProperties):
    """
    A mixin that has the properties of the survey. It doesn't do anything on
    its own
    """

    # survey parameters
    freqs = properties.Array(
        "source frequencies",
        required=False,
        dtype=float
    )

    timeSteps = TimeStepArray(
        "times-steps at which to solve",
        required=False,
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


class Wholespace(SurveyParametersMixin, BaseCasing):
    """
    Model and survey parameters for an electromagnetic survey in a wholespace
    """
    filename = properties.String(
        "Filename to which the properties are serialized and written to",
        default=SIMULATION_PARAMETERS_FILENAME
    )

    sigma_back = properties.Float(
        "conductivity of the background (S/m)",
        default=1e-2,
        min=0.
    )

    mur_back = properties.Float(
        "relative permittivity of the background",
        default=1.,
        min=0.
    )

    def __init__(self, filename=None, **kwargs):
        Utils.setKwargs(self, **kwargs)

    # handy functions
    def skin_depth(self, sigma=None, mu=None, f=None):
        """
        Skin depth

        .. math::

            \delta = \sqrt(\\frac{2}{\omega \mu \sigma})

        """
        if sigma is None:
            sigma = self.sigma_back
        if mu is None:
            mu = mu_0
        if f is None:
            f = self.freqs
        return np.sqrt(2./(2.*np.pi*f*mu*sigma))

    def diffusion_distance(self, t=None, sigma=None, mu=None):
        """
        Difusion distance

        .. math::

        """
        if sigma is None:
            sigma = self.sigma_back
        if mu is None:
            mu = mu_0
        if t is None:
            t = self.timeSteps.sum()
        return np.sqrt(2*t/(mu*sigma))

    def sigma(self, mesh):
        """
        Electrical conductivity on a mesh
        :param discretize.BaseMesh mesh: a discretize mesh
        :rtype: numpy.ndarray
        :return: electrical conductivity on the mesh
        """
        return self.sigma_back * np.ones(mesh.nC)

    def mur(self, mesh):
        """
        Relative magnetic permeability on a mesh
        :param discretize.BaseMesh mesh: a discretize mesh
        :rtype: numpy.ndarray
        :return: relative magnetic permeability on the mesh
        """
        return self.mur_back * np.ones(mesh.nC)

    def mu(self, mesh):
        """
        Magnetic permeability on a mesh
        :param discretize.BaseMesh mesh: a discretize mesh
        :rtype: numpy.ndarray
        :return: magnetic permeability on the mesh
        """
        return mu_0 * self.mur(mesh)


class Halfspace(Wholespace):
    """
    Model and survey parameters for an electromagnetic survey in a halfspace
    """
    sigma_air = properties.Float(
        "conductivity of the air (S/m)",
        default=1e-6
    )

    surface_z = properties.Float(
        "elevation of the air-earth interface (m)",
        default=0
    )

    def ind_air(self, mesh):
        """
        indices where the air is

        :param discretize.BaseMesh mesh: mesh to find the air cells of
        :rtype: bool
        """
        return mesh.gridCC[:, 2] > self.surface_z

    def sigma(self, mesh):
        """
        put the conductivity model on a mesh

        :param discretize.BaseMesh mesh: mesh to find air cells of
        :rtype: numpy.array
        """
        sigma = super(Halfspace, self).sigma(mesh)
        sigma[self.ind_air(mesh)] = self.sigma_air
        return sigma


class SingleLayer(Halfspace):
    """
    A model consisting of air, subsurface and a single subsurface layer
    """
    sigma_layer = properties.Float(
        "conductivity of the layer (S/m)",
        default=1e-2
    )

    layer_z = properties.Array(
        "z-limits of the layer",
        shape=(2,),
        default=np.r_[-1000., -900.]
    )

    def ind_layer(self, mesh):
        """
        indices where the layer is

        :param discretize.BaseMesh mesh: mesh to find layer indices on
        :rtype: numpy.array
        """
        return (
            (mesh.gridCC[:, 2] < self.layer_z[1]) &
            (mesh.gridCC[:, 2] > self.layer_z[0])
        )

    def sigma(self, mesh):
        """
        put the conductivity model on a mesh
        """
        sigma = super(self, sigma)(mesh)
        sigma[self.ind_layer(mesh)] = self.sigma_layer
        return sigma


# class Layers(BaseCasing):
#     pass


class BaseCasingParametersMixin(BaseCasing):
    """
    Parameters used to set up a casing in a background. This class does not
    function on its own. It should be mixed in with the background model of
    your choice
    """
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

    # useful quantities to work in
    @property
    def casing_r(self):
        """
        Casing radius

        :rtype: float
        """
        return self.casing_d/2.

    @property
    def casing_a(self):
        """
        Inner casing radius

        :rtype: float
        """
        return self.casing_r - self.casing_t/2.  # inner radius

    @property
    def casing_b(self):
        """
        Outer casing radius

        :rtype: float
        """
        return self.casing_r + self.casing_t/2.  # outer radius

    @property
    def casing_z(self):
        """
        z-extent of the casing

        :rtype: numpy.array
        """
        return np.r_[-self.casing_l, 0.] + self.casing_top

    def indx_casing(self, mesh):
        """
        x-indices of the casing

        :param discretize.BaseMesh mesh: a discretize mesh
        :rtype: numpy.array
        """
        return (
            (mesh.gridCC[:, 0] > self.casing_a) &
            (mesh.gridCC[:, 0] < self.casing_b)
        )

    def indz_casing(self, mesh):
        """
        z-indices of the casing

        :param discretize.BaseMesh mesh: a discretize mesh
        :rtype: numpy.array
        """
        return (
            (mesh.gridCC[:, 2] > self.casing_z[0]) &
            (mesh.gridCC[:, 2] < self.casing_z[1])
        )

    def indx_inside(self, mesh):
        """
        x indicies of the inside of the casing

        :param discretize.BaseMesh mesh: a discretize mesh
        :rtype: numpy.array
        """
        return mesh.gridCC[:, 0] < self.casing_a

    def ind_casing(self, mesh):
        """
        indices of the cell centers of the casing

        :param discretize.BaseMesh mesh: a discretize mesh
        :rtype: numpy.array
        """
        return self.indx_casing(mesh) & self.indz_casing(mesh)

    def ind_inside(self, mesh):
        """
        indices of the cell centers of the inside portion of the casing

        :param discretize.BaseMesh mesh: a discretize mesh
        :rtype: numpy.array
        """
        return self.indx_inside(mesh) & self.indz_casing(mesh)

    def add_sigma_casing(self, mesh, sigma):
        """
        add the conductivity of the casing to the provided conductivity model
        :param discretize.BaseMesh mesh: a discretize mesh
        :param numpy.ndarray sigma: electrical conductivity model to modify
        :rtype: numpy.ndarray
        :return: electrical conductivity model with casing
        """
        sigma[self.ind_casing(mesh)] = self.sigma_casing
        sigma[self.ind_inside(mesh)] = self.sigma_inside
        return sigma

    def add_mur_casing(self, mesh, mur):
        """
        add relative magnetic permeability of the casing to the provided model
        :param discretize.BaseMesh mesh: a discretize mesh
        :param numpy.ndarray mur: relative magnetic permittivity model to modify
        :rtype: numpy.ndarray
        :return: relative magnetic permeability model with casing
        """
        mur[self.ind_casing(mesh)] = self.mur_casing
        return mur


class CasingInWholespace(Wholespace, BaseCasingParametersMixin):
    """
    A model of casing in a wholespace
    """
    def sigma(self, mesh):
        """
        put the conductivity model on a mesh

        :param discretize.BaseMesh mesh: a discretize mesh
        :rtype: numpy.array
        """
        sigma = super(CasingInWholespace, self).sigma(mesh)
        return self.add_sigma_casing(mesh, sigma)

    def mur(self, mesh):
        """
        put the permeability model on a mesh

        :param discretize.BaseMesh mesh: a discretize mesh
        :rtype: numpy.array
        """
        mur = super(CasingInWholespace, self).mur(mesh)
        return self.add_mur_casing(mesh, mur)


class CasingInHalfspace(Halfspace, BaseCasingParametersMixin):
    """
    A model of casing in a halfspace
    """
    def sigma(self, mesh):
        """
        put the conductivity model on a mesh

        :param discretize.BaseMesh mesh: a discretize mesh
        :rtype: numpy.array
        """
        sigma = super(CasingInHalfspace, self).sigma(mesh)
        return self.add_sigma_casing(mesh, sigma)

    def mur(self, mesh):
        """
        put the permeability model on a mesh

        :param discretize.BaseMesh mesh: a discretize mesh
        :rtype: numpy.array
        """
        mur = super(CasingInHalfspace, self).mur(mesh)
        return self.add_mur_casing(mesh, mur)


class CasingInSingleLayer(SingleLayer, BaseCasingParametersMixin):
    """
    A model of casing in an earth that has a single layer
    """
    def sigma(self, mesh):
        """
        put the conductivity model on a mesh

        :param discretize.BaseMesh mesh: a discretize mesh
        :rtype: numpy.array
        """
        sigma = super(CasingInSingleLayer, self).sigma(mesh)
        return self.add_sigma_casing(mesh, sigma)

    def mur(self, mesh):
        """
        put the permeability model on a mesh

        :param discretize.BaseMesh mesh: a discretize mesh
        :rtype: numpy.array
        """
        mur = super(CasingInSingleLayer, self).mur(mesh)
        return self.add_mur_casing(mesh, mur)


class PhysicalProperties(object):
    """
    Physical properties on the mesh
    """
    def __init__(self, meshGenerator, modelParameters):
        self.meshGenerator = meshGenerator
        self.mesh = meshGenerator.mesh
        self.modelParameters = modelParameters

    @property
    def mur(self):
        """
        relative permeability

        :rtype: numpy.array
        """
        if getattr(self, '_mur', None) is None:
            self._mur = self.modelParameters.mur(self.mesh)
        return self._mur

    @property
    def mu(self):
        """
        permeability

        :rtype: numpy.array
        """
        return mu_0 * self.mur

    @property
    def sigma(self):
        """
        electrical conductivity

        :rtype: numpy.array
        """
        if getattr(self, '_sigma', None) is None:
            self._sigma = self.modelParameters.sigma(self.mesh)
        return self._sigma

    @property
    def model(self):
        """
        model vector [sigma, mu]

        :rtype: numpy.array
        """
        return np.hstack([self.sigma, self.mu])

    @property
    def wires(self):
        """
        wires to hook up maps to sigma, mu

        :rtype: SimPEG.Maps.Wires
        """
        if getattr(self, '_wires', None) is None:
            self._wires = Maps.Wires(
                ('sigma', self.mesh.nC), ('mu', self.mesh.nC)
            )
        return self._wires

    def plot_prop(self, prop, ax=None, clim=None, pcolorOpts=None):
        """
        Plot a cell centered property

        :param numpy.array prop: cell centered property to plot
        :param matplotlib.axes ax: axis
        :param numpy.array clim: colorbar limits
        :param dict pcolorOpts: dictionary of pcolor options
        """

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        if pcolorOpts is None:
            pcolorOpts = {}

        # generate a 2D mesh for plotting slices
        mesh2D = discretize.CylMesh(
            [self.mesh.hx, 1., self.mesh.hz], x0=self.mesh.x0
        )

        propplt = prop.reshape(self.mesh.vnC, order='F')

        cb = plt.colorbar(
            mesh2D.plotImage(
                discretize.utils.mkvc(propplt[:, 0, :]), ax=ax,
                mirror=True, pcolorOpts=pcolorOpts
            )[0], ax=ax,

        )

        if clim is not None:
            cb.set_clim(clim)
            cb.update_ticks()

        return ax

    def plot_sigma(self, ax=None, clim=None, pcolorOpts=None):
        """
        plot the electrical conductivity

        :param matplotlib.axes ax: axis
        :param numpy.array clim: colorbar limits
        :param dict pcolorOpts: dictionary of pcolor options
        """
        self.plot_prop(self.sigma, ax=ax, clim=clim, pcolorOpts=pcolorOpts)
        ax.set_title('$\sigma$')
        return ax

    def plot_mur(self, ax=None, clim=None, pcolorOpts=None):
        """
        plot the relative permeability

        :param matplotlib.axes ax: axis
        :param numpy.array clim: colorbar limits
        :param dict pcolorOpts: dictionary of pcolor options
        """

        self.plot_prop(self.mur, ax=ax, clim=clim, pcolorOpts=pcolorOpts)
        ax.set_title('$\mu_r$')
        return ax

    def plot(self, ax=None, clim=[None, None], pcolorOpts=None):
        """
        plot the electrical conductivity and relative permeability

        :param matplotlib.axes ax: axis
        :param list clim: list of numpy arrays: colorbar limits
        :param dict pcolorOpts: dictionary of pcolor options
        """

        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        self.plot_sigma(ax=ax[0], clim=clim[0], pcolorOpts=pcolorOpts)
        self.plot_mur(ax=ax[1], clim=clim[1], pcolorOpts=pcolorOpts)

        plt.tight_layout()
        return ax



