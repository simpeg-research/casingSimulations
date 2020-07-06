import numpy as np
import properties
import json
import os
from SimPEG import maps
from SimPEG.utils import setKwargs
from scipy.constants import mu_0

import discretize
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from .base import BaseCasing
from .view import plot_slice


##############################################################################
#                                                                            #
#                                 Defaults                                   #
#                                                                            #
##############################################################################

# Filenames
SIMULATION_PARAMETERS_FILENAME = "ModelParameters.json"

# Conductivities
SIGMA_BACK = 1e-2
SIGMA_AIR = 1e-6
SIGMA_CASING = 5.5e6

# Magnetic Permeability
MUR = 1.

# Casing parameters
CASING_L = 1000
CASING_D = 10e-2 # 10cm diameter
CASING_T = 1e-2 # 1cm thickness


##############################################################################
#                                                                            #
#                           Simulation Parameters                            #
#                                                                            #
##############################################################################

class TimeStepArray(properties.Array):

    class_info = "an array or list of tuples specifying the mesh tensor"

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
        "down-hole z-location for the source"
        # default=np.r_[0., 0., 0.]
    )

    src_b = properties.Array(
        "B electrode location"
        # default=np.r_[CASING_L, 0., 0.]
    )

    @property
    def info_survey(self):
        info = "\n ---- Survey ---- "

        # # src locations
        # info += "\n\n   src_a: {:s}".format(str(self.src_a))
        # info += "\n   src_b: {:s}".format(str(self.src_b))
        # info += "\n"

        # frequencies or times
        if self.freqs is not None:
            info += (
                "\n   {:1.0f} frequencies. "
                "min: {:1.1e} Hz, max: {:1.1e} Hz".format(
                    len(self.freqs), self.freqs.min(), self.freqs.max()
                )
            )

        if self.timeSteps is not None:
            info += (
                "\n   {:1.0f} time steps. min time step: {:1.1e} s, "
                "max time step: {:1.1e} s. Total time: {:1.1e} s".format(
                    len(self.timeSteps), self.timeSteps.min(),
                    self.timeSteps.max(), self.timeSteps.sum()
                )
            )
        return info


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
        default=SIGMA_BACK,
        min=0.
    )

    mur_back = properties.Float(
        "relative permittivity of the background",
        default=MUR,
        min=0.
    )

    def __init__(self, filename=None, **kwargs):
        setKwargs(self, **kwargs)

    def __str__(self):
        return self.info

    @property
    def info_model(self):
        info = "\n ---- Model ---- "
        info += "\n\n  background: "
        info += "\n    - conductivity: {:1.1e} S/m".format(self.sigma_back)
        info += "\n    - permeability: {:1.1f} mu_0".format(self.mur_back)
        return info

    @property
    def info(self):
        info = self.info_survey
        info += "\n\n" + self.info_model
        if hasattr(self, 'info_casing'):
            info += "\n\n" + self.info_casing
        return info

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
        default=SIGMA_AIR
    )

    surface_z = properties.Float(
        "elevation of the air-earth interface (m)",
        default=0
    )

    @property
    def info_model(self):
        info = super(Halfspace, self).info_model
        info += "\n\n  air: "
        info += "\n    - conductivity: {:1.1e} S/m".format(self.sigma_air)
        info += "\n    - earth surface elevaation: {:1.1f} m".format(
            self.surface_z
        )
        return info

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
        default=SIGMA_BACK
    )

    layer_z = properties.Array(
        "z-limits of the layer",
        shape=(2,),
        default=np.r_[-CASING_L, -CASING_L*0.9]
    )

    @property
    def info_model(self):
        info = super(SingleLayer, self).info_model
        info += "\n\n  layer: "
        info += "\n    - conductivity: {:1.1e} S/m".format(self.sigma_layer)
        info += "\n    - layer z: {} m".format(self.layer_z)
        return info

    def ind_layer(self, mesh):
        """
        Indices where the layer is
        """
        return (
            (mesh.gridCC[:, 2] < self.layer_z[1]) &
            (mesh.gridCC[:, 2] > self.layer_z[0])
        )

    def sigma(self, mesh):
        """
        Construct the conductivity model on a mesh

        :param discretize.BaseMesh mesh: mesh to put conductivity model on
        """
        sigma = super(SingleLayer, self).sigma(mesh)
        sigma[self.ind_layer(mesh)] = self.sigma_layer
        return sigma


class Layers(Halfspace):
    """
    A model containing an arbitrary number of layers
    """

    sigma_layers = properties.List(
        "list containing the conductivity of each of the layers (S/m)",
        properties.Float(
            "conductivity of the layer (S/m)",
            min=0.
        ),
        default=[SIGMA_BACK]
    )

    layer_tops = properties.List(
        "top of each of the layers",
        properties.Float(
            "top of each of the layers, z (m)"
        ),
        default=[0.]
    )

    # todo: sanity checking that sigma_layers and layer_tops the same size

    def sigma(self, mesh):
        """
        Construct the conductivity model on a mesh

        :param discretize.BaseMesh mesh: mesh to put conductivity model on
        """
        sigma = super(Layers, self).sigma(mesh)
        for z, sig in zip(self.layer_tops, self.sigma_layers):
            z_inds = mesh.gridCC[:, 2] < z
            sigma[z_inds] = sig
        return sigma


class TargetMixin(BaseCasing):

    target_radius = properties.Array(
        "radial extent of the target (m) [min, max]",
        shape=(2,),
        default=np.r_[0., 25.]
    )

    target_z = properties.Array(
        "vertical extent of the target (m) [min, max]",
        shape=(2,),
        default=np.r_[-925., -900.]
    )

    target_theta = properties.Array(
        "azimuthal extent of the target (m) [min, max]",
        shape=(2,),
        default=np.r_[0., 2*np.pi]
    )

    sigma_target = properties.Float(
        "conductivity of the target (S/m)",
        min=0.,
        default=SIGMA_BACK
    )

    def indx_target(self, mesh):
        return (
            (mesh.gridCC[:, 0] >= self.target_radius[0]) &
            (mesh.gridCC[:, 0] <= self.target_radius[1])
        )

    def indy_target(self, mesh):
        return (
            (mesh.gridCC[:, 1] >= self.target_theta[0]) &
            (mesh.gridCC[:, 1] <= self.target_theta[1])
        )

    def indz_target(self, mesh):
        return (
            (mesh.gridCC[:, 2] >= self.target_z[0]) &
            (mesh.gridCC[:, 2] <= self.target_z[1])
        )

    def ind_target(self, mesh):
        return (
            self.indx_target(mesh) & self.indy_target(mesh) &
            self.indz_target(mesh)
        )

    def add_sigma_target(self, mesh, sigma):
        ind_target = self.ind_target(mesh)
        sigma[ind_target] = self.sigma_target
        return sigma


class TargetInHalfspace(Halfspace, TargetMixin):

    def sigma(self, mesh):
        sigma = super(TargetInHalfspace, self).sigma(mesh)
        return self.add_sigma_target(mesh, sigma)

class CasingMixin(BaseCasing):
    """
    Parameters used to set up a casing in a background. This class does not
    function on its own. It should be mixed in with the background model of
    your choice
    """
    sigma_casing = properties.Float(
        "conductivity of the casing (S/m)",
        default=SIGMA_CASING
    )

    sigma_inside = properties.Float(
        "conductivity of the fluid inside the casing (S/m)",
        default=SIGMA_BACK
    )

    # Magnetic Permeability
    mur_casing = properties.Float(
        "relative permeability of the casing",
        default=MUR
    )

    mur_inside = properties.Float(
        "relative permeability of the casing",
        default=MUR
    )

    # Casing Geometry
    casing_top = properties.Float(
        "top of the casing (m)",
        default=0.
    )
    casing_l = properties.Float(
        "length of the casing (m)",
        default=CASING_L
    )

    casing_d = properties.Float(
        "diameter of the casing (m)",
        default=CASING_D
    )
    casing_t = properties.Float(
        "thickness of the casing (m)",
        default=CASING_T
    )

    @property
    def info_casing(self):
        info = "\n ---- Casing ---- "

        info += "\n\n  properties: "
        info += "\n    - conductivity: {:1.1e} S/m".format(self.sigma_casing)
        info += "\n    - permeability: {:1.1f} mu_0".format(self.mur_casing)
        info += "\n    - conductivity inside: {:1.1e} S/m".format(
            self.sigma_inside
        )

        info += "\n\n  geometry: "
        info += "\n    - casing top: {:1.1f} m".format(self.casing_top)
        info += "\n    - casing length: {:1.1f} m".format(self.casing_l)
        info += "\n    - casing diameter: {:1.1e} m".format(self.casing_d)
        info += "\n    - casing thickness: {:1.1e} m".format(self.casing_t)

        return info

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
        mur[self.ind_inside(mesh)] = self.mur_inside
        return mur


class FlawedCasingMixin(CasingMixin):

    """
    Model parameters for a flawed well.
    """

    flaw_r = properties.Array(
        "Radius of the flawed section of the casing [inner radius, outer "
        "radius]",
        shape=(2,),
        default=np.r_[0., 0.]
    )

    flaw_theta = properties.Array(
        "Azimuth of the flawed section of the casing [min azimuth, max "
        "azimuth]",
        shape=(2,),
        default=np.r_[0., 2*np.pi]
    )

    flaw_z = properties.Array(
        "vertical extent of the flawed section of the casing [z min, z max]",
        shape=(2,),
        default=np.r_[0., 0.]
    )

    sigma_flaw = properties.Float(
        "conductivity of the flawed section",
        default=SIGMA_CASING,
        min=0.
    )

    mur_flaw = properties.Float(
        "magnetic permeability of the flawed section",
        default=MUR,
        min=0.
    )

    def _indices_flaw_r(self, mesh):
        return (
            (mesh.gridCC[:, 0] >= self.flaw_r[0]) &
            (mesh.gridCC[:, 0] <= self.flaw_r[1])
        )

    def _indices_flaw_theta(self, mesh):
        return (
            (mesh.gridCC[:, 1] >= self.flaw_theta[0]) &
            (mesh.gridCC[:, 1] <= self.flaw_theta[1])
        )

    def _indices_flaw_z(self, mesh):
        return (
            (mesh.gridCC[:, 2] >= self.flaw_z[0]) &
            (mesh.gridCC[:, 2] <= self.flaw_z[1])
        )

    def indices_flaw(self, mesh):
        return (
            self._indices_flaw_r(mesh) &
            self._indices_flaw_theta(mesh) &
            self._indices_flaw_z(mesh)
        )

    def add_sigma_casing(self, mesh, sigma):
        """
        add the conductivity of the casing to the provided conductivity model
        :param discretize.BaseMesh mesh: a discretize mesh
        :param numpy.ndarray sigma: electrical conductivity model to modify
        :rtype: numpy.ndarray
        :return: electrical conductivity model with casing
        """
        sigma = super(FlawedCasingMixin, self).add_sigma_casing(mesh, sigma)
        sigma[self.indices_flaw(mesh)] = self.sigma_flaw
        return sigma

    def add_mur_casing(self, mesh, mur):
        """
        add relative magnetic permeability of the casing to the provided model
        :param discretize.BaseMesh mesh: a discretize mesh
        :param numpy.ndarray mur: relative magnetic permittivity model to modify
        :rtype: numpy.ndarray
        :return: relative magnetic permeability model with casing
        """
        mur = super(FlawedCasingMixin, self).add_mur_casing(mesh, mur)
        mur[self.indices_flaw(mesh)] = self.mur_flaw
        return mur

class CasingInWholespace(Wholespace, CasingMixin):
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


class CasingInHalfspace(Halfspace, CasingMixin):
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


class FlawedCasingInHalfspace(CasingInHalfspace, FlawedCasingMixin):
    """
    A model of a flawed casing in a wholespace
    """

class CasingInHalfspaceWithTarget(TargetInHalfspace, CasingMixin):
    """
    Casing in a halfspace with a target
    """

    def sigma(self, mesh):
        """
        put the conductivity model on a mesh

        :param discretize.BaseMesh mesh: a discretize mesh
        :rtype: numpy.array
        """
        sigma = super(CasingInHalfspaceWithTarget, self).sigma(mesh)
        return self.add_sigma_casing(mesh, sigma)

class CasingInSingleLayer(SingleLayer, CasingMixin):
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


class FlawedCasingInSingleLayer(CasingInSingleLayer, FlawedCasingMixin):
    """
    Flawed casing in a halfspace with a single layer present
    """


class CasingInLayers(Layers, CasingMixin):
    """
    A model of a casing in a layered space
    """

    def sigma(self, mesh):
        """
        put the conductivity model on a mesh

        :param discretize.BaseMesh mesh: a discretize mesh
        :rtype: numpy.array
        """
        sigma = super(CasingInLayers, self).sigma(mesh)
        return self.add_sigma_casing(mesh, sigma)

    def mur(self, mesh):
        """
        put the permeability model on a mesh

        :param discretize.BaseMesh mesh: a discretize mesh
        :rtype: numpy.array
        """
        mur = super(CasingInLayers, self).mur(mesh)
        return self.add_mur_casing(mesh, mur)


class FlawedCasingInLayers(CasingInLayers, FlawedCasingMixin):
    """
    Flawed casing in a layered space
    """


##############################################################################
#                                                                            #
#                             Physical Properties                            #
#                                                                            #
##############################################################################
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

        :rtype: SimPEG.maps.Wires
        """
        if getattr(self, '_wires', None) is None:
            self._wires = maps.Wires(
                ('sigma', self.mesh.nC), ('mu', self.mesh.nC)
            )
        return self._wires

    def plot_prop(
        self, prop, ax=None, clim=None, theta_ind=0, pcolorOpts=None,
        cb_extend=None, show_cb=True
    ):
        """
        Plot a cell centered property

        :param numpy.array prop: cell centered property to plot
        :param matplotlib.axes ax: axis
        :param numpy.array clim: colorbar limits
        :param dict pcolorOpts: dictionary of pcolor options
        """
        return plot_slice(
            self.mesh, prop, ax=ax, clim=clim, pcolorOpts=pcolorOpts,
            theta_ind=theta_ind, cb_extend=cb_extend, show_cb=show_cb
        )

    def plot_sigma(
        self, ax=None, clim=None, theta_ind=0, pcolorOpts=None, cb_extend=None,
        show_cb=True
    ):
        """
        plot the electrical conductivity

        :param matplotlib.axes ax: axis
        :param numpy.array clim: colorbar limits
        :param dict pcolorOpts: dictionary of pcolor options
        """
        out = self.plot_prop(
            self.sigma, ax=ax, clim=clim, theta_ind=theta_ind,
            pcolorOpts=pcolorOpts, cb_extend=cb_extend, show_cb=show_cb
        )
        out[1].set_title('$\sigma$')
        return out

    def plot_mur(
        self, ax=None, clim=None, theta_ind=0, pcolorOpts=None,
        cb_extend=None, show_cb=True
    ):
        """
        plot the relative permeability

        :param matplotlib.axes ax: axis
        :param numpy.array clim: colorbar limits
        :param dict pcolorOpts: dictionary of pcolor options
        """

        out = self.plot_prop(
            self.mur, ax=ax, clim=clim, theta_ind=theta_ind,
            pcolorOpts=pcolorOpts, cb_extend=cb_extend, show_cb=show_cb
        )
        out[1].set_title('$\mu_r$')
        return out

    def plot(
        self, ax=None, clim=[None, None], pcolorOpts=None, cb_extend=None
    ):
        """
        plot the electrical conductivity and relative permeability

        :param matplotlib.axes ax: axis
        :param list clim: list of numpy arrays: colorbar limits
        :param dict pcolorOpts: dictionary of pcolor options
        """

        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        if not isinstance(pcolorOpts, list):
            pcolorOpts = [pcolorOpts]*2

        if not isinstance(cb_extend, list):
            cb_extend = [cb_extend]*2

        self.plot_sigma(
            ax=ax[0], clim=clim[0], pcolorOpts=pcolorOpts[0],
            cb_extend=cb_extend[0]
        )
        self.plot_mur(
            ax=ax[1], clim=clim[1], pcolorOpts=pcolorOpts[1],
            cb_extend=cb_extend[1]
        )

        plt.tight_layout()
        return ax
