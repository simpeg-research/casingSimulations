import numpy as np
import properties
import json
import os
import matplotlib.pyplot as plt
import inspect

import properties
from SimPEG.utils import setKwargs

import discretize
from discretize import utils
from discretize.utils import mkvc

from . import model
from .base import BaseCasing
# __all__ = [TensorMeshGenerator, CylMeshGenerator]


# class DiscretizeClass(properties.Instance):

#     class_info = "a class type (for example when checking that a mesh is a "
#     "reasonable type)"

#     def validate(self, instance, value):
#         assert inspect.isclass(value), "value must be a class"
#         assert issubclass(CylMesh, discretize.BaseMesh), (
#             "value must be a subclass of discretize.BaseMesh"
#         )
#         return value

def pad_for_casing_and_data(
    casing_b=None,
    csx1=2.5e-3,
    csx2=25,
    pfx1=1.3,
    pfx2=1.5,
    domain_x=1000,
    npadx=10
):

    ncx1 = np.ceil(casing_b/csx1+2)
    npadx1 = np.floor(np.log(csx2/csx1) / np.log(pfx1))

    # finest uniform region
    hx1a = utils.meshTensor([(csx1, ncx1)])

    # pad to second uniform region
    hx1b = utils.meshTensor([(csx1, npadx1, pfx1)])

    # scale padding so it matches cell size properly
    dx1 = np.sum(hx1a)+np.sum(hx1b)
    dx1 = 3 #np.floor(dx1/self.csx2)
    hx1b *= (dx1*csx2 - np.sum(hx1a))/np.sum(hx1b)

    # second uniform chunk of mesh
    ncx2 = np.ceil((domain_x - dx1)/csx2)
    hx2a = utils.meshTensor([(csx2, ncx2)])

    # pad to infinity
    hx2b = utils.meshTensor([(csx2, npadx, pfx2)])

    return np.hstack([hx1a, hx1b, hx2a, hx2b])



class BaseMeshGenerator(BaseCasing):
    """
    Base Mesh Generator Class
    """

    filename = properties.String(
        "filename to serialize properties to",
        default="MeshParameters.json"
    )

    # casing parameters
    modelParameters = properties.Instance(
        "casing parameters instance",
        model.Wholespace,
        required=True
    )

    def __init__(self, **kwargs):
        setKwargs(self, **kwargs)

    @property
    def mesh(self):
        """
        discretize mesh

        :rtype: discretize.BaseMesh
        """
        if getattr(self, '_mesh', None) is None:
            self._mesh = self._discretizePair(
                [self.hx, self.hy, self.hz],
                x0=self.x0
            )
        return self._mesh

    def copy(self):
        """
        Make a copy of the object

        :rtype: BaseMeshGenerator
        """
        cpy = super(BaseMeshGenerator, self).copy()
        cpy.modelParameters = self.modelParameters  # see https://github.com/3ptscience/properties/issues/175
        return cpy


class TensorMeshGenerator(BaseMeshGenerator):
    """
    Tensor mesh designed based on the source and formulation
    """

    # cell sizes in each of the dimensions
    csx = properties.Float(
        "cell size in the x-direction", default=25.
    )
    csy = properties.Float(
        "cell size in the y-direction", default=25.
    )
    csz = properties.Float(
        "cell size in the z-direction", default=25.
    )

    # padding factors in each direction
    pfx = properties.Float(
        "padding factor to pad to infinity", default=1.5
    )
    pfy = properties.Float(
        "padding factor to pad to infinity", default=1.5
    )
    pfz = properties.Float(
        "padding factor to pad to infinity", default=1.5
    )

    # number of extra cells horizontally, above the air-earth interface and
    # below the casing
    nch = properties.Integer(
        "number of cells to add on each side of the mesh horizontally",
        default=10
    )
    nca = properties.Integer(
        "number of extra cells above the air-earth interface",
        default=5
    )
    ncb = properties.Integer(
        "number of cells below the casing",
        default=5
    )

    # number of padding cells in each direction
    npadx = properties.Integer(
        "number of x-padding cells", default=10
    )
    npady = properties.Integer(
        "number of y-padding cells", default=10
    )
    npadz = properties.Integer(
        "number of z-padding cells", default=10
    )

    # domain extent in the y-direction
    domain_x = properties.Float(
        "domain extent in the x-direction", default=1000.
    )
    domain_y = properties.Float(
        "domain extent in the y-direction", default=1000.
    )

    # Instantiate the class with casing parameters
    def __init__(self, **kwargs):
        super(TensorMeshGenerator, self).__init__(**kwargs)
        self._discretizePair = discretize.TensorMesh

    @property
    def x0(self):
        """
        Origin of the mesh

        :rtype: numpy.array
        """
        if getattr(self, '_x0', None) is None:
            self._x0 = np.r_[
                (
                    -self.hx.sum()/2. +
                    (
                        self.modelParameters.src_b[0] +
                        self.modelParameters.src_a[0]
                    )/2.
                ),
                -self.hy.sum()/2.,
                -self.hz[:self.npadz+self.ncz-self.nca].sum()
            ]
        return self._x0

    @x0.setter
    def x0(self, value):
        assert len(value) == 3, (
            'length of x0 must be 3, not {}'.format(len(x0))
        )

        self._x0 = value

    @property
    def domain_z(self):
        """
        vertical extent of the mesh

        :rtype: float
        """
        if getattr(self, '_domain_z', None) is None:
            if getattr(self.modelParameters, 'casing_z', None) is not None:
                domain_z = max([
                    np.absolute(
                        self.modelParameters.casing_z[1] -
                        self.modelParameters.casing_z[0]
                    ),
                    np.absolute(
                        self.modelParameters.src_b[2] -
                        self.modelParameters.src_a[2]
                    )
                ])
            else:
                domain_z = np.absolute(
                    self.modelParameters.src_b[2] -
                    self.modelParameters.src_a[2]
                )
            self._domain_z = domain_z
        return self._domain_z

    @domain_z.setter
    def domain_z(self, value):
        self._domain_z = value

    # number of cells in each direction
    @property
    def ncx(self):
        """
        number of x-cells

        :rtype: int
        """
        if getattr(self, '_ncx', None) is None:
            self._ncx = int(
                np.ceil(self.domain_x / self.csx) +
                2*self.nch
            )
        return self._ncx

    @property
    def ncy(self):
        """
        number of y-cells

        :rtype: int
        """
        if getattr(self, '_ncy', None) is None:
            self._ncy = int(
                np.ceil(self.domain_y / self.csy) + 2*self.nch
            )
        return self._ncy

    @property
    def ncz(self):
        """
        number of z-cells

        :rtype: int
        """
        if getattr(self, '_ncz', None) is None:
            self._ncz = int(
                np.ceil(self.domain_z / self.csz) + self.nca + self.ncb
            )
        return self._ncz

    # cell spacings in each direction
    @property
    def hx(self):
        """
        vector of cell spacings in the x-direction

        :rtype: numpy.array
        """
        if getattr(self, '_hx', None) is None:
            self._hx = utils.meshTensor([
                (self.csx, self.npadx, -self.pfx),
                (self.csx, self.ncx),
                (self.csx, self.npadx, self.pfx)
            ])
        return self._hx

    @property
    def hy(self):
        """
        vector of cell spacings in the y-direction

        :rtype: numpy.array
        """
        if getattr(self, '_hy', None) is None:
            self._hy = utils.meshTensor([
                (self.csy, self.npady, -self.pfy),
                (self.csy, self.ncy),
                (self.csy, self.npady, self.pfy)
            ])
        return self._hy

    @property
    def hz(self):
        """
        vector of cell spacings in the z-direction

        :rtype: numpy.array
        """
        if getattr(self, '_hz', None) is None:
            self._hz = utils.meshTensor([
                (self.csz, self.npadz, -self.pfz),
                (self.csz, self.ncz),
                (self.csz, self.npadz, self.pfz)
            ])
        return self._hz


class BaseCylMixin(properties.HasProperties):
    """
    Mixin class that contains properties and methods common to a Cyl Mesh
    Generator
    """

    # cell sizes in the vertical direction
    csz = properties.Float(
        "cell size in the z-direction", default=25.
    )

    # Theta direction of the mesh
    hy = properties.Array(
        "cell spacings in the y direction",
        dtype=float,
        default=np.r_[2*np.pi]  # default is cyl symmetric
    )

    # z-direction of the mesh
    nca = properties.Integer(
        "number of fine cells above the air-earth interface", default=5
    )
    ncb = properties.Integer(
        "number of fine cells below the casing", default=5
    )
    pfz = properties.Float(
        "padding factor in the z-direction", default=1.5
    )

    # number of padding cells
    npadx = properties.Integer(
        "number of padding cells required to get to infinity!", default=23
    )
    npadz = properties.Integer(
        "number of padding cells in z", default=38
    )

    # domain extent in the x-direction
    domain_x = properties.Float(
        "domain extent in the x-direction", default=1000.
    )

    @property
    def x0(self):
        """
        Origin of the mesh
        """
        if getattr(self, '_x0', None) is None:
            self._x0 = np.r_[
                0., 0., -np.sum(self.hz[:self.npadz+self.ncz-self.nca])
            ]
        return self._x0

    @property
    def domain_z(self):
        """
        z-extent extent of the core mesh
        """
        if getattr(self, '_domain_z', None) is None:
            if getattr(self.modelParameters, 'casing_z', None) is not None:
                domain_z = max([
                    np.absolute(
                        self.modelParameters.casing_z[1] -
                        self.modelParameters.casing_z[0]
                    ),
                    np.absolute(
                        self.modelParameters.src_b[2] -
                        self.modelParameters.src_a[2]
                    )
                ])
            else:
                domain_z = np.absolute(
                    self.modelParameters.src_b[2] -
                    self.modelParameters.src_a[2]
                )
            self._domain_z = domain_z
        return self._domain_z

    @domain_z.setter
    def domain_z(self, value):
        self._domain_z = value

    @properties.observer('hy')
    def _ensure_2pi(self, change):
        value = change['value']
        assert np.absolute(value.sum() - 2*np.pi) < 1e-6

    @property
    def ncy(self):
        """
        number of core y-cells

        :rtype: float
        """
        if getattr(self, '_ncz', None) is None:
            self._ncy = len(self.hy)
        return self.ncy

    @property
    def ncz(self):
        """
        number of core z-cells

        :rtype: float
        """
        if getattr(self, '_ncz', None) is None:
            # number of core z-cells (add 10 below the end of the casing)
            self._ncz = (
                np.int(np.ceil(self.domain_z/self.csz)) +
                self.nca + self.ncb
            )
        return self._ncz

    @property
    def hz(self):
        """
        cell spacings in the z-direction

        :rtype: numpy.array
        """
        if getattr(self, '_hz', None) is None:

            self._hz = utils.meshTensor([
                (self.csz, self.npadz, -self.pfz),
                (self.csz, self.ncz),
                (self.csz, self.npadz, self.pfz)
            ])
        return self._hz

    def create_2D_mesh(self):
        """
        create cylindrically symmetric mesh generator
        """
        mesh2D = self.copy()
        # mesh2D.modelParameters = self.modelParameters.copy()  # see https://github.com/3ptscience/properties/issues/175
        mesh2D.hy = np.r_[2*np.pi]
        return mesh2D

    # Plot the physical Property Models
    def plotModels(
        self, sigma, mu, xlim=[0., 1.], zlim=[-1200., 100.], ax=None
    ):
        """
        Plot conductivity and permeability models
        """
        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        plt.colorbar(
            self.mesh.plotImage(np.log10(sigma), ax=ax[0])[0], ax=ax[0]
        )
        plt.colorbar(
            self.mesh.plotImage(mu/mu_0, ax=ax[1])[0], ax=ax[1]
        )

        ax[0].set_xlim(xlim)
        ax[1].set_xlim(xlim)

        ax[0].set_ylim(zlim)
        ax[1].set_ylim(zlim)

        ax[0].set_title('$\log_{10}\sigma$')
        ax[1].set_title('$\mu_r$')

        plt.tight_layout()

        return ax


class CylMeshGenerator(BaseMeshGenerator, BaseCylMixin):
    """
    Simple 3D cylindrical mesh
    """
    csx = properties.Float(
        "cell size in the x-direction", default=25.
    )

    # padding factors in each direction
    pfx = properties.Float(
        "padding factor to pad to infinity", default=1.5
    )

    # number of extra cells horizontally, above the air-earth interface and
    # below the casing
    nch = properties.Integer(
        "number of cells to add on each side of the mesh horizontally",
        default=10.
    )

    # Instantiate the class with casing parameters
    def __init__(self, **kwargs):
        super(CylMeshGenerator, self).__init__(**kwargs)
        self._discretizePair = discretize.CylMesh

    # number of cells in each direction
    @property
    def ncx(self):
        if getattr(self, '_ncx', None) is None:
            self._ncx = int(
                np.ceil(self.domain_x / self.csx) +
                self.nch
            )
        return self._ncx

    # cell spacings in each direction
    @property
    def hx(self):
        if getattr(self, '_hx', None) is None:
            self._hx = utils.meshTensor([
                (self.csx, self.ncx),
                (self.csx, self.npadx, self.pfx)
            ])
        return self._hx


class CasingMeshGenerator(BaseMeshGenerator, BaseCylMixin):
    """
    Mesh that makes sense for casing examples
    """

    # X-direction of the mesh
    csx1 = properties.Float(
        "finest cells in the x-direction", default=2.5e-3
    )
    csx2 = properties.Float(
        "second uniform cell region in x-direction", default=25.
    )
    pfx1 = properties.Float(
        "padding factor to pad from csx1 to csx2", default=1.3
    )
    pfx2 = properties.Float(
        "padding factor to pad to infinity", default=1.5
    )

    # Instantiate the class with casing parameters
    def __init__(self, **kwargs):
        super(CasingMeshGenerator, self).__init__(**kwargs)
        self._discretizePair = discretize.CylMesh

    @property
    def hx(self):
        """
        cell spacings in the x-direction
        """
        if getattr(self, '_hx', None) is None:

            self._hx = pad_for_casing_and_data(
                self.modelParameters.casing_b,
                self.csx1,
                self.csx2,
                self.pfx1,
                self.pfx2,
                self.domain_x,
                self.npadx
            )

        return self._hx

