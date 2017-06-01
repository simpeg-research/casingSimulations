import numpy as np
import properties
import json
import os
import matplotlib.pyplot as plt
import inspect

import properties
from SimPEG import Utils

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


class BaseMeshGenerator(BaseCasing):
    """
    Base Mesh Generator Class
    """

    filename = properties.String(
        "filename to serialize properties to",
        default="MeshParameters.json"
    )

    # casing parameters
    cp = properties.Instance(
        "casing parameters instance",
        model.Wholespace,
        required=True
    )

    def __init__(self, **kwargs):
        Utils.setKwargs(self, **kwargs)

    @property
    def mesh(self):
        if getattr(self, '_mesh', None) is None:
            self._mesh = self._discretizePair(
                [self.hx, self.hy, self.hz],
                x0=self.x0
            )
        return self._mesh

    def copy(self):
        cpy = super(BaseMeshGenerator, self).copy()
        cpy.cp = self.cp  # see https://github.com/3ptscience/properties/issues/175
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
        if getattr(self, '_x0', None) is None:
            self._x0 = np.r_[
                -self.hx.sum()/2. + (self.cp.src_b[0] + self.cp.src_a[0])/2.,
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
        if getattr(self, '_domain_z', None) is None:
            if getattr(self.cp, 'casing_z', None) is not None:
                domain_z = max([
                    (self.cp.casing_z[1] - self.cp.casing_z[0]),
                    (self.cp.src_b[2] - self.cp.src_a[2])
                ])
            else:
                domain_z = (self.cp.src_b[2] - self.cp.src_a[2])
            self._domain_z = domain_z
        return self._domain_z

    @domain_z.setter
    def domain_z(self, value):
        self._domain_z = value

    # number of cells in each direction
    @property
    def ncx(self):
        if getattr(self, '_ncx', None) is None:
            self._ncx = int(
                np.ceil(self.domain_x / self.csx) +
                2*self.nch
            )
        return self._ncx

    @property
    def ncy(self):
        if getattr(self, '_ncy', None) is None:
            self._ncy = int(
                np.ceil(self.domain_y / self.csy) + 2*self.nch
            )
        return self._ncy

    @property
    def ncz(self):
        if getattr(self, '_ncz', None) is None:
            self._ncz = int(
                np.ceil(self.domain_z / self.csz) + self.nca + self.ncb
            )
        return self._ncz

    # cell spacings in each direction
    @property
    def hx(self):
        if getattr(self, '_hx', None) is None:
            self._hx = utils.meshTensor([
                (self.csx, self.npadx, -self.pfx),
                (self.csx, self.ncx),
                (self.csx, self.npadx, self.pfx)
            ])
        return self._hx

    @property
    def hy(self):
        if getattr(self, '_hy', None) is None:
            self._hy = utils.meshTensor([
                (self.csy, self.npady, -self.pfy),
                (self.csy, self.ncy),
                (self.csy, self.npady, self.pfy)
            ])
        return self._hy

    @property
    def hz(self):
        if getattr(self, '_hz', None) is None:
            self._hz = utils.meshTensor([
                (self.csz, self.npadz, -self.pfz),
                (self.csz, self.ncz),
                (self.csz, self.npadz, self.pfz)
            ])
        return self._hz


class CylMeshGenerator(BaseMeshGenerator):
    """
    Simple 3D cylindrical mesh

    """
    csx = properties.Float(
        "cell size in the x-direction", default=25.
    )
    csz = properties.Float(
        "cell size in the z-direction", default=25.
    )

    # padding factors in each direction
    pfx = properties.Float(
        "padding factor to pad to infinity", default=1.5
    )
    pfz = properties.Float(
        "padding factor to pad to infinity", default=1.5
    )

    # Theta direction of the mesh
    hy = properties.Array(
        "cell spacings in the y direction",
        dtype=float,
        default=np.r_[2*np.pi] # default is cyl symmetric
    )

    # number of extra cells horizontally, above the air-earth interface and
    # below the casing
    nch = properties.Integer(
        "number of cells to add on each side of the mesh horizontally",
        default=10.
    )
    nca = properties.Integer(
        "number of extra cells above the air-earth interface",
        default=5.
    )
    ncb = properties.Integer(
        "number of cells below the casing",
        default=5.
    )

    # number of padding cells in each direction
    npadx = properties.Integer(
        "number of x-padding cells", default=10
    )
    npadz = properties.Integer(
        "number of z-padding cells", default=10
    )

    # domain extent in the y-direction
    domain_x = properties.Float(
        "domain extent in the x-direction", default=1000.
    )

    # Instantiate the class with casing parameters
    def __init__(self, **kwargs):
        super(CylMeshGenerator, self).__init__(**kwargs)
        self._discretizePair = discretize.CylMesh

    @property
    def x0(self):
        if getattr(self, '_x0', None) is None:
            self._x0 = np.r_[
                0., 0., -np.sum(self.hz[:self.npadz+self.ncz-self.nca])
            ]
        return self._x0

    @property
    def domain_z(self):
        if getattr(self, '_domain_z', None) is None:
            if getattr(self.cp, 'casing_z', None) is not None:
                domain_z = max([
                    (self.cp.casing_z[1] - self.cp.casing_z[0]),
                    (self.cp.src_b[2] - self.cp.src_a[2])
                ])
            else:
                domain_z = (self.cp.src_b[2] - self.cp.src_a[2])
        self._domain_z = domain_z
        return self._domain_z

    @domain_z.setter
    def domain_z(self, value):
        self._domain_z = value

    # number of cells in each direction
    @property
    def ncx(self):
        if getattr(self, '_ncx', None) is None:
            self._ncx = int(
                np.ceil(self.domain_x / self.csx) +
                self.nch
            )
        return self._ncx

    @property
    def ncy(self):
        if getattr(self, '_ncy', None) is None:
            self._ncy = int(
                np.ceil(self.domain_y / self.csy) + 2*self.nch
            )
        return self._ncy

    @property
    def ncz(self):
        if getattr(self, '_ncz', None) is None:
            self._ncz = int(
                np.ceil(self.domain_z / self.csz) + self.nca + self.ncb
            )
        return self._ncz

    # cell spacings in each direction
    @property
    def hx(self):
        if getattr(self, '_hx', None) is None:
            self._hx = utils.meshTensor([
                (self.csx, self.ncx),
                (self.csx, self.npadx, self.pfx)
            ])
        return self._hx

    @property
    def hz(self):
        if getattr(self, '_hz', None) is None:
            self._hz = utils.meshTensor([
                (self.csz, self.npadz, -self.pfz),
                (self.csz, self.ncz),
                (self.csz, self.npadz, self.pfz)
            ])
        return self._hz

    def create_2D_mesh(self):
        mesh2D = self.copy()
        mesh2D.cp = self.cp  # see https://github.com/3ptscience/properties/issues/175
        mesh2D.hy = np.r_[2*np.pi]
        return mesh2D


class CasingMeshGenerator(BaseMeshGenerator):
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
    domain_x2 = properties.Float(
        "domain extent for uniform cell region", default=1000.
    )

    # Theta direction of the mesh
    hy = properties.Array(
        "cell spacings in the y direction",
        dtype=float,
        default=np.r_[2*np.pi] # default is cyl symmetric
    )

    # z-direction of the mesh
    csz = properties.Float(
        "cell size in the z-direction", default=0.05
    )
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

    # Instantiate the class with casing parameters
    def __init__(self, **kwargs):
        super(CasingMeshGenerator, self).__init__(**kwargs)
        self._discretizePair = discretize.CylMesh

    @property
    def ncx1(self):
        """number of cells with size csx1"""
        return np.ceil(self.cp.casing_b/self.csx1+2)

    @property
    def npadx1(self):
        """number of padding cells to get from csx1 to csx2"""
        return np.floor(np.log(self.csx2/self.csx1) / np.log(self.pfx1))

    @property
    def hx(self):
        """
        cell spacings in the x-direction
        """
        if getattr(self, '_hx', None) is None:

            # finest uniform region
            hx1a = Utils.meshTensor([(self.csx1, self.ncx1)])

            # pad to second uniform region
            hx1b = Utils.meshTensor([(self.csx1, self.npadx1, self.pfx1)])

            # scale padding so it matches cell size properly
            dx1 = sum(hx1a)+sum(hx1b)
            dx1 = np.floor(dx1/self.csx2)
            hx1b *= (dx1*self.csx2 - sum(hx1a))/sum(hx1b)

            # second uniform chunk of mesh
            ncx2 = np.ceil((self.domain_x2 - dx1)/self.csx2)
            hx2a = Utils.meshTensor([(self.csx2, ncx2)])

            # pad to infinity
            hx2b = Utils.meshTensor([(self.csx2, self.npadx, self.pfx2)])

            self._hx = np.hstack([hx1a, hx1b, hx2a, hx2b])

        return self._hx

    @properties.observer('hy')
    def _ensure_2pi(self, change):
        value = change['value']
        assert np.absolute(value.sum() - 2*np.pi) < 1e-6

    @property
    def ncy(self):
        if getattr(self, '_ncz', None) is None:
            self._ncy = len(self.hy)
        return self.ncy


    @property
    def ncz(self):
        """
        number of core z-cells
        """
        if getattr(self, '_ncz', None) is None:
            # number of core z-cells (add 10 below the end of the casing)
            self._ncz = (
                np.int(np.ceil(-self.cp.casing_z[0]/self.csz)) +
                self.nca + self.ncb
            )
        return self._ncz

    @property
    def hz(self):
        if getattr(self, '_hz', None) is None:

            self._hz = Utils.meshTensor([
                (self.csz, self.npadz, -self.pfz),
                (self.csz, self.ncz),
                (self.csz, self.npadz, self.pfz)
            ])
        return self._hz

    @property
    def x0(self):
        if getattr(self, '_x0', None) is None:
            self._x0 = np.r_[
                0., 0., -np.sum(self.hz[:self.npadz+self.ncz-self.nca])
            ]
        return self._x0

    @x0.setter
    def x0(self, value):
        assert len(value) == 3, 'x0 must be length 3, not {}'.format(len(x0))

    def create_2D_mesh(self):
        mesh2D = self.copy()
        mesh2D.cp = self.cp  # see https://github.com/3ptscience/properties/issues/175
        mesh2D.hy = np.r_[2*np.pi]
        return mesh2D

    # Plot the physical Property Models
    def plotModels(self, sigma, mu, xlim=[0., 1.], zlim=[-1200., 100.], ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        plt.colorbar(self.mesh.plotImage(np.log10(sigma), ax=ax[0])[0], ax=ax[0])
        plt.colorbar(self.mesh.plotImage(mu/mu_0, ax=ax[1])[0], ax=ax[1])

        ax[0].set_xlim(xlim)
        ax[1].set_xlim(xlim)

        ax[0].set_ylim(zlim)
        ax[1].set_ylim(zlim)

        ax[0].set_title('$\log_{10}\sigma$')
        ax[1].set_title('$\mu_r$')

        plt.tight_layout()

        return ax

