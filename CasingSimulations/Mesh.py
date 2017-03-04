import numpy as np
import properties
import json
import os

import discretize as Mesh
from SimPEG import Utils

from discretize.utils import mkvc


class CasingMesh(properties.HasProperties):
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
    dx2 = properties.Float(
        "domain extent for uniform cell region", default=1000.
    )
    npadx = properties.Integer(
        "number of padding cells required to get to infinity!", default=23
    )

    # Theta direction of the mesh
    ncy = properties.Integer(
        "number of cells in the theta direction of the mesh. "
        "1 --> cyl symmetric", default=1
    )

    # Z-direction of mesh
    csz = properties.Float(
        "cell size in the z-direction", default=0.05
    )
    nza = properties.Integer(
        "number of fine cells above the air-earth interface", default=10
    )
    pfz = properties.Float(
        "padding factor in the z-direction", default=1.5
    )
    npadz = properties.Integer(
        "number of padding cells in z", default=38
    )

    # Instantiate the class with casing parameters
    def __init__(self, cp, **kwargs):
        Utils.setKwargs(self, **kwargs)
        self.cp = cp

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
            ncx2 = np.ceil((self.dx2 - dx1)/self.csx2)
            hx2a = Utils.meshTensor([(self.csx2, ncx2)])

            # pad to infinity
            hx2b = Utils.meshTensor([(self.csx2, self.npadx, self.pfx2)])

            self._hx = np.hstack([hx1a, hx1b, hx2a, hx2b])

        return self._hx

    @property
    def hy(self):
        """
        cell spacings in the y-direction
        """
        if getattr(self, '_hy', None) is None:
            if self.ncy == 1:
                self._hy = 1
            else:
                self._hy = 2*np.pi * np.ones(self.ncy) / self.ncy
        return self._hy

    @hy.setter
    def hy(self, val):
        H = val.sum()
        if H != 2*np.pi:
            val = val*2*np.pi/val.sum()
        self._hy = val

    @property
    def ncz(self):
        """
        number of core z-cells
        """
        if getattr(self, '_ncz', None) is None:
            # number of core z-cells (add 10 below the end of the casing)
            self._ncz = (
                np.int(np.ceil(-self.cp.casing_z[0]/self.csz))+10
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
    def mesh(self):
        if getattr(self, '_mesh', None) is None:
            self._mesh = Mesh.CylMesh(
                [self.hx, self.hy, self.hz],
                [0., 0., -np.sum(self.hz[:self.npadz+self.ncz-self.nza])]
            )
        return self._mesh

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

    def save(self, filename='MeshParameters.json', directory='.'):
        """
        Save the casing mesh parameters to json
        :param str file: filename for saving the casing mesh parameters
        """
        if not os.path.isdir(directory):  # check if the directory exists
            os.mkdir(directory)  # if not, create it
        f = '/'.join([directory, filename])
        with open(f, 'w') as outfile:
            cp = json.dump(self.serialize(), outfile)


# grab 2D slices
def face3DthetaSlice(mesh3D, j3D, theta_ind=0):
    """
    Grab a theta slice through a 3D field defined on faces
    (x, z components), consistent with what would be found from a
    2D simulation

    :param discretize.CylMesh mesh3D: 3D cyl mesh
    :param numpy.ndarray j3D: vector of fluxes on mesh
    :param int theta_ind: index of the theta slice that you want
    """
    j3D_x = j3D[:mesh3D.nFx].reshape(mesh3D.vnFx, order='F')
    j3D_z = j3D[mesh3D.vnF[:2].sum():].reshape(mesh3D.vnFz, order='F')

    j3Dslice = np.vstack([
        utils.mkvc(j3D_x[:, theta_ind, :], 2),
        utils.mkvc(j3D_z[:, theta_ind, :], 2)
    ])

    return j3Dslice


def edge3DthetaSlice(mesh3D, h3D, theta_ind=0):
    """
    Grab a theta slice through a 3D field defined on edges
    (y component), consistent with what would be found from a
    2D simulation

    :param discretize.CylMesh mesh3D: 3D cyl mesh
    :param numpy.ndarray h3D: vector of fields on mesh
    :param int theta_ind: index of the theta slice that you want
    """

    h3D_y = h3D[mesh3D.nEx:mesh3D.vnE[:2].sum()].reshape(
        mesh3D.vnEy, order='F'
    )

    return mkvc(h3D_y[:, theta_ind, :])
