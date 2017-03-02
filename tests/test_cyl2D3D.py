import unittest
import discretize
from discretize import utils
import numpy as np
import scipy.sparse as sp
import sympy

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from scipy.constants import mu_0

from SimPEG.EM import FDEM
from SimPEG import Utils, Maps

from pymatsolver import Pardiso

import CasingSimulations


plotIt = False
TOL = 1e-4
ZERO = 1e-7


def getSrcWire(mesh, cp):
    """
    Define a wirepath on the mesh

    :param discretize.BaseMesh mesh: mesh on which to define the source
    :param CasingSimulations cp: casing parameters
    :rtype: numpy.ndarray
    :return: source current density on the mesh
    """
    wire = np.zeros(mesh.vnF[2])

    xfaces = mesh.gridFz[:, 0] < mesh.hx.min()
    zfaces = (
        (mesh.gridFz[:, 2] > cp.src_a[2]) &
        (mesh.gridFz[:, 2] < cp.src_b[2])
    )
    wire[xfaces & zfaces] = 1

    return np.hstack([np.zeros(mesh.nFx), np.zeros(mesh.nFy), wire])


def getPhysProps(mesh, cp):
    """
    Put the phys prop models on the mesh

    :param discretize.BaseMesh mesh: simulation mesh
    :param CasingSimulations cp: casing parameters
    :rtype: tuple
    :return: (sigma, mu) on mesh
    """

    casing_x = (
        (mesh.gridCC[:, 0] > cp.casing_a) & (mesh.gridCC[:, 0] < cp.casing_b)
    )
    casing_z = (
        (mesh.gridCC[:, 2] > cp.casing_z[0]) &
        (mesh.gridCC[:, 2] < cp.casing_z[1])
    )

    inside_x = (mesh.gridCC[:, 0] < cp.casing_a)

    sigma = cp.sigma_back * np.ones(mesh.nC)
    sigma[casing_x & casing_z] = cp.sigma_casing
    sigma[inside_x & casing_z] = cp.sigma_inside

    mu = np.ones(mesh.nC)
    mu[casing_x & casing_z] = cp.mu_casing
    mu = mu * mu_0

    return sigma, mu


class Test2Dv3DCyl(unittest.TestCase):
    """
    Tests to make sure that the 2D and 3D cyl meshes produce consistent
    results for cylindrically symmetric problems.
    """

    @classmethod
    def setUpClass(self):
        """
        Set up a cyl symmetric EM problem on 2D and 3D meshes.
        """

        sigma_back = 1e-1 # wholespace

        cp = CasingSimulations.CasingProperties(
            casing_l = 10.,
            src_a = np.r_[0., 0., -9.],
            src_b = np.r_[0., 0., -1.],
            freqs = np.r_[0.1, 1., 2.],
            sigma_back = sigma_back, # wholespace
            sigma_layer = sigma_back,
            sigma_air = sigma_back,

        )

        # Set up the meshes
        npadx, npadz = 11, 26
        dx2 = 500.

        mesh2D = CasingSimulations.CasingMesh(
            cp=cp, npadx=npadx, npadz=npadz, dx2=dx2
        ).mesh
        mesh3D = CasingSimulations.CasingMesh(
            cp=cp, ncy=4, npadx=npadx, npadz=npadz, dx2=dx2
        ).mesh

        # get wirepath on mesh
        wire2D = getSrcWire(mesh2D, cp)
        wire3D = getSrcWire(mesh3D, cp)

        # create sources
        srcList2D = [
            FDEM.Src.RawVec_e(s_e=wire2D, freq=freq, rxList=[]) for freq in
            cp.freqs
        ]
        srcList3D = [
            FDEM.Src.RawVec_e(s_e=wire3D, freq=freq, rxList=[]) for freq in
            cp.freqs
        ]

        # get phys prop models
        sigma2D, mu2D = getPhysProps(mesh2D, cp)
        sigma3D, mu3D = getPhysProps(mesh3D, cp)

        # plot the phys prop models
        fig, ax = plt.subplots(1, 1)
        plt.colorbar(
            mesh2D.plotImage(np.log10(sigma2D), ax=ax, mirror=True)[0], ax=ax
        )
        ax.set_xlim([-1., 1.])
        ax.set_ylim([-20., 10.])

        if plotIt:
            plt.show()

        # use wires to have 2 active models
        wires2D = Maps.Wires(('sigma', mesh2D.nC), ('mu', mesh2D.nC))
        wires3D = Maps.Wires(('sigma', mesh3D.nC), ('mu', mesh3D.nC))

        # create the problems and surveys
        prb2D = FDEM.Problem3D_h(
            mesh2D, sigmaMap=wires2D.sigma, muMap=wires2D.mu, Solver=Pardiso
        )
        prb3D = FDEM.Problem3D_h(
            mesh3D, sigmaMap=wires3D.sigma, muMap=wires3D.mu, Solver=Pardiso
        )

        survey2D = FDEM.Survey(srcList2D)
        survey3D = FDEM.Survey(srcList3D)

        prb2D.pair(survey2D)
        prb3D.pair(survey3D)

        print('starting 2D solve ... ')
        fields2D = prb2D.fields(np.hstack([sigma2D, mu2D]))
        print('  ... done \n')
        print('starting 3D solve ...')
        fields3D = prb3D.fields(np.hstack([sigma3D, mu3D]))
        print('  ... done \n')

        # assign the properties that will be helpful
        self.mesh2D = mesh2D
        self.mesh3D = mesh3D

        self.srcList2D = srcList2D
        self.srcList3D = srcList3D

        self.prb2D = prb2D
        self.prb3D = prb3D

        self.survey2D = survey2D
        self.survey3D = survey3D

        self.fields2D = fields2D
        self.fields3D = fields3D

    def getj3Dthetaslice(self, j3D, theta_ind=0):
        """
        grab theta slice through j
        """
        j3D_x = j3D[:self.mesh3D.nFx].reshape(self.mesh3D.vnFx, order='F')
        j3D_z = j3D[self.mesh3D.vnF[:2].sum():].reshape(
            self.mesh3D.vnFz, order='F'
        )

        j3Dslice = np.vstack([
            utils.mkvc(j3D_x[:, theta_ind, :], 2),
            utils.mkvc(j3D_z[:, theta_ind, :], 2)
        ])

        return j3Dslice

    def geth3Dthetaslice(self, h3D, theta_ind=0):
        """
        grab theta slice through h
        """
        h3D_y = h3D[self.mesh3D.vnE[0]:self.mesh3D.vnE[:2].sum()].reshape(
            self.mesh3D.vnEy, order='F'
        )

        return utils.mkvc(h3D_y[:, theta_ind, :], 2)

    def test_j_cyl3Dsymmetry(self):
        """
        The problem is symmetric, so the current density in each quadrent of
        the 3D cyl mesh should be the same
        """

        for src in self.srcList3D:

            print('\nTesting 3D j symmetry ({} Hz)'.format(src.freq))

            j3D = self.fields3D[src, 'j']
            j0 = self.getj3Dthetaslice(j3D)

            norm_j0 = np.linalg.norm(j0)

            for ind in np.arange(1, self.mesh3D.vnC[1]):
                diff = np.linalg.norm(
                        j0-self.getj3Dthetaslice(j3D, theta_ind=ind)
                    )
                err = diff / norm_j0
                passed = err < TOL

                print(
                    '|j0 - j{}|: {:1.4e}, |j0|: {:1.4e},  '
                    '{:1.4e} < {:1.1e} ? {}'.format(
                        ind, diff, norm_j0, err, TOL, passed
                    )
                )

                self.assertTrue(passed)

    def test_j_thetasymmetry(self):
        """
        j_theta should be close to zero
        """

        for src in self.srcList3D:

            print(
                '\nTesting that |j_theta| < {} ({}) Hz'.format(
                    ZERO, src.freq
                )
            )

            j3D = self.fields3D[src, 'j']
            jtheta = j3D[self.mesh3D.nFx:self.mesh3D.nFz]

            self.assertTrue(np.all(jtheta < ZERO))

            print(' ... ok')

    def test_h_rzsymmetry(self):
        """
        h_r, h_z should be close to zero
        """

        for src in self.srcList3D:

            print(
                '\nTesting that |h_r, h_z| < {} ({}) Hz'.format(
                    ZERO, src.freq
                )
            )

            h3D = self.fields3D[src, 'h']
            hr = h3D[:self.mesh3D.nEx]
            hz = h3D[self.mesh3D.vnE[:2].sum():]

            self.assertTrue(np.all(hr < ZERO))
            self.assertTrue(np.all(hz < ZERO))

    def test_h_cyl3Dsymmetry(self):
        """
        The problem is symmetric, so the magnetic field in each quadrent of
        the 3D cyl mesh should be the same
        """

        for src in self.srcList3D:

            print('\nTesting 3D h symmetry ({} Hz)'.format(src.freq))

            h3D = self.fields3D[src, 'h']
            h0 = self.geth3Dthetaslice(h3D)

            norm_h0 = np.linalg.norm(h0)

            for ind in np.arange(1, self.mesh3D.vnC[1]):
                diff = np.linalg.norm(
                        h0-self.geth3Dthetaslice(h3D, theta_ind=ind)
                    )
                err = diff / norm_h0
                passed = err < TOL

                print(
                    '|h0 - h{}|: {:1.4e}, |h0|: {:1.4e},  '
                    '{:1.4e} < {:1.1e} ? {}'.format(
                        ind, diff, norm_h0, err, TOL, passed
                    )
                )

                self.assertTrue(passed)

    def test_j_2Dv3D(self):
        """
        test that the 2D and 3D solutions agree
        """
        print('\nTesting 2D v 3D j symmetry')

        for src2D, src3D in zip(self.srcList2D, self.srcList3D):

            j2D = self.fields2D[src2D, 'j']
            j3D = self.fields3D[src3D, 'j']

            j3D_slice = self.getj3Dthetaslice(j3D)

            diff = np.linalg.norm(j3D_slice - j2D)
            norm_j2D = np.linalg.norm(j2D)

            err = diff/norm_j2D
            passed = err < TOL

            print(
                '  {} Hz.  |j3D - j2D|: {:1.4e}, |j3D - j2D|/|j2D|: {:1.4e} '
                '< {} ? {}'.format(
                    src3D.freq, diff, err, TOL, passed
                )
            )

            self.assertTrue(passed)

    def test_h_2Dv3D(self):
        """
        make sure h2D and h3D solutions agree
        """

        print('\nTesting 2D v 3D h symmetry')

        for src2D, src3D in zip(self.srcList2D, self.srcList3D):

            h2D = self.fields2D[src2D, 'h']
            h3D = self.fields3D[src3D, 'h']

            h3D_slice = self.geth3Dthetaslice(h3D)

            diff = np.linalg.norm(h3D_slice - h2D)
            norm_h2D = np.linalg.norm(h2D)

            err = diff/norm_h2D
            passed = err < TOL

            print(
                '  {} Hz.  |h3D - h2D|: {:1.4e}, |h3D - h2D|/|h2D|: {:1.4e} '
                '< {} ? {}'.format(
                    src3D.freq, diff, err, TOL, passed
                )
            )

            self.assertTrue(passed)


if __name__ == '__main__':
    unittest.main()
