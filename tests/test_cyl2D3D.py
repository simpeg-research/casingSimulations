import unittest
import discretize
from discretize import utils
import numpy as np
import scipy.sparse as sp
import sympy

from scipy.constants import mu_0

from SimPEG.electromagnetics import frequency_domain as fdem
from SimPEG import utils, maps

from pymatsolver import Pardiso

import casingSimulations

plotIt = False
TOL = 1e-4
ZERO = 1e-7


def getSrcWire(mesh, modelParameters):
    """
    Define a wirepath on the mesh

    :param discretize.BaseMesh mesh: mesh on which to define the source
    :param casingSimulations modelParameters: casing parameters
    :rtype: numpy.ndarray
    :return: source current density on the mesh
    """
    wire = np.zeros(mesh.vnF[2])

    xfaces = mesh.gridFz[:, 0] < mesh.hx.min()
    zfaces = (
        (mesh.gridFz[:, 2] > modelParameters.src_a[2]) &
        (mesh.gridFz[:, 2] < modelParameters.src_b[2])
    )
    wire[xfaces & zfaces] = 1

    return np.hstack([np.zeros(mesh.nFx), np.zeros(mesh.nFy), wire])


def getPhysProps(mesh, modelParameters):
    """
    Put the phys prop models on the mesh

    :param discretize.BaseMesh mesh: simulation mesh
    :param casingSimulations modelParameters: casing parameters
    :rtype: tuple
    :return: (sigma, mu) on mesh
    """

    casing_x = (
        (mesh.gridCC[:, 0] > modelParameters.casing_a) &
        (mesh.gridCC[:, 0] < modelParameters.casing_b)
    )
    casing_z = (
        (mesh.gridCC[:, 2] > modelParameters.casing_z[0]) &
        (mesh.gridCC[:, 2] < modelParameters.casing_z[1])
    )

    inside_x = (mesh.gridCC[:, 0] < modelParameters.casing_a)

    sigma = modelParameters.sigma_back * np.ones(mesh.nC)
    sigma[casing_x & casing_z] = modelParameters.sigma_casing
    sigma[inside_x & casing_z] = modelParameters.sigma_inside

    mu = np.ones(mesh.nC)
    mu[casing_x & casing_z] = modelParameters.mu_casing
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

        sigma_back = 1e-1  # wholespace

        modelParameters = casingSimulations.model.Wholespace(
            src_a=np.r_[0., 0., -9.],
            src_b=np.r_[0., 0., -1.],
            freqs=np.r_[0.1, 1., 2.],
            sigma_back=sigma_back,  # wholespace
        )

        # Set up the meshes
        npadx, npadz = 11, 26

        mesh2D = casingSimulations.CylMeshGenerator(
            modelParameters=modelParameters, npadx=npadx, npadz=npadz, csz=2.
        )
        mesh3D = casingSimulations.CylMeshGenerator(
            modelParameters=modelParameters, hy=np.ones(4)*2*np.pi/4., csz=2.,
            npadx=npadx, npadz=npadz
        )

        # get wirepath on mesh
        wire2D = getSrcWire(mesh2D.mesh, modelParameters)
        wire3D = getSrcWire(mesh3D.mesh, modelParameters)

        # create sources
        srcList2D = [
            fdem.sources.RawVec_e(s_e=wire2D, freq=freq, rxList=[]) for freq in
            modelParameters.freqs
        ]
        srcList3D = [
            fdem.sources.RawVec_e(s_e=wire3D, freq=freq, rxList=[]) for freq in
            modelParameters.freqs
        ]

        # get phys prop models
        physprops2D = casingSimulations.model.PhysicalProperties(
            mesh2D, modelParameters
        )
        physprops3D = casingSimulations.model.PhysicalProperties(
            mesh3D, modelParameters
        )

        # create the problems and surveys
        prb2D = fdem.Problem3D_h(
            mesh2D.mesh,
            sigmaMap=physprops2D.wires.sigma,
            muMap=physprops2D.wires.mu,
            Solver=Pardiso
        )
        prb3D = fdem.Problem3D_h(
            mesh3D.mesh,
            sigmaMap=physprops3D.wires.sigma,
            muMap=physprops3D.wires.mu,
            Solver=Pardiso
        )

        survey2D = fdem.Survey(srcList2D)
        survey3D = fdem.Survey(srcList3D)

        prb2D.pair(survey2D)
        prb3D.pair(survey3D)

        print('starting 2D solve ... ')
        fields2D = prb2D.fields(physprops2D.model)
        print('  ... done \n')
        print('starting 3D solve ...')
        fields3D = prb3D.fields(physprops3D.model)
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
        j3D_x = j3D[:self.mesh3D.mesh.nFx].reshape(
            self.mesh3D.mesh.vnFx, order='F'
        )
        j3D_z = j3D[self.mesh3D.mesh.vnF[:2].sum():].reshape(
            self.mesh3D.mesh.vnFz, order='F'
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
        h3D_y = h3D[
            self.mesh3D.mesh.vnE[0]:self.mesh3D.mesh.vnE[:2].sum()
        ].reshape(
            self.mesh3D.mesh.vnEy, order='F'
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

            for ind in np.arange(1, self.mesh3D.mesh.vnC[1]):
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
            jtheta = j3D[self.mesh3D.mesh.nFx:self.mesh3D.mesh.nFz]

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
            hr = h3D[:self.mesh3D.mesh.nEx]
            hz = h3D[self.mesh3D.mesh.vnE[:2].sum():]

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

            for ind in np.arange(1, self.mesh3D.mesh.vnC[1]):
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
