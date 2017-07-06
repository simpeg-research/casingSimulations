import unittest
import numpy as np
import discretize
import os
import pytest
from discretize import utils
from scipy.constants import mu_0

import casingSimulations as casingSim


class ModelTests(unittest.TestCase):

    def setUp(self):
        self.mesh = discretize.TensorMesh([10, 4, 10], x0='CCC')

    def test_wholespace(self):
        wholespace = casingSim.model.Wholespace()

        # check the defaults
        self.assertTrue(wholespace.sigma_back == 1e-2)
        self.assertTrue(wholespace.mur_back == 1.)
        self.assertTrue(np.all(wholespace.mu(self.mesh) == mu_0))
        self.assertTrue(np.all(wholespace.sigma(self.mesh) == 1e-2))
        # update things
        wholespace.sigma_back = 10.
        self.assertTrue(wholespace.sigma_back == 10.)

        # check that conductivity can't be negative
        with self.assertRaises(Exception):
            wholespace.sigma_back = -10
            wholespace.mur_back = -9

    def test_halfspace(self):
        halfspace = casingSim.model.Halfspace(surface_z=0.1)
        sigma = halfspace.sigma(self.mesh)
        self.assertTrue(
            np.all(sigma[self.mesh.gridCC[:, 2] < halfspace.surface_z] == 1e-2)
        )
        self.assertTrue(
            np.all(sigma[self.mesh.gridCC[:, 2] > halfspace.surface_z] == 1e-6)
        )

    def test_layer(self):
        layer = casingSim.model.SingleLayer(
            surface_z=0.1, layer_z=[-np.inf, -0.2]
        )

    def test_different_version(self):
        with pytest.warns(UserWarning):
            wholespace = casingSim.model.Wholespace(version='0.0.1')

        #     wholespace.validate()


if __name__ == '__main__':
    unittest.main()
