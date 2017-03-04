import unittest
import numpy as np
import shutil

import CasingSimulations


plotIt = False
TOL = 1e-4
ZERO = 1e-7


class ForwardSimulationTest(unittest.TestCase):

    dir2D = './sim2D'

    def test_simulation2D(self):
        sigma_back = 1e-1 # wholespace

        cp = CasingSimulations.CasingParameters(
            casing_l = 1000.,
            src_a = np.r_[0., np.pi, 0.], # the source fcts will take care of coupling it to the casing
            src_b = np.r_[1e3, np.pi, 0.], # return electrode
            freqs = np.r_[0.5],
            sigma_back = sigma_back, # wholespace
            sigma_layer = sigma_back,
            sigma_air = sigma_back,

        )

        npadx, npadz = 8, 19
        dx2 = 200.
        csz = 0.25

        mesh2D = CasingSimulations.CasingMesh(
            cp=cp, npadx=npadx, npadz=npadz, dx2=dx2, csz=csz
        )

        src = CasingSimulations.Sources.TopCasingSrc(
            cp, mesh2D.mesh
        )
        src.validate()

        simulation = CasingSimulations.Run.SimulationFDEM(
            cp, mesh2D, src, directory=self.dir2D
        )

        fields2D = simulation.run()

        loadedFields = np.load('/'.join([self.dir2D, 'fields.npy']))

        self.assertTrue(np.all(fields2D[:, 'h'] == loadedFields))

    def tearDown(self):
        for d in [self.dir2D]:
            shutil.rmtree(d)


if __name__ == '__main__':
    unittest.main()
