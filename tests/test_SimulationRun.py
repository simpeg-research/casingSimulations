import unittest
import numpy as np
import os
import shutil

import casingSimulations


plotIt = False
TOL = 1e-4
ZERO = 1e-7


class ForwardSimulationTestCyl2D(unittest.TestCase):

    dir2D = './sim2D'

    def setUp(self):
        sigma_back = 1e-1 # wholespace

        modelParameters = casingSimulations.model.CasingInWholespace(
            src_a = np.r_[0., np.pi, 0.], # the source fcts will take care of coupling it to the casing
            src_b = np.r_[1e3, np.pi, 0.], # return electrode
            freqs = np.r_[0.5],
            sigma_back = sigma_back, # wholespace
        )

        npadx, npadz = 8, 19
        dx2 = 200.
        csz = 0.25

        meshGenerator = casingSimulations.CasingMeshGenerator(
            modelParameters=modelParameters, npadx=npadx, npadz=npadz, csz=csz
        )

        self.modelParameters = modelParameters
        self.meshGenerator = meshGenerator

    def runSimulation(self, src):
        simulation = casingSimulations.run.SimulationFDEM(
            modelParameters=self.modelParameters,
            meshGenerator=self.meshGenerator,
            src=src,
            directory=self.dir2D
        )

        fields2D = simulation.run()

        loadedFields = np.load('/'.join([self.dir2D, 'fields.npy']))

        self.assertTrue(np.all(fields2D[:, 'h'] == loadedFields))

    def test_simulation2DTopCasing(self):

        src = casingSimulations.sources.TopCasingSrc(
            modelParameters=self.modelParameters,
            meshGenerator=self.meshGenerator,
        )
        src.validate()

        self.runSimulation(src)

    def test_simulation2DDownHoleCasingSrc(self):

        src = casingSimulations.sources.DownHoleCasingSrc(
            modelParameters=self.modelParameters,
            meshGenerator=self.meshGenerator,
        )
        src.validate()

        self.runSimulation(src)

    def test_simulation2DDownHoleTerminatingSrc(self):

        src = casingSimulations.sources.DownHoleTerminatingSrc(
            modelParameters=self.modelParameters,
            meshGenerator=self.meshGenerator,
        )
        src.validate()

        self.runSimulation(src)

    def tearDown(self):
        for d in [self.dir2D]:
            if os.path.isdir(d):
                shutil.rmtree(d)


if __name__ == '__main__':
    unittest.main()
