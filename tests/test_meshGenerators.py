import unittest
import numpy as np
import discretize
import os
from discretize import utils

import casingSimulations


def compareTensorMeshes(mesh_a, mesh_b, testname):
    print("\nTesting {} ... ".format(testname))

    # check the x0's
    passed = np.all(mesh_a.x0 == mesh_b.x0)
    print("   x0: {} == {} ? {}".format(
        mesh_a.x0, mesh_b.x0, 'ok' if passed else 'FAIL'
    ))

    assert passed, ("{}: x0 are different {}, {}".format(
        testname, mesh_a.x0, mesh_b.x0
    ))

    # check the h's
    for orientation in ['x', 'y', 'z']:
        ha = getattr(mesh_a, 'h{}'.format(orientation))
        hb = getattr(mesh_b, 'h{}'.format(orientation))
        passed = np.all(ha == hb)

        print("   h{}: |a|={}, |b|={} ? {}".format(
            orientation, np.linalg.norm(ha), np.linalg.norm(hb), passed
        ))

        assert passed, ("{}: h{} are different {}, {}".format(
            testname, np.linalg.norm(ha), np.linalg.norm(hb), orientation,
            'ok' if passed else 'FAIL'
        ))

    return True


class TestTensorMeshConstruction(unittest.TestCase):

    def setUp(self):
        sigma_back = 0.1
        cart_modelParameters = casingSimulations.model.CasingInWholespace(
            sigma_casing = sigma_back,
            sigma_inside = sigma_back,
            sigma_back = sigma_back,
            mur_casing = 1.,
            src_a = np.r_[0., 0., -950.],
            src_b = np.r_[-1e3, 0., 0.]
        )

        csx = 25.
        csy = csx
        csz = csx

        pfx = 1.5
        pfy = 1.5
        pfz = 1.5

        npadx = 8
        npady = npadx
        npadz = npadx

        nca = 5  # number of core cells above the air-earth interface
        ncb = 5  # number of core cells below the borehole
        nch = 5  # number of extra cells in the horizontal direction

        domain_y = 300.

        ncx = int(
            np.ceil((cart_modelParameters.src_a[0] - cart_modelParameters.src_b[0]) / csx) + 2*nch
        )
        ncy = int(
            np.ceil(domain_y / csy) + 2*nch
        )
        ncz = int(
            np.ceil((cart_modelParameters.casing_z[1] - cart_modelParameters.casing_z[0]) / csz) + nca + ncb
        )

        hx = utils.meshTensor(
            [(csx, npadx, -pfx), (csx, ncx), (csx, npadx, pfx)]
        )
        hy = utils.meshTensor(
            [(csy, npady, -pfy), (csy, ncy), (csy, npady, pfy)]
        )
        hz = utils.meshTensor(
            [(csz, npadz, -pfz), (csz, ncz), (csz, npadz, pfz)]
        )

        x0x = -hx.sum()/2. + (cart_modelParameters.src_b[0] + cart_modelParameters.src_a[0])/2.
        x0y = -hy.sum()/2.
        x0z = -hz[:npadz+ncz-nca].sum()

        x0 = np.r_[x0x, x0y, x0z]

        self.mesh_d = discretize.TensorMesh([hx, hy, hz], x0=x0)
        self.meshGen = casingSimulations.TensorMeshGenerator(
            modelParameters = cart_modelParameters,
            csx = csx,
            csy = csy,
            csz = csz,
            domain_y = domain_y,
            pfx = pfx,
            pfy = pfy,
            pfz = pfz,
            npadx = npadx,
            npady = npadx,
            npadz = npadx,
            nca = 5,  # number of core cells above the air-earth interface
            ncb = 5,  # number of core cells below the borehole
            nch = 5  # number of extra cells in the horizontal direction
        )
        self.mesh_c = self.meshGen.mesh

    def test_TensorCreation(self):
        compareTensorMeshes(self.mesh_c, self.mesh_d, 'TensorCreation')

    def test_TensorCopy(self):
        mesh_a = self.meshGen.mesh
        mesh_b = self.meshGen.copy().mesh
        compareTensorMeshes(mesh_a, mesh_b, 'TensorCopy')

    def test_TensorSaveLoad(self):
        self.meshGen.save()
        meshGen2 = casingSimulations.load_properties(
            os.path.sep.join([self.meshGen.directory, self.meshGen.filename])
        )
        compareTensorMeshes(self.meshGen.mesh, meshGen2.mesh, 'TensorSaveLoad')


if __name__ == '__main__':
    unittest.main()
