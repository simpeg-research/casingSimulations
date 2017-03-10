import unittest
import numpy as np
import discretize
from discretize import utils

import casingSimulations


class TestMeshConstruction(unittest.TestCase):

    def setUp(self):
        sigma_back = 0.1
        cart_cp = casingSimulations.CasingParameters(
            sigma_casing = sigma_back,
            sigma_inside = sigma_back,
            sigma_layer = sigma_back,
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
            np.ceil((cart_cp.src_a[0] - cart_cp.src_b[0]) / csx) + 2*nch
        )
        ncy = int(
            np.ceil(domain_y / csy) + 2*nch
        )
        ncz = int(
            np.ceil((cart_cp.casing_z[1] - cart_cp.casing_z[0]) / csz) +
            nca + ncb
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

        x0x = -hx.sum()/2. + (cart_cp.src_b[0] + cart_cp.src_a[0])/2.
        x0y = -hy.sum()/2.
        x0z = -hz[:npadz+ncz-nca].sum()

        x0 = np.r_[x0x, x0y, x0z]

        self.mesh_d = discretize.TensorMesh([hx, hy, hz], x0=x0)
        self.meshGen = casingSimulations.TensorMeshGenerator(
            cp = cart_cp,
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
        print(
            'x0 discretize: {}, x0 meshGen: {}'.format(
                self.mesh_d.x0, self.mesh_c.x0
            )
        )
        self.assertTrue(np.all(self.mesh_d.x0 == self.mesh_c.x0))
        self.assertTrue(np.all(self.mesh_d.hx == self.mesh_c.hx))
        self.assertTrue(np.all(self.mesh_d.hy == self.mesh_c.hy))
        self.assertTrue(np.all(self.mesh_d.hz == self.mesh_c.hz))

if __name__ == '__main__':
    unittest.main()
