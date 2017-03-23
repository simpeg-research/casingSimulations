import time
import numpy as np
import scipy.sparse as sp
import os
import json
from scipy.constants import mu_0

import mkl
import discretize
import properties
from discretize import utils
from pymatsolver import Pardiso
from SimPEG.EM import FDEM
from SimPEG import Utils, Maps

from .model import PhysicalProperties, CasingParameters
from .mesh import CylMeshGenerator, TensorMeshGenerator
from .utils import load_properties
from . import sources


class BaseSimulation(properties.HasProperties):
    """
    Base class wrapper to run an EM Forward Simulation
    :param CasingSimulations.CasingParameters cp: casing parameters object
    :param CasingSimulations.MeshGenerator mesh: a CasingSimulation mesh generator object
    """

    formulation = properties.StringChoice(
        "Formulation of the problem to solve [e, b, h, j]",
        default='h',
        choices=['e', 'b', 'h', 'j']
    )

    directory = properties.String(
        "working directory",
        default='.'
    )

    cp_filename = properties.String(
        "filename for the casing properties",
        default='casingParameters.json'
    )

    mesh_filename = properties.String(
        "filename for the mesh",
        default='meshParameters.json'
    )

    # mesh_type = properties.StringChoice(
    #     "type of mesh cyl or tensor",
    #     default="cyl",
    #     choices=["cyl", "tensor", "Cyl", "Tensor"]
    # )

    fields_filename = properties.String(
        "filename for the fields",
        default='fields.npy'
    )

    num_threads = properties.Integer(
        "number of threads",
        default=1
    )

    def __init__(self, cp, meshGenerator, src, **kwargs):
        # set keyword arguments
        Utils.setKwargs(self, **kwargs)

        # if cp is a string, it is a filename, load in the json and create the
        # CasingParameters object
        if isinstance(cp, str):
            cp = load_properties(cp)
        self.cp = cp

        # if cp is a string, it is a filename, load in the json and create the
        # CasingParameters object
        if isinstance(meshGenerator, str):
            meshGenerator = load_properties(meshGenerator)
        self.meshGenerator = meshGenerator

        # if src is a string, create a source of that type
        if isinstance(src, str):
            src = getattr(sources, src)(
                self.cp, self.meshGenerator.mesh
            )
        self.src = src

        # if the working directory does not exsist, create it
        if not os.path.isdir(self.directory):
            os.mkdir(self.directory)


class SimulationFDEM(BaseSimulation):
    """
    A wrapper to run an FDEM Forward Simulation
    :param CasingSimulations.CasingParameters cp: casing parameters object
    :param CasingSimulations.MeshGenerator mesh: a CasingSimulation mesh generator object
    """

    def __init__(self, cp, meshGenerator, src, **kwargs):
        super(SimulationFDEM, self).__init__(cp, meshGenerator, src, **kwargs)

        self._prob = getattr(
                FDEM, 'Problem3D_{}'.format(self.formulation)
                )(
                self.meshGenerator.mesh,
                sigmaMap=self.physprops.wires.sigma,
                muMap=self.physprops.wires.mu,
                Solver=Pardiso
            )
        self._survey = FDEM.Survey(self.src.srcList)

        self._prob.pair(self._survey)

    @property
    def physprops(self):
        if getattr(self, '_physprops', None) is None:
            self._physprops = PhysicalProperties(
                self.meshGenerator.mesh, self.cp
            )
        return self._physprops

    @property
    def prob(self):
        return self._prob

    @property
    def survey(self):
        return self._survey

    def run(self):
        """
        Run the forward simulation
        """

        # ----------------- Validate Parameters ----------------- #

        print('Validating parameters...')

        # Casing Parameters
        self.cp.validate()
        self.cp.save(directory=self.directory, filename=self.cp_filename)
        print('  Saved casing properties: {}')
        print('    skin depths in casing: {}'.format(
            self.cp.skin_depth(
                sigma=self.cp.sigma_casing, mu=self.cp.mur_casing*mu_0
            )
        ))
        print('    casing thickness: {}'.format(
            self.cp.casing_t
        ))
        print('    skin depths in background: {}'.format(self.cp.skin_depth()))

        # Mesh Parameters
        self.meshGenerator.validate()
        self.meshGenerator.save(
            directory=self.directory, filename=self.cp_filename
        )
        print('   Saved Mesh Parameters')
        sim_mesh = self.meshGenerator.mesh # grab the discretize mesh off of the mesh object
        print('      max x: {}, min z: {}, max z: {}'.format(
            sim_mesh.vectorNx.max(),
            sim_mesh.vectorNz.min(),
            sim_mesh.vectorNz.max()
        ))

        # Source (only validation, no saving, can be re-created from cp)
        self.src.validate()
        print('    Using {} sources'.format(len(self.src.srcList)))
        print('... parameters valid\n')

        # --------------- Set the number of threads --------------- #
        mkl.set_num_threads(self.num_threads)

        # ----------------- Set up the simulation ----------------- #
        physprops = self.physprops
        prb = self.prob
        # survey = self.survey
        # prb.pair(survey)

        # ----------------- Run the the simulation ----------------- #
        print('Starting Simulation')
        t = time.time()
        fields = prb.fields(physprops.model)
        np.save(
            '/'.join([self.directory, self.fields_filename]),
            fields[:, '{}Solution'.format(self.formulation)]
        )
        print('Elapsed time : {}'.format(time.time()-t))

        self._fields = fields
        return fields

    def fields(self):
        """
        fields from the forward simulation
        """
        if getattr(self, '_fields', None) is None:
            self._fields = self.run()
        return self._fields

