import time
import numpy as np
import scipy.sparse as sp
import os
import json
from scipy.constants import mu_0

import discretize
import properties
from discretize import utils
from pymatsolver import Pardiso
from SimPEG.EM import FDEM
from SimPEG import Utils, Maps

from .Model import PhysicalProperties, CasingParameters
from .Mesh import MeshGenerator
from . import Sources


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

    fields_filename = properties.String(
        "filename for the fields",
        default='fields.npy'
    )

    def __init__(self, cp, mesh, src, **kwargs):
        # if cp is a string, it is a filename, load in the json and create the
        # CasingParameters object
        if isinstance(cp, str):
            with open(cp, 'r') as outfile:
                cp = CasingParameters.deserialize(
                    json.load(outfile)
                )
        self.cp = cp

        # if cp is a string, it is a filename, load in the json and create the
        # CasingParameters object
        if isinstance(mesh, str):
            with open(mesh, 'r') as outfile:
                mesh = MeshGenerator(self.cp)
                mesh.deserialize(json.load(outfile))
        self.mesh = mesh

        # if src is a string, create a source of that type
        if isinstance(src, str):
            src = getattr(Sources, src)(
                self.mesh.mesh, self.cp
            )
        self.src = src

        # set keyword arguments
        Utils.setKwargs(self, **kwargs)

        # if the working directory does not exsist, create it
        if not os.path.isdir(self.directory):
            os.mkdir(self.directory)


class SimulationFDEM(BaseSimulation):
    """
    A wrapper to run an FDEM Forward Simulation
    :param CasingSimulations.CasingParameters cp: casing parameters object
    :param CasingSimulations.MeshGenerator mesh: a CasingSimulation mesh generator object
    """

    def __init__(self, cp, mesh, src, **kwargs):
        super(SimulationFDEM, self).__init__(cp, mesh, src, **kwargs)

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
            self.cp.skin_depth(sigma=self.cp.sigma_casing, mu=self.cp.mur_casing*mu_0)
        ))
        print('    casing thickness: {}'.format(
            self.cp.casing_t
        ))
        print('    skin depths in background: {}'.format(self.cp.skin_depth()))

        # Mesh Parameters
        self.mesh.validate()
        self.mesh.save(directory=self.directory, filename=self.cp_filename)
        print('   Saved Mesh Parameters')
        sim_mesh = self.mesh.mesh # grab the discretize mesh off of the mesh object
        print('      max x: {}, min z: {}, max z: {}'.format(
            sim_mesh.vectorNx.max(),
            sim_mesh.vectorNz.min(),
            sim_mesh.vectorNz.max()
        ))

        # Source (only validation, no saving, can be re-created from cp)
        self.src.validate()
        print('    Using {} sources'.format(len(self.src.srcList)))
        print('... parameters valid\n')

        # ----------------- Set up the simulation ----------------- #
        physprops = PhysicalProperties(sim_mesh, self.cp)
        prb = getattr(FDEM, 'Problem3D_{}'.format(self.formulation))(
            sim_mesh,
            sigmaMap=physprops.wires.sigma,
            muMap=physprops.wires.mu,
            Solver=Pardiso
        )

        survey = FDEM.Survey(self.src.srcList)
        prb.pair(survey)

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

