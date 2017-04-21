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
from SimPEG.EM.Static import DC

from .model import PhysicalProperties, CasingParameters
from .mesh import BaseMeshGenerator, CylMeshGenerator, TensorMeshGenerator
from .utils import load_properties
from . import sources
from .info import __version__


class LoadableInstance(properties.Instance):

    class_info = "an instance of a class or the name of a file from which the "
    "instance can be created"

    def validate(self, instance, value):
        if isinstance(value, str):
            return value
        return super(LoadableInstance, self).validate(instance, value)


class BaseSimulation(properties.HasProperties):
    """
    Base class wrapper to run an EM Forward Simulation
    :param CasingSimulations.CasingParameters cp: casing parameters object
    :param CasingSimulations.MeshGenerator mesh: a CasingSimulation mesh generator object
    """

    version = properties.String(
        "version of casingSimulations",
        default=__version__
    )

    directory = properties.String(
        "working directory",
        default="."
    )

    # cp_filename = properties.String(
    #     "filename for the casing properties",
    #     default="casingParameters.json"
    # )

    # mesh_filename = properties.String(
    #     "filename for the mesh",
    #     default="meshParameters.json"
    # )

    fields_filename = properties.String(
        "filename for the fields",
        default="fields.npy"
    )

    simulation_filename = properties.String(
        "filename for the simulation parameters",
        default="simulationParameters.json"
    )

    num_threads = properties.Integer(
        "number of threads",
        default=1
    )

    cp = LoadableInstance(
        "Casing Parameters instance",
        CasingParameters,
        required=True
    )

    meshGenerator = LoadableInstance(
        "mesh generator instance",
        BaseMeshGenerator,
        required=True
    )

    def __init__(self, **kwargs):
        # set keyword arguments
        Utils.setKwargs(self, **kwargs)

        # if the working directory does not exsist, create it
        if not os.path.isdir(self.directory):
            os.mkdir(self.directory)

    @properties.validator('cp')
    def _cp_load(self, change):
        # if cp is a string, it is a filename, load in the json and create the
        # CasingParameters object
        cp = change['value']
        if isinstance(cp, str):
            change['value'] = load_properties(cp)

    @properties.validator('meshGenerator')
    def _meshGenerator_load(self, change):
        # if cp is a string, it is a filename, load in the json and create the
        # CasingParameters object
        meshGenerator = change['value']
        if isinstance(meshGenerator, str):
            change['value'] = load_properties(meshGenerator)

    @property
    def src(self):
        if getattr(self, '_src', None) is None:
            self._src = getattr(sources, self.srcType)(
                self.cp, self.meshGenerator.mesh
            )
        return self._src

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

    def save(self, filename=None, directory=None):
        """
        Save the simulation parameters to json
        :param str file: filename for saving the simulation parameters
        """
        if directory is None:
            directory = self.directory
        if filename is None:
            filename = self.simulation_filename

        if not os.path.isdir(directory):  # check if the directory exists
            os.mkdir(directory)  # if not, create it
        f = '/'.join([directory, filename])
        with open(f, 'w') as outfile:
            cp = json.dump(self.serialize(), outfile)

    def fields(self):
        """
        fields from the forward simulation
        """
        if getattr(self, '_fields', None) is None:
            self._fields = self.run()
        return self._fields


class SimulationFDEM(BaseSimulation):
    """
    A wrapper to run an FDEM Forward Simulation
    :param CasingSimulations.CasingParameters cp: casing parameters object
    :param CasingSimulations.MeshGenerator mesh: a CasingSimulation mesh generator object
    """

    formulation = properties.StringChoice(
        "Formulation of the problem to solve [e, b, h, j]",
        default="h",
        choices=["e", "b", "h", "j"]
    )

    srcType = properties.String(
        "source class",
        required=True
    )

    def __init__(self, **kwargs):
        super(SimulationFDEM, self).__init__(**kwargs)

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

    def run(self):
        """
        Run the forward simulation
        """

        # ----------------- Validate Parameters ----------------- #

        print('Validating parameters...')
        self.validate()

        sim_mesh = self.meshGenerator.mesh # grab the discretize mesh off of the mesh object
        print('      max x: {}, min z: {}, max z: {}'.format(
            sim_mesh.vectorNx.max(),
            sim_mesh.vectorNz.min(),
            sim_mesh.vectorNz.max()
        ))

        # save simulation parameters
        self.save()

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


class SimulationDC(BaseSimulation):

    src_a = properties.Vector3(
        "a electrode location", required=True
    )

    src_b = properties.Vector3(
        "return electrode location", required=True
    )

    fields_filename = properties.String(
        "filename for the fields",
        default="fieldsDC.npy"
    )

    def __init__(self, **kwargs):
        super(SimulationDC, self).__init__(**kwargs)

        self._prob = DC.Problem3D_CC(
            self.meshGenerator.mesh,
            sigmaMap=self.physprops.wires.sigma,
            bc_type='Dirichlet',
            Solver=Pardiso
        )
        self._src = DC.Src.Dipole([], self.src_a, self.src_b)
        self._survey = DC.Survey([self._src])

        self._prob.pair(self._survey)

    def run(self):
        """
        Run the forward simulation
        """

        # ----------------- Validate Parameters ----------------- #

        print('Validating parameters...')
        self.validate()

        sim_mesh = self.meshGenerator.mesh # grab the discretize mesh off of the mesh object
        print('      max x: {}, min z: {}, max z: {}'.format(
            sim_mesh.vectorNx.max(),
            sim_mesh.vectorNz.min(),
            sim_mesh.vectorNz.max()
        ))

        # save simulation parameters
        self.save()

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
            fields[:, 'phiSolution']
        )
        print('Elapsed time : {}'.format(time.time()-t))

        self._fields = fields
        return fields
