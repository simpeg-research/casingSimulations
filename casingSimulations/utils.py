import json
import numpy as np
import datetime
from discretize import utils
import casingSimulations


def load_properties(filename, targetClass=None):
    """
    Open a json file and load the properties into the target class
    :param str filename: name of file to read in
    :param str targetClass: name of the target class to recreate
    """
    with open(filename, 'r') as outfile:
        jsondict = json.load(outfile)
        if targetClass is None:
            targetClass = getattr(
                casingSimulations, jsondict['__class__']
            )
        data = targetClass.deserialize(jsondict)
    return data


# grab 2D slices
def face3DthetaSlice(mesh3D, j3D, theta_ind=0):
    """
    Grab a theta slice through a 3D field defined on faces
    (x, z components), consistent with what would be found from a
    2D simulation

    :param discretize.CylMesh mesh3D: 3D cyl mesh
    :param numpy.ndarray j3D: vector of fluxes on mesh
    :param int theta_ind: index of the theta slice that you want
    """
    j3D_x = j3D[:mesh3D.nFx].reshape(mesh3D.vnFx, order='F')
    j3D_z = j3D[mesh3D.vnF[:2].sum():].reshape(mesh3D.vnFz, order='F')

    j3Dslice = np.vstack([
        utils.mkvc(j3D_x[:, theta_ind, :], 2),
        utils.mkvc(j3D_z[:, theta_ind, :], 2)
    ])

    return j3Dslice


def edge3DthetaSlice(mesh3D, h3D, theta_ind=0):
    """
    Grab a theta slice through a 3D field defined on edges
    (y component), consistent with what would be found from a
    2D simulation

    :param discretize.CylMesh mesh3D: 3D cyl mesh
    :param numpy.ndarray h3D: vector of fields on mesh
    :param int theta_ind: index of the theta slice that you want
    """

    h3D_y = h3D[mesh3D.nEx:mesh3D.vnE[:2].sum()].reshape(
        mesh3D.vnEy, order='F'
    )

    return utils.mkvc(h3D_y[:, theta_ind, :])


def ccv3DthetaSlice(mesh3D, v3D, theta_ind=0):
    """
    Grab a theta slice through a 3D field defined at cell centers

    :param discretize.CylMesh mesh3D: 3D cyl mesh
    :param numpy.ndarray v3D: vector of fields on mesh
    :param int theta_ind: index of the theta slice that you want
    """

    ccv_x = v3D[:mesh3D.nC].reshape(mesh3D.vnC, order='F')
    ccv_y = v3D[mesh3D.nC:2*mesh3D.nC].reshape(mesh3D.vnC, order='F')
    ccv_z = v3D[2*mesh3D.nC:].reshape(mesh3D.vnC, order='F')

    return np.vstack([
        utils.mkvc(ccv_x[:, theta_ind, :], 2),
        utils.mkvc(ccv_y[:, theta_ind, :], 2),
        utils.mkvc(ccv_z[:, theta_ind, :], 2)
    ])


def writeSimulationPy(
    cp='CasingParameters.json',
    meshGenerator='MeshParameters.json',
    srcType='VerticalElectricDipole',
    physics='FDEM',
    fields_filename='fields.npy',
    directory='.',
    simulation_filename='simulation.py'
):

    sim_file = '/'.join([directory, simulation_filename])
    with open(sim_file, 'w') as f:

        # add comment stating when the file was generated
        f.write(
            "# Autogenerated at {date} on version {version}\n".format(
                date=datetime.datetime.now().isoformat(),
                version=casingSimulations.__version__
            )
        )

        # write the imports
        f.write(
            "import casingSimulations\n"
        )

        # write the simulation
        f.write(
            """
# Set up the simulation
sim = casingSimulations.run.Simulation{physics}(
    cp='{cp}',
    meshGenerator='{meshGenerator}',
    srcType='{srcType}',
    fields_filename='{fields_filename}'
)
\n""".format(
                physics=physics,
                cp=cp,
                meshGenerator=meshGenerator,
                srcType=srcType,
                fields_filename=fields_filename
            )
        )

        # write the run
        f.write(
            "# run the simulation \nfields = sim.run()"
        )

    print('wrote {}'.format(sim_file))
