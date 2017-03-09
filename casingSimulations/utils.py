import json
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
            targetClass = jsondict['__class__']
        data = getattr(
            casingSimulations, targetClass
        ).deserialize(jsondict)
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

    return mkvc(h3D_y[:, theta_ind, :])
