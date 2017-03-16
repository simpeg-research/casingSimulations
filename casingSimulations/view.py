import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plotFace2D(
    mesh2D,
    j, real_or_imag='real', ax=None, range_x=np.r_[0., 1000.],
    range_y=np.r_[-2000., 0.], sample_grid=np.r_[5., 5.],
    logScale=True
):
    """
    Create a streamplot (a slice in the theta direction) of a face vector

    :param discretize.CylMesh mesh2D: cylindrically symmetric mesh
    :param np.ndarray j: face vector (x, z components)
    :param str real_or_imag: real or imaginary component
    :param matplotlib.axes ax: axes
    :param numpy.ndarray range_x: x-extent over which we want to plot
    :param numpy.ndarray range_y: y-extent over which we want to plot
    :param numpy.ndarray sample_grid: x, y spacings at which to re-sample the plotting grid
    :param bool logScale: use a log scale for the colorbar?
    """
    if ax is None:
        fig, ax = plt.subplots(1, 3, figsize=(10, 4))

    if len(j) == mesh2D.nF:
        vType = 'F'
    elif len(j) == mesh2D.nC*2:
        vType = 'CCv'

    if logScale is True:
        pcolorOpts = {
                'norm':LogNorm()
        }
    else:
        pcolorOpts = {}

    plt.colorbar(
        mesh2D.plotImage(
                getattr(j, real_or_imag),
                view='vec', vType=vType, ax=ax,
                range_x=range_x, range_y=range_y, sample_grid=sample_grid,
                mirror=False,
                pcolorOpts=pcolorOpts,
            )[0], ax=ax
    )

    return ax


def plotEdge2D(
    mesh2D,
    h, real_or_imag='real', ax=None, range_x=np.r_[0., 1000.],
    range_y=np.r_[-2000., 0.], sample_grid=np.r_[5., 5.],
    logScale=True
):
    """
    Create a pcolor plot (a slice in the theta direction) of an edge vector

    :param discretize.CylMesh mesh2D: cylindrically symmetric mesh
    :param np.ndarray h: edge vector (y components)
    :param str real_or_imag: real or imaginary component
    :param matplotlib.axes ax: axes
    :param numpy.ndarray range_x: x-extent over which we want to plot
    :param numpy.ndarray range_y: y-extent over which we want to plot
    :param numpy.ndarray sample_grid: x, y spacings at which to re-sample the plotting grid
    :param bool logScale: use a log scale for the colorbar?
    """

    if ax is None:
        fig, ax = plt.subplots(1, 3, figsize=(10, 4))

    if len(j) == mesh2D.nE:
        vType = 'E'
    elif len(j) == mesh2D.nC:
        vType = 'CC'

    if logScale is True:
        pcolorOpts = {
            'norm':LogNorm()
        }
    else:
        pcolorOpts = {}

    plt.colorbar(
        mesh2D.plotImage(
                getattr(h, real_or_imag),
                view='real', vType=vType, ax=ax,
                range_x=range_x, range_y=range_y, sample_grid=sample_grid,
                mirror=False,
                pcolorOpts=pcolorOpts,
            )[0], ax=ax
    )
