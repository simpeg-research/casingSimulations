import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import discretize

from . import utils


def plot_slice(
    mesh, v, ax=None, clim=None, pcolorOpts=None, theta_ind=0
):
    """
    Plot a cell centered property

    :param numpy.array prop: cell centered property to plot
    :param matplotlib.axes ax: axis
    :param numpy.array clim: colorbar limits
    :param dict pcolorOpts: dictionary of pcolor options
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    if pcolorOpts is None:
        pcolorOpts = {}

    # generate a 2D mesh for plotting slices
    mesh2D = discretize.CylMesh(
        [mesh.hx, 1., mesh.hz], x0=mesh.x0
    )

    vplt = v.reshape(mesh.vnC, order='F')

    cb = plt.colorbar(
        mesh2D.plotImage(
            discretize.utils.mkvc(vplt[:, theta_ind, :]), ax=ax,
            mirror=True, pcolorOpts=pcolorOpts
        )[0], ax=ax,
    )

    if clim is not None:
        cb.set_clim(clim)
        cb.update_ticks()

    return ax, cb


def plotFace2D(
    mesh2D,
    j, real_or_imag='real', ax=None, range_x=None,
    range_y=None, sample_grid=None,
    logScale=True, clim=None, mirror=False, pcolorOpts=None,
    cbar=True
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
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    if len(j) == mesh2D.nF:
        vType = 'F'
    elif len(j) == mesh2D.nC*2:
        vType = 'CCv'

    if pcolorOpts is None:
        pcolorOpts = {}

    if logScale is True:
        pcolorOpts['norm'] = LogNorm()
    else:
        pcolorOpts = {}

    f = mesh2D.plotImage(
        getattr(j, real_or_imag),
        view='vec', vType=vType, ax=ax,
        range_x=range_x, range_y=range_y, sample_grid=sample_grid,
        mirror=mirror,
        pcolorOpts=pcolorOpts,
    )

    out = (ax,)

    if cbar is True:
        cb = plt.colorbar(f[0], ax=ax)
        out += (cbar,)

        if clim is not None:
            cb.set_clim(clim)
            cb.update_ticks()

    return out


def plotEdge2D(
    mesh2D,
    h, real_or_imag='real', ax=None, range_x=None,
    range_y=None, sample_grid=None,
    logScale=True, clim=None, mirror=False, pcolorOpts=None
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
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    if len(h) == mesh2D.nE:
        vType = 'E'
    elif len(h) == mesh2D.nC:
        vType = 'CC'
    elif len(h) == 2*mesh2D.nC:
        vType = 'CCv'

    if logScale is True:
        pcolorOpts['norm'] = LogNorm()
    else:
        pcolorOpts = {}

    cb = plt.colorbar(
        mesh2D.plotImage(
            getattr(h, real_or_imag),
            view='real', vType=vType, ax=ax,
            range_x=range_x, range_y=range_y, sample_grid=sample_grid,
            mirror=mirror,
            pcolorOpts=pcolorOpts,
        )[0], ax=ax
    )

    if clim is not None:
        cb.set_clim(clim)

    return ax, cb


def plotLinesFx(
    mesh,
    field,
    pltType='semilogy',
    ax=None,
    theta_ind=0,
    xlim=[0., 2500.],
    zloc=0.,
    real_or_imag='real',
    color_ind=0,
    label=None,
    linestyle='-'
):

    mesh2D = discretize.CylMesh([mesh.hx, 1., mesh.hz], x0=mesh.x0)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    fplt = utils.face3DthetaSlice(
        mesh, field, theta_ind=theta_ind
    )

    fx = discretize.utils.mkvc(fplt[:mesh2D.vnF[0]].reshape(
        [mesh2D.vnFx[0], mesh2D.vnFx[2]], order='F')
    )

    xind = ((mesh2D.gridFx[:, 0] > xlim[0]) & (mesh2D.gridFx[:, 0] < xlim[1]))
    zind = (
        (mesh2D.gridFx[:, 2] > -mesh2D.hz.min()+zloc) & (mesh2D.gridFx[:, 2] < zloc)
    )
    pltind = xind & zind

    fx = getattr(fx[pltind], real_or_imag)
    x = mesh2D.gridFx[pltind, 0]

    if pltType in ['semilogy', 'loglog']:
        getattr(ax, pltType)(x, -fx, '--', color='C{}'.format(color_ind))

    getattr(ax, pltType)(
        x, fx.real, linestyle, color='C{}'.format(color_ind),
        label=label
    )

    # # plot the DC solution
    # fxDC = utils.face3DthetaSlice(
    #     mesh, fieldsDC[:, fieldType], theta_ind=theta_ind
    # )
    # fxDC = discretize.utils.mkvc(
    #     fxDC[:mesh2D.vnF[0]].reshape(
    #         [mesh2D.vnFx[0], mesh2D.vnFx[2]], order='F'
    #     )
    # )
    # fxDC = fxDC[pltind]

    # getattr(ax[0], pltType)(x, -fxDC, '--', color='k')
    # getattr(ax[0], pltType)(x, fxDC, '-', color='k', label='DC')


    # [a.set_xlim([2., 1000.]) for a in ax]
    ax.grid('both', linestyle=linestyle, linewidth=0.4, color=[0.8, 0.8, 0.8])
    ax.set_xlabel('distance from well (m)')

    plt.tight_layout()

    return ax

