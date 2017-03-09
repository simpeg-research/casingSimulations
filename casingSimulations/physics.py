from SimPEG import Utils

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# Calculate Casing Currents from fields object
def CasingCurrents(cp, fields, mesh, survey):
    IxCasing = {}
    IzCasing = {}

    # casing_ind = sigma_m.copy()
    # casing_ind[[0, 1, 3]] = 0. # zero outside casing
    # casing_ind[2] = 1. # 1 inside casing

    # actMap_Zeros = Maps.InjectActiveCells(mesh, indActive, 0.)

    # indCasing = actMap_Zeros * casingMap * casing_ind

    # casing_faces = mesh.aveF2CC.T * indCasing
    # casing_faces[casing_faces < 0.25] = 0

    casing_faces_x = (
        (mesh.gridFx[:, 0] >= cp.casing_a) &
        (mesh.gridFx[:, 0] <= cp.casing_b) &
        (mesh.gridFx[:, 2] <= cp.casing_z[1]) &
        (mesh.gridFx[:, 2] >= cp.casing_z[0])
    )
    casing_faces_z = (
        (mesh.gridFz[:, 0] >= cp.casing_a) &
        (mesh.gridFz[:, 0] <= cp.casing_b) &
        (mesh.gridFz[:, 2] <= cp.casing_z[1]) &
        (mesh.gridFz[:, 2] >= cp.casing_z[0])
    )

    for mur in cp.muModels:
        j = fields[mur][:, 'j']
        jA = Utils.sdiag(mesh.area) * j

        jACasing = Utils.sdiag(
            np.hstack([casing_faces_x, casing_faces_z])
        ) * jA

        ixCasing = []
        izCasing = []

        for ind in range(len(survey.srcList)):
            jxCasing = jACasing[:mesh.nFx, ind].reshape(
                mesh.vnFx[0], mesh.vnFx[2], order='F'
            )
            jzCasing = jACasing[mesh.nFx:, ind].reshape(
                mesh.vnFz[0], mesh.vnFz[2], order='F'
            )

            ixCasing.append(jxCasing.sum(0))
            izCasing.append(jzCasing.sum(0))

        IxCasing[mur] = ixCasing
        IzCasing[mur] = izCasing
    return IxCasing, IzCasing



def plotCurrentDensity(
    mesh,
    fields_j, saveFig=False,
    figsize=(4, 5), fontsize=12, csx=5., csz=5.,
    xmax=1000., zmin=0., zmax=-1200., real_or_imag='real',
    mirror=False, ax=None, fig=None, clim=None
):
    csx, ncx = csx, np.ceil(xmax/csx)
    csz, ncz = csz, np.ceil((zmin-zmax)/csz)

    if mirror is True:
        xlim = [-xmax, xmax]
        x0 = [-xmax, -csx/2., zmax]
        ncx *= 2.
    else:
        xlim = [0., xmax]
        x0 = [0, -csx/2., zmax]

    ylim=[zmax, zmin]

    # define the tensor mesh
    meshcart = Mesh.TensorMesh(
        [[(csx, ncx)], [(csx, 1)], [(csz, ncz)]], x0
    )

    projF = mesh.getInterpolationMatCartMesh(meshcart, 'F')

    jcart = projF*fields_j
    jcart = getattr(jcart, real_or_imag)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    if saveFig is True:
        # this looks obnoxious inline, but nice in the saved png
        f = meshcart.plotSlice(
            jcart, normal='Y', vType='F', view='vec',
            pcolorOpts={
                'norm': LogNorm(), 'cmap': plt.get_cmap('viridis')
            },
            streamOpts={'arrowsize': 6, 'color': 'k'},
            ax=ax
        )
    else:
        f = meshcart.plotSlice(
            jcart, normal='Y', vType='F', view='vec',
            pcolorOpts={
                'norm': LogNorm(), 'cmap': plt.get_cmap('viridis')
            },
            ax=ax
        )
    plt.colorbar(
        f[0], label='{} current density (A/m$^2$)'.format(
            real_or_imag
        )
    )

    if clim is not None:
        f.set_clim(clim)

    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    # ax.set_title('Current Density')
    ax.set_xlabel('radius (m)', fontsize=fontsize)
    ax.set_ylabel('z (m)', fontsize=fontsize)

    if saveFig is True:
        fig.savefig('primaryCurrents', dpi=300, bbox_inches='tight')

    return ax


def plot_currents_over_freq(
    IxCasing, IzCasing, cp, mesh,
    mur=1, subtract=None, real_or_imag='real', ax=None, xlim=[-1100., 0.],
    logScale=True, srcinds=[0], ylim_0=None, ylim_1=None

):
    print("mu = {} mu_0".format(mur))

    ixCasing = IxCasing[mur]
    izCasing = IzCasing[mur]

    if ax is None:
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    for a in ax:
        a.grid(
            which='both', linestyle='-', linewidth=0.4, color=[0.8, 0.8, 0.8],
            alpha=0.5
        )
        # getattr(a, 'semilogy' if logScale is True else 'plot')(
        #     [cp.src_a[2], cp.src_a[2]], [1e-14, 1], color=[0.3, 0.3, 0.3]
        # )
        a.set_xlim(xlim)
        a.invert_xaxis()

    col = ['b', 'g', 'r', 'c', 'm', 'y']
    pos_linestyle = ['-', '-']
    neg_linestyle = ['--', '--']
    leg = []

    for i, f in enumerate(cp.freqs):
        for srcind in srcinds:
            # src = survey.getSrcByFreq(survey.freqs[freqind])[srcind]
            # j = Utils.mkvc(fields[mur][src, 'j'].copy())

            Iind = i + srcind*len(cp.freqs)

            Ix, Iz = ixCasing[Iind].copy(), izCasing[Iind].copy()

            if subtract is not None:
                Ix += -IxCasing[subtract][Iind].copy()
                Iz += -IzCasing[subtract][Iind].copy()

            Ix_plt = getattr(Ix, real_or_imag)
            Iz_plt = getattr(Iz, real_or_imag)

            if logScale is True:
                ax0 = ax[0].semilogy(
                    mesh.vectorNz, Iz_plt,
                    '{linestyle}{color}'.format(
                        linestyle=pos_linestyle[srcind],
                        color=col[i]
                    ),
                    label="{} Hz".format(f)
                )
                ax[0].semilogy(
                    mesh.vectorNz, -Iz_plt,
                    '{linestyle}{color}'.format(
                        linestyle=neg_linestyle[srcind],
                        color=col[i]
                    )
                )
                ax[1].semilogy(
                    mesh.vectorCCz, Ix_plt, '{linestyle}{color}'.format(
                        linestyle=pos_linestyle[srcind],
                        color=col[i]
                    )
                )
                ax[1].semilogy(
                    mesh.vectorCCz, -Ix_plt, '{linestyle}{color}'.format(
                        linestyle=neg_linestyle[srcind],
                        color=col[i]
                    )
                )
            else:
                ax0 = ax[0].plot(
                    mesh.vectorNz, Iz_plt, '{linestyle}{color}'.format(
                        linestyle=pos_linestyle[srcind],
                        color=col[i]
                    ), label="{} Hz".format(f)
                )
                ax[1].plot(
                    mesh.vectorCCz, Ix_plt, '{linestyle}{color}'.format(
                        linestyle=pos_linestyle[srcind],
                        color=col[i]
                    )
                )

            leg.append(ax0)

    if ylim_0 is not  None:
        ax[0].set_ylim(ylim_0)

    if ylim_1 is not None:
        ax[1].set_ylim(ylim_1)

    ax[0].legend(bbox_to_anchor=[1.25, 1])
    # plt.show()

    return ax


# plot current density over mu
def plot_currents_over_mu(
    IxCasing, IzCasing, cp, mesh,
    freqind=0, real_or_imag='real',
    subtract=None, ax=None, fig=None, logScale=True,
    srcinds=[0],
    ylim_0=None, ylim_1=None
):
    print("{} Hz".format(cp.freqs[freqind]))

    if ax is None:
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    for a in ax:
        a.grid(
            which='both', linestyle='-', linewidth=0.4, color=[0.8, 0.8, 0.8],
            alpha=0.5
        )
        # getattr(a, 'semilogy' if logScale is True else 'plot')(
        #     [cp.src_a[2], cp.src_a[2]], [1e-14, 1], color=[0.3, 0.3, 0.3]
        # )
        a.set_xlim([-1100., 0.])
    #     a.set_ylim([1e-3, 1.])
        a.invert_xaxis()

    col = ['b', 'g', 'r', 'c', 'm', 'y']
    pos_linestyle = ['-', '-']
    neg_linestyle = ['--', '--']
    leg = []

    for i, mur in enumerate(cp.muModels):
        for srcind in srcinds:

            Iind = i + srcind*len(cp.freqs)

            ixCasing = IxCasing[mur]
            izCasing = IzCasing[mur]

            Ix, Iz = ixCasing[Iind].copy(), izCasing[Iind].copy()

            if subtract is not None:
                Ix = Ix - IxCasing[subtract][Iind]
                Iz = Iz - IzCasing[subtract][Iind]

            Iz_plt = getattr(Iz, real_or_imag)
            Ix_plt = getattr(Ix, real_or_imag)

            if logScale is True:
                ax0 = ax[0].semilogy(
                    mesh.vectorNz, Iz_plt, '{linestyle}{color}'.format(
                        linestyle=pos_linestyle[srcind], color=col[i]
                    ), label="{} $\mu_0$".format(mur)
                )
                ax[0].semilogy(
                    mesh.vectorNz, -Iz_plt, '{linestyle}{color}'.format(
                        linestyle=neg_linestyle[srcind], color=col[i]
                    )
                )
                ax[1].semilogy(
                    mesh.vectorCCz, Ix_plt, '{linestyle}{color}'.format(
                        linestyle=pos_linestyle[srcind], color=col[i]
                    )
                )
                ax[1].semilogy(
                    mesh.vectorCCz, -Ix_plt, '{linestyle}{color}'.format(
                        linestyle=neg_linestyle[srcind], color=col[i]
                    )
                )
            else:
                ax0 = ax[0].plot(
                    mesh.vectorNz, Iz_plt, '{linestyle}{color}'.format(
                        linestyle=pos_linestyle[srcind], color=col[i]
                    ), label="{} $\mu_0$".format(mur)
                )
                ax[1].plot(
                    mesh.vectorCCz, Ix_plt, '{linestyle}{color}'.format(
                        linestyle=pos_linestyle[srcind], color=col[i]
                    )
                )

            leg.append(ax0)

    if ylim_0 is not  None:
        ax[0].set_ylim(ylim_0)

    if ylim_1 is not None:
        ax[1].set_ylim(ylim_1)

    ax[0].legend(bbox_to_anchor=[1.25, 1])
    # plt.show()
    return ax


# plot over mu
def plot_j_over_mu_z(
    cp, fields, mesh, survey, freqind=0, r=1., xlim=[-1100., 0.],
    real_or_imag='real', subtract=None, ax=None, logScale=True, srcinds=[0],
    ylim_0=None, ylim_1=None, fig=None
):
    print("{} Hz".format(cp.freqs[freqind]))

    x_plt = np.r_[r]
    z_plt = np.linspace(xlim[0], xlim[1], int(xlim[1]-xlim[0]))

    XYZ = Utils.ndgrid(x_plt, np.r_[0], z_plt)

    Pfx = mesh.getInterpolationMat(XYZ, 'Fx')
    Pfz = mesh.getInterpolationMat(XYZ, 'Fz')

    Pc = mesh.getInterpolationMat(XYZ, 'CC')
    Zero = sp.csr_matrix(Pc.shape)
    Pcx, Pcz = sp.hstack([Pc, Zero]), sp.hstack([Zero, Pc])

    if ax is None:
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    for a in ax:
        a.grid(
            which='both', linestyle='-', linewidth=0.4,
            color=[0.8, 0.8, 0.8], alpha=0.5
        )
        # getattr(a, 'semilogy' if logScale is True else 'plot')(
        #     [cp.src_a[2], cp.src_a[2]], [1e-14, 1], color=[0.3, 0.3, 0.3]
        # )
        a.set_xlim(xlim)
        a.invert_xaxis()

    col = ['b', 'g', 'r', 'c', 'm', 'y']
    pos_linestyle = ['-', '-']
    neg_linestyle = ['--', '--']
    leg = []

    for i, mur in enumerate(cp.muModels):
        for srcind in srcinds:
            src = survey.getSrcByFreq(survey.freqs[freqind])[srcind]
            j = Utils.mkvc(fields[mur][src, 'j'].copy())

            if subtract is not None:
                j = j - Utils.mkvc(
                    fields[subtract][src, 'j'].copy()
                )

            if real_or_imag == 'real':
                j = j.real
            else:
                j = j.imag

            jx, jz = Pfx * j, Pfz * j

            if logScale is True:
                ax0 = ax[0].semilogy(
                    z_plt, jz, '{linestyle}{color}'.format(
                        linestyle=pos_linestyle[srcind],
                        color=col[i]
                    ),
                    label="{} $\mu_0$".format(mur)
                )
                ax[0].semilogy(
                    z_plt, -jz, '{linestyle}{color}'.format(
                        linestyle=neg_linestyle[srcind],
                        color=col[i]
                    )
                )

                ax[1].semilogy(
                    z_plt, jx, '{linestyle}{color}'.format(
                        linestyle=pos_linestyle[srcind],
                        color=col[i]
                    )
                )
                ax[1].semilogy(
                    z_plt, -jx, '{linestyle}{color}'.format(
                        linestyle=neg_linestyle[srcind],
                        color=col[i]
                    )
                )
            else:
                ax0 = ax[0].plot(
                    z_plt, jz, '{linestyle}{color}'.format(
                        linestyle=pos_linestyle[srcind],
                        color=col[i]
                    ), label="{} $\mu_0$".format(mur)
                )
                ax[1].plot(
                    z_plt, jx, '{linestyle}{color}'.format(
                        linestyle=pos_linestyle[srcind],
                        color=col[i]
                    )
                )

            leg.append(ax0)

    if ylim_0 is not  None:
        ax[0].set_ylim(ylim_0)

    if ylim_1 is not None:
        ax[1].set_ylim(ylim_1)

    ax[0].legend(bbox_to_anchor=[1.25, 1])
    return ax


# plot over mu
def plot_j_over_freq_z(
    cp, fields, mesh, survey, mur=1., r=1., xlim=[-1100., 0.],
    real_or_imag='real', subtract=None, ax=None, logScale=True, srcinds=[0],
    ylim_0=None, ylim_1=None, fig=None
):
    print("mu = {} mu_0".format(mur))

    x_plt = np.r_[r]
    z_plt = np.linspace(xlim[0], xlim[1], int(xlim[1]-xlim[0]))

    XYZ = Utils.ndgrid(x_plt, np.r_[0], z_plt)

    Pfx = mesh.getInterpolationMat(XYZ, 'Fx')
    Pfz = mesh.getInterpolationMat(XYZ, 'Fz')

    Pc = mesh.getInterpolationMat(XYZ, 'CC')
    Zero = sp.csr_matrix(Pc.shape)
    Pcx, Pcz = sp.hstack([Pc, Zero]), sp.hstack([Zero, Pc])

    if ax is None:
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    for a in ax:
        a.grid(
            which='both', linestyle='-', linewidth=0.4,
            color=[0.8, 0.8, 0.8], alpha=0.5
        )

        a.set_xlim(xlim)
        a.invert_xaxis()

    col = ['b', 'g', 'r', 'c', 'm', 'y']
    pos_linestyle = ['-', '-']
    neg_linestyle = ['--', '--']
    leg = []

    for i, freq in enumerate(cp.freqs):
        for srcind in srcinds:
            src = survey.getSrcByFreq(freq)[srcind]
            j = Utils.mkvc(fields[mur][src, 'j'].copy())

            if subtract is not None:
                j = j - Utils.mkvc(
                    fields[subtract][src, 'j'].copy()
                )

            if real_or_imag == 'real':
                j = j.real
            else:
                j = j.imag

            jx, jz = Pfx * j, Pfz * j

            if logScale is True:
                ax0 = ax[0].semilogy(
                    z_plt, jz, '{linestyle}{color}'.format(
                        linestyle=pos_linestyle[srcind],
                        color=col[i]
                    ),
                    label="{} Hz".format(freq)
                )
                ax[0].semilogy(
                    z_plt, -jz, '{linestyle}{color}'.format(
                        linestyle=neg_linestyle[srcind],
                        color=col[i]
                    )
                )

                ax[1].semilogy(
                    z_plt, jx, '{linestyle}{color}'.format(
                        linestyle=pos_linestyle[srcind],
                        color=col[i]
                    )
                )
                ax[1].semilogy(
                    z_plt, -jx, '{linestyle}{color}'.format(
                        linestyle=neg_linestyle[srcind],
                        color=col[i]
                    )
                )
            else:
                ax0 = ax[0].plot(
                    z_plt, jz, '{linestyle}{color}'.format(
                        linestyle=pos_linestyle[srcind],
                        color=col[i]
                    ), label="{} $\mu_0$".format(mur)
                )
                ax[1].plot(
                    z_plt, jx, '{linestyle}{color}'.format(
                        linestyle=pos_linestyle[srcind],
                        color=col[i]
                    )
                )

            leg.append(ax0)

    if ylim_0 is not  None:
        ax[0].set_ylim(ylim_0)

    if ylim_1 is not None:
        ax[1].set_ylim(ylim_1)

    ax[0].legend(bbox_to_anchor=[1.25, 1])
    return ax


# plot over mu
def plot_j_over_mu_x(
    cp, fields, mesh, survey, srcind=0, mur=1, z=-950., real_or_imag='real',
    subtract=None, xlim=[0., 2000.], logScale=True, srcinds=[0],
    ylim_0=None, ylim_1=None, ax=None, fig=None
):
    print("mu = {} mu_0".format(mur))

    x_plt = np.linspace(xlim[0], xlim[1], xlim[1])
    z_plt = np.r_[z]

    XYZ = Utils.ndgrid(x_plt, np.r_[0], z_plt)

    Pfx = mesh.getInterpolationMat(XYZ, 'Fx')
    Pfz = mesh.getInterpolationMat(XYZ, 'Fz')

    Pc = mesh.getInterpolationMat(XYZ, 'CC')
    Zero = sp.csr_matrix(Pc.shape)
    Pcx, Pcz = sp.hstack([Pc, Zero]), sp.hstack([Zero, Pc])

    if ax is None:
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    for a in ax:
        a.grid(
            which='both', linestyle='-', linewidth=0.4, color=[0.8, 0.8, 0.8],
            alpha=0.5
        )
#         a.semilogy([src_a[2], src_a[2]], [1e-14, 1], color=[0.3, 0.3, 0.3])
        a.set_xlim(xlim)
#         a.invert_xaxis()

    col = ['b', 'g', 'r', 'c', 'm', 'y']
    pos_linestyle = ['-', '-']
    neg_linestyle = ['--', '--']
    leg = []

    for i, f in enumerate(cp.freqs):
        for srcind in srcinds:
            src = survey.getSrcByFreq(survey.freqs[freqind])[srcind]
            j = Utils.mkvc(fields[mur][src, 'j'].copy())

            if subtract is not None:
                j = j - Utils.mkvc(
                    fields[subtract][src, 'j'].copy()
                )

            if real_or_imag == 'real':
                j = j.real
            else:
                j = j.imag

            jx, jz = Pfx * j, Pfz * j

            if logScale is True:
                if np.any(jz > 0):
                    ax0 = ax[0].semilogy(
                        x_plt, jz, '{linestyle}{color}'.format(
                            linestyle=pos_linestyle[srcind], color=col[i]
                        ), label="{} $\mu_0$".format(mur)
                    )
                if np.any(jz < 0):
                    ax[0].semilogy(
                        x_plt, -jz, '{linestyle}{color}'.format(
                            linestyle=neg_linestyle[srcind], color=col[i]
                        )
                    )

                if np.any(jx > 0):
                    ax[1].semilogy(x_plt, jx, '{linestyle}{color}'.format(
                        linestyle=pos_linestyle[srcind], color=col[i]
                    ))
                if np.any(jx < 0):
                    ax[1].semilogy(
                        x_plt, -jx, '{linestyle}{color}'.format(
                            linestyle=neg_linestyle[srcind], color=col[i]
                        )
                    )
            else:
                ax0 = ax[0].plot(
                    x_plt, jz, '{linestyle}{color}'.format(
                        linestyle=pos_linestyle[srcind], color=col[i]
                    ), label="{} $\mu_0$".format(mur)
                )
                ax[1].semilogy(x_plt, jx, '{linestyle}{color}'.format(
                    linestyle=pos_linestyle[srcind], color=col[i]
                ))

        leg.append(ax0)

    if ylim_0 is not  None:
        ax[0].set_ylim(ylim_0)

    if ylim_1 is not None:
        ax[1].set_ylim(ylim_1)

    ax[0].legend(bbox_to_anchor=[1.25, 1])
    return ax
