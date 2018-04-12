from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from scipy.spatial import cKDTree
import ipywidgets
import discretize

import properties

from .utils import face3DthetaSlice
# from .run import SimulationDC, SimulationFDEM, SimulationTDEM


def plot_slice(
    mesh, v, ax=None, clim=None, pcolorOpts=None, theta_ind=0,
    cb_extend=None
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
            mirror=True, pcolorOpts=pcolorOpts, clim=clim
        )[0], ax=ax, extend=cb_extend if cb_extend is not None else "neither"
    )

    if clim is not None:
        cb.set_clim(clim)
        cb.update_ticks()

    return ax, cb


def plotFace2D(
    mesh2D,
    j, real_or_imag='real', ax=None, range_x=None,
    range_y=None, sample_grid=None,
    log_scale=True, clim=None, mirror=False, mirror_data=None,
    pcolorOpts=None,
    show_cb=True,
    stream_threshold=None, stream_opts=None
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
    :param bool log_scale: use a log scale for the colorbar?
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    if len(j) == mesh2D.nF:
        vType = 'F'
    elif len(j) == mesh2D.nC*2:
        vType = 'CCv'

    if pcolorOpts is None:
        pcolorOpts = {}

    if log_scale is True:
        pcolorOpts['norm'] = LogNorm()
    else:
        pcolorOpts = {}

    f = mesh2D.plotImage(
        getattr(j, real_or_imag),
        view='vec', vType=vType, ax=ax,
        range_x=range_x, range_y=range_y, sample_grid=sample_grid,
        mirror=mirror,
        mirror_data= (
            getattr(mirror_data, real_or_imag) if mirror_data is not None
            else None)
        ,
        pcolorOpts=pcolorOpts, clim=clim, stream_threshold=stream_threshold,
        streamOpts=stream_opts
    )

    out = f + (ax,)

    if show_cb is True:
        cb = plt.colorbar(f[0], ax=ax)
        out += (cb,)

        # if clim is not None:
        #     cb.set_clim(clim)
        #     cb.update_ticks()

    return out


def plotEdge2D(
    mesh2D,
    h, real_or_imag='real', ax=None, range_x=None,
    range_y=None, sample_grid=None,
    log_scale=True, clim=None, mirror=False, pcolorOpts=None
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
    :param bool log_scale: use a log scale for the colorbar?
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    if len(h) == mesh2D.nE:
        vType = 'E'
    elif len(h) == mesh2D.nC:
        vType = 'CC'
    elif len(h) == 2*mesh2D.nC:
        vType = 'CCv'

    if log_scale is True:
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

    fplt = face3DthetaSlice(
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

    ax.grid('both', linestyle=linestyle, linewidth=0.4, color=[0.8, 0.8, 0.8])
    ax.set_xlabel('distance from well (m)')

    plt.tight_layout()

    return ax


class FieldsViewer(properties.HasProperties):

    eps = properties.Float(
        "small value to add to colorbar so it is not strictly zero for "
        "logarithmic fields",
        default=1e-20
    )

    primary_key=properties.String(
        "key indicating which model we treat as the primary"
    )

    def __init__(
        self, sim_dict, fields_dict, model_keys=None,
        **kwargs
    ):
        self.sim_dict = sim_dict
        self.fields_dict = fields_dict
        self.model_keys = (
            model_keys if model_keys is not None else sorted(sim_dict.keys())
        )

        if all(
            sim.__class__.__name__ == "SimulationDC"
            for sim in sim_dict.values()
        ):
            self._physics = 'DC'
            self.fields_opts = ['e',  'j', 'phi', 'charge']
            self.reim_opts = ["real"]

        elif all(
            sim.__class__.__name__ == "SimulationFDEM"
            for sim in sim_dict.values()
        ):
            self._physics = 'FDEM'
            self.fields_opts = ['e', 'j', 'h', 'b']
            self.reim_opts = ["real", "imag"]

        elif all(
            sim.__class__.__name__ == "SimulationTDEM"
            for sim in sim_dict.values()
        ):
            self._physics = 'TDEM'
            self.fields_opts = ['e', 'j']
            if self.sim_dict[model_keys[0]].prob._fieldType=="j":
                self.fields_opts += ["charge"]
            elif self.sim_dict[model_keys[0]].prob._fieldType in ["b", "h"]:
                self.fields_opts += ['h', 'b']
            self.reim_opts = ["real"]

    @property
    def prim_sec_opts(self):
        if self.primary_key is not None:
            assert primary_key in self.model_keys, (
                'the provided primary_key {} is not in {}'.format(
                    primary_key, self.model_keys
                )
            )
            return ['total', 'primary', 'secondary']
        else:
            return None

    def _mesh2D(self, model_key):
        if self.sim_dict[model_key].meshGenerator.mesh.isSymmetric:
            return self.sim_dict[model_key].meshGenerator.mesh
        return self.sim_dict[model_key].meshGenerator.create_2D_mesh().mesh

    def _mesh(self, model_key):
        return self.sim_dict[model_key].meshGenerator.mesh

    def _check_inputs(
        self, model_key, view, prim_sec, real_or_imag
    ):
        def error_statement(field, provided, allowed):
            return (
                "The provided {field}, {provided}, is not in the allowed "
                "{field} list {allowed}".format(
                    field=field, provided=provided, allowed=allowed
                )
            )

        def run_assert(field, provided, allowed):
            assert provided in allowed, (
                error_statement(field, provided, allowed)
            )

        # check inputs are valid
        run_assert('model_key', model_key, self.model_keys)
        run_assert('view', view, self.fields_opts)

        if self.prim_sec_opts is not None:
            run_assert('prim_sec', prim_sec, self.prim_sec_opts)

        if self._physics in ['DC', 'TDEM']:
            run_assert('real_or_imag', real_or_imag, ['real'])
        elif self._physics == 'FDEM':
            run_assert('real_or_imag', real_or_imag,['real', 'imag'])


    def plot_cross_section(
        self,
        ax=None,
        model_key=None,
        xlim=None,
        zlim=None,
        clim=None,
        view=None,
        prim_sec=None,
        real_or_imag='real',
        theta_ind=0,
        src_ind=0,
        time_ind=0,
        casing_outline=True,
        cb_extend=None,
        show_cb=True,
        show_mesh=False,
        use_aspect=False,
        stream_opts=None
    ):
        """
        Plot the fields
        """

        # create default at
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        # set defaults and check inputs
        if model_key is None:
            model_key = self.model_keys[0]

        if view is None:
            view = self.fields_opts[0]

        if prim_sec is None and self.prim_sec_opts is not None:
            prim_sec = 'total'

        self._check_inputs(
            model_key, view, prim_sec, real_or_imag
        )

        # get background model if doing primsec
        if prim_sec == 'primary':
            model_key = self.primary_key

        # grab relevant parameters
        src = self.sim_dict[model_key].survey.srcList[src_ind]
        if self._physics == "TDEM":
            plotme = self.fields_dict[model_key][src, view, time_ind]
        else:
            plotme = self.fields_dict[model_key][src, view]

        mesh = self._mesh(model_key)
        norm = None

        if prim_sec == 'secondary':
            prim_src = self.sim_dict[self.primary_key].survey.srcList[src_ind]
            if self._physics == "TDEM":
                background = self.fields_dict[self.primary_key][
                    prim_src, view, time_ind
                ]
            else:
                background = self.fields_dict[self.primary_key][
                    prim_src, view
                ]
            plotme = plotme - background

        if not mesh.isSymmetric:
            theta_ind_mirror = (
                theta_ind+int(mesh.vnC[1]/2)
                if theta_ind < int(mesh.vnC[1]/2)
                else theta_ind-int(mesh.vnC[1]/2)
            )
        else:
            theta_ind_mirror = 0
            mirror_data=None

        if view in ['charge', 'phi']:
            plot_type = "scalar"
            if clim is None and view == 'charge':
                clim = np.r_[-1., 1.] * np.max(np.absolute(plotme))

            # if not mesh.isSymmetric:
            plotme = plotme.reshape(mesh.vnC, order='F')
            mirror_data = discretize.utils.mkvc(
                plotme[:, theta_ind_mirror, :]
            )
            plotme = discretize.utils.mkvc(plotme[:, theta_ind, :])


        elif view in ['j', 'e']:
            if self.sim_dict[model_key].prob._formulation == "HJ":
                plt_vec = face3DthetaSlice(
                    self._mesh(model_key), plotme,
                    theta_ind=theta_ind
                )
                mirror_data = face3DthetaSlice(
                    mesh, plotme, theta_ind=theta_ind_mirror
                )
                plot_type = "vec"

            elif self.sim_dict[model_key].prob._formulation == "EB":
                plotme = mesh.aveE2CC * plotme
                mirror_data = -plotme
                plot_type = "scalar"
                norm = SymLogNorm(
                    clim[0] if clim is not None else
                    np.max([self.eps, np.min(np.absolute(plotme))])
                )
                clim = clim[1]*np.r_[-1., 1.] if clim is not None else None

        elif view in ['h', 'b']:

            if self.sim_dict[model_key].prob._formulation == "EB":
                plt_vec = face3DthetaSlice(
                    mesh, plotme, theta_ind=theta_ind
                )
                mirror_data = face3DthetaSlice(
                    mesh, plotme, theta_ind=theta_ind_mirror
                )
                plot_type = "vec"

            elif self.sim_dict[model_key].prob._formulation == "HJ":
                plotme = mesh.aveE2CC * plotme
                mirror_data = discretize.utils.mkvc(
                    -plotme[:, theta_ind_mirror, :]
                )
                plot_type = "scalar"
                norm = SymLogNorm(
                    clim[0] if clim is not None else
                    np.max([self.eps, np.min(np.absolute(plotme))])
                )
                clim = clim[1]*np.r_[-1., 1.] if clim is not None else None

        if plot_type == "scalar":
            out = self._mesh2D(model_key).plotImage(
                getattr(plotme, real_or_imag), ax=ax,
                pcolorOpts = {
                    'cmap': 'bwr' if view == 'charge' else 'viridis',
                    'norm': norm
                },
                clim=clim,
                mirror_data=mirror_data,
                mirror=True
            )

            if show_cb:
                cb = plt.colorbar(
                    out[0], ax=ax,
                    extend='neither' if cb_extend is None else cb_extend,
                )

                out += (cb,)

        elif plot_type == "vec":
            out = plotFace2D(
                self._mesh2D(model_key),
                plt_vec + self.eps,
                real_or_imag=real_or_imag,
                ax=ax,
                range_x=xlim,
                range_y=zlim,
                sample_grid=(
                    np.r_[np.diff(xlim)/100., np.diff(zlim)/100.]
                    if xlim is not None and zlim is not None else None
                ),
                log_scale=True,
                clim=clim,
                stream_threshold=clim[0] if clim is not None else None,
                mirror=True,
                mirror_data=mirror_data,
                stream_opts=stream_opts,
                show_cb=show_cb,
            )

        if clim is not None and show_cb is True:
            cb = out[-1]
            cb.set_clim(clim)
            cb.update_ticks()

        if show_mesh is True:
            self._mesh2D(model_key).plotGrid(ax=ax)

        title = "{} \n{} {}".format(model_key, prim_sec, view)
        if self._physics == "FDEM":
            title += "\nf = {:1.1e} Hz".format(src.freq)
        elif self._physics == "TDEM":
            title += "\n t = {:1.1e} s".format(
                self.sim_dict[model_key].prob.times[time_ind]
            )
        ax.set_title(
            title, fontsize=13
        )
        ax.set_xlim(xlim)
        ax.set_ylim(zlim)

        # plot outline of casing
        if casing_outline is True:
            m = self.sim_dict[model_key].modelParameters
            factor = [-1, 1]
            [
                ax.plot(
                    fact * np.r_[
                        m.casing_a, m.casing_a, m.casing_b,
                        m.casing_b, m.casing_a
                    ],
                    np.r_[
                        m.casing_z[1], m.casing_z[0], m.casing_z[0],
                        m.casing_z[1], m.casing_z[1]
                    ],
                    'k',
                    lw = 0.5
                )
                for fact in factor
            ]

        if use_aspect is True:
            ax.set_aspect(1)

        return out

    def _get_cKDTree(self, model_key, plan_mesh, k=10):
        # construct interpolation
        mesh = self._mesh(model_key)
        CCcart = mesh.cartesianGrid('CC')
        tree = cKDTree(CCcart[:mesh.vnC[:2].prod(),:2])
        d, ii = tree.query(plan_mesh.gridCC, k=k)

        weights = 1./d
        weights = discretize.utils.sdiag(1./weights.sum(1))*weights

        return ii, weights

    def plot_depth_slice(
        self,
        ax=None,
        model_key=None,
        xlim=None,
        ylim=None,
        clim=None,
        z_ind=None,
        view=None,
        prim_sec=None,
        real_or_imag='real',
        src_ind=0,
        time_ind=0,
        # casing_outline=True,
        cb_extend=None,
        show_cb=True,
        # show_mesh=False,
        use_aspect=False,
        stream_opts=None,
        plan_mesh=None,
        tree_query=None,
        rotate=False
    ):

        # create default ax
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        # set defaults and check inputs
        if model_key is None:
            model_key = self.model_keys[0]

        if view is None:
            view = self.fields_opts[0]

        if prim_sec is None and self.prim_sec_opts is not None:
            prim_sec = 'total'

        self._check_inputs(
            model_key, view, prim_sec, real_or_imag
        )

        # get background model if doing primsec
        if prim_sec == 'primary':
            model_key = self.primary_key

        # grab relevant parameters
        src = self.sim_dict[model_key].survey.srcList[src_ind]
        if self._physics == "TDEM":
            plotme = self.fields_dict[model_key][src, view, time_ind]
        else:
            plotme = self.fields_dict[model_key][src, view]
        mesh = self._mesh(model_key)

        if prim_sec == 'secondary':
            prim_src = self.sim_dict[self.primary_key].survey.srcList[src_ind]
            if self._physics == "TDEM":
                background = self.fields_dict[self.primary_key][
                    prim_src, view, time_ind
                ]
            else:
                background = self.fields_dict[self.primary_key][
                    prim_src, view
                ]
            plotme = plotme - background

        # interpolate to cell centers
        if view in ['e', 'j']:
            if self.sim_dict[model_key].prob._formulation == 'HJ':
                plotme = mesh.aveF2CCV * plotme
            elif self.sim_dict[model_key].prob._formulation == 'EB':
                plotme = mesh.aveE2CCV * plotme
        elif view in ['b', 'h']:
            if self.sim_dict[model_key].prob._formulation == 'HJ':
                plotme = mesh.aveE2CCV * plotme
            elif self.sim_dict[model_key].prob._formulation == 'EB':
                plotme = mesh.aveF2CCV * plotme

        plotme = plotme.reshape(mesh.gridCC.shape, order='F')
        plotme_cart = discretize.utils.rotate_vec_cyl2cart(mesh.gridCC, plotme)

        # construct plan mesh if it doesn't exist
        if plan_mesh is None:
            nC = 200
            hx = np.diff(xlim) * np.ones(nC)/ nC
            hy = np.diff(ylim) * np.ones(nC)/ nC
            plan_mesh = discretize.TensorMesh([hx, hy], x0=[xlim[0], ylim[0]])

        # construct interpolation
        if tree_query is None:
            inds, weights = self._get_cKDTree(model_key, plan_mesh)

        else:
            inds, weights = tree_query

        # deal with vectors
        if view in ['e', 'b', 'h', 'j']:
            plotme_x = discretize.utils.mkvc(
                (plotme_cart[:, 0]).reshape(mesh.vnC, order='F')[:, :, z_ind]
            )
            plotme_y = discretize.utils.mkvc(
                (plotme_cart[:, 1]).reshape(mesh.vnC, order='F')[:, :, z_ind]
            )

            plotme = np.hstack([
                (plotme_x[inds] * weights).sum(1),
                (plotme_y[inds] * weights).sum(1)
            ])

            if rotate is True:
                plotme_x = discretize.utils.mkvc(
                    plotme[:plan_mesh.nC].reshape(plan_mesh.vnC, order='F').T
                )
                plotme_y = discretize.utils.mkvc(
                    plotme[plan_mesh.nC:].reshape(plan_mesh.vnC, order='F').T
                )
                plotme = np.hstack([plotme_y, plotme_x])

        else:
            plotme = (plotme[inds] * weights).sum(1)

            if rotate is True:
                plotme = plotme.reshape(plan_mesh.vnC, order='F').T

        if view in ['e', 'b', 'h', 'j']:
            out = plan_mesh.plotImage(
                plotme, view='vec', vType='CCv', ax=ax,
                pcolorOpts={'norm':LogNorm()},
                clim=clim,
                streamOpts=stream_opts,
            )

        if show_cb:
            out += plt.colorbar(out[0], ax=ax),

        if use_aspect is True:
            ax.set_aspect(1)

        title = "{} \n{} {} \n z={:1.1e}m".format(
            model_key, prim_sec, view, mesh.vectorCCz[z_ind]
        )
        if self._physics == "FDEM":
            title += "\nf = {:1.1e} Hz".format(src.freq)
        elif self._physics == "TDEM":
            title += "\n t = {:1.1e} s".format(
                self.sim_dict[model_key].prob.times[time_ind]
            )
        ax.set_title(title)
        return out

    def _cross_section_widget_wrapper(
        self,
        ax=None,
        max_r=None,
        min_depth=None,
        max_depth=None,
        clim_min=None,
        clim_max=None,
        model_key=None,
        view=None,
        prim_sec=None,
        real_or_imag='real',
        src_ind=0,
        theta_ind=0,
        time_ind=0,
        show_mesh=False,
        use_aspect=False,
        casing_outline=True,
        figwidth=5
    ):

        if isinstance(model_key, str):
            if model_key == 'all':
                model_key = self.model_keys
            else:
                model_key = [model_key]

        if ax is None:
            fig, ax = plt.subplots(
                1, len(model_key), figsize=(len(model_key)*figwidth, 6)
            )

        if len(model_key) == 1:
            ax = [ax]

        clim = None
        if clim_max is not None and clim_max != 0:
            if view in ['charge', 'phi']:
                clim = np.r_[-1., 1.]*clim_max
            else:
                clim = np.r_[self.eps, clim_max]

            if clim_min is not None:
                clim[0] = clim_min

        for a, mod in zip(ax, model_key):

            self.plot_cross_section(
                ax=a, model_key=mod,
                xlim=max_r*np.r_[-1., 1.], zlim=np.r_[-max_depth, -min_depth],
                clim=clim, view=view, prim_sec=prim_sec,
                real_or_imag=real_or_imag,
                theta_ind=theta_ind, src_ind=src_ind, time_ind=time_ind,
                casing_outline=casing_outline,
                cb_extend=None, show_cb=True,
                show_mesh=show_mesh, use_aspect=use_aspect
            )

        plt.tight_layout()
        plt.show()

    def widget_cross_section(self, ax=None, defaults={}, fixed={}, figwidth=5):

        widget_defaults = {
            "max_r": 2*self.sim_dict[self.model_keys[0]].modelParameters.casing_b,
            "min_depth": -10.,
            "max_depth": 1.25*self.sim_dict[self.model_keys[0]].modelParameters.casing_l,
            "clim_min": 0,
            "clim_max": 0,
            "model_key": self.model_keys[0],
            "view": self.fields_opts[0],
            "show_mesh": False,
            "use_aspect": False,
            "casing_outline": True,
        }

        [
            widget_defaults.pop(key) for key in fixed.keys()
            if key in widget_defaults.keys()
        ]

        fixed["ax"] = ax
        fixed["figwidth"] = figwidth

        if not self.sim_dict[self.model_keys[0]].meshGenerator.mesh.isSymmetric:
            widget_defaults["theta_ind"]=0
        else:
            fixed["theta_ind"] = 0

        if len(self.sim_dict[self.model_keys[0]].survey.srcList) == 1:
            fixed["src_ind"] = 0
        else:
            widget_defaults["src_ind"] = 0

        if self._physics == "TDEM":
            widget_defaults["time_ind"] = 0
        else:
            fixed["time_ind"] = None

        if self._physics == "FDEM":
            widget_defaults["real_or_imag"] = "real"
        else:
            fixed["real_or_imag"] = "real"

        if self.primary_key is not None:
            widget_defaults["prim_sec"] = "total"
        else:
            fixed["prim_sec"] = "total"

        for key, val in defaults.iteritems():
            widget_defaults[key] = val

        widget_dict = {
            key: ipywidgets.fixed(value=val) for key, val in fixed.iteritems()
        }

        for key in ["max_r", "min_depth", "max_depth", "clim_min", "clim_max"]:
            if key in widget_defaults.keys():
                widget_dict[key] = ipywidgets.FloatText(
                    value=widget_defaults[key]
                )

        for key, option in zip(
            ["model_key", "view", "prim_sec", "real_or_imag"],
            ["model_keys", "fields_opts", "prim_sec_opts", "reim_opts"]
        ):
            if key in widget_defaults.keys():
                options = getattr(self, option, None)
                if options is None:
                    options = option
                if key == "model_key":
                    options = options + ["all"]
                widget_dict[key] = ipywidgets.ToggleButtons(
                    options=options,
                    value=widget_defaults[key]
                )

        for key, max_len in zip(
            ["theta_ind", "src_ind", "time_ind"],
            [
                self.sim_dict[self.model_keys[0]].meshGenerator.mesh.vnC[1] - 1,
                len(self.sim_dict[self.model_keys[0]].survey.srcList) - 1,
                self.sim_dict[self.model_keys[0]].prob.nT if
                self._physics == "TDEM" else None
            ]
        ):
            if key in widget_defaults.keys():
                widget_dict[key] = ipywidgets.IntSlider(
                    min=0, max=max_len, value=widget_defaults[key]
                )

        for key in ["show_mesh", "use_aspect", "casing_outline"]:
            if key in widget_defaults.keys():
                widget_dict[key] = ipywidgets.Checkbox(
                    value=widget_defaults[key]
                )

        return ipywidgets.interact(
            self._cross_section_widget_wrapper,
            **widget_dict
        )

    def _depth_slice_widget_wrapper(
        self,
        ax=None,
        max_r=None,
        clim_min=None,
        clim_max=None,
        model_key=None,
        view=None,
        prim_sec=None,
        real_or_imag='real',
        z_ind=None,
        src_ind=0,
        time_ind=0,
        use_aspect=False,
        # casing_outline=True,
        rotate=False,
        figwidth=5
    ):
        if isinstance(model_key, str):
            if model_key == 'all':
                model_key = self.model_keys
            else:
                model_key = [model_key]

        if ax is None:
            fig, ax = plt.subplots(
                1, len(model_key), figsize=(len(model_key)*figwidth, 6)
            )

        if len(model_key) == 1:
            ax = [ax]

        clim = None
        if clim_max is not None and clim_max != 0:
            if view in ['charge', 'phi']:
                clim = np.r_[-1., 1.]*clim_max
            else:
                clim = np.r_[self.eps, clim_max]

            if clim_min is not None:
                clim[0] = clim_min

        xlim = max_r * np.r_[-1., 1.]
        ylim = max_r * np.r_[-1., 1.]

        def assign_cKDTree():
            nC = 200
            hx = np.diff(xlim) * np.ones(nC)/ nC
            hy = np.diff(ylim) * np.ones(nC)/ nC
            plan_mesh = discretize.TensorMesh([hx, hy], x0=[xlim[0], ylim[0]])

            inds, weights = self._get_cKDTree(model_key[0], plan_mesh)

            self._cached_cKDTree = {
                'xlim': xlim,
                'ylim': ylim,
                'inds': inds,
                'weights': weights
            }

        if getattr(self, '_cached_cKDTree', None) is None:
            assign_cKDTree()
        elif not (
            np.all(self._cached_cKDTree['xlim'] == xlim) &
            np.all(self._cached_cKDTree['ylim'] == ylim)
        ):
            assign_cKDTree()

        for a, mod in zip(ax, model_key):

            self.plot_depth_slice(
                ax=a, model_key=mod, xlim=xlim, ylim=ylim,
                clim=clim, z_ind=z_ind, view=view, prim_sec=prim_sec,
                real_or_imag=real_or_imag,
                src_ind=src_ind, time_ind=time_ind,
                # casing_outline=casing_outline,
                cb_extend=None, show_cb=True,
                use_aspect=use_aspect,
                tree_query=(
                    self._cached_cKDTree['inds'],
                    self._cached_cKDTree['weights']
                ),
                rotate=rotate
            )

        plt.tight_layout()
        plt.show()


    def widget_depth_slice(self, ax=None, defaults={}, fixed={}, figwidth=5):

        widget_defaults = {
            "max_r": self.sim_dict[self.model_keys[0]].modelParameters.casing_l,
            "z_ind": int(
                self.sim_dict[self.model_keys[0]].meshGenerator.mesh.vnC[2]/2.
            ),
            "clim_min": 0,
            "clim_max": 0,
            "model_key": self.model_keys[0],
            "view": self.fields_opts[0],
            "show_mesh": False,
            "use_aspect": False,
            "rotate":False
            # "casing_outline": True
        }

        [
            widget_defaults.pop(key) for key in fixed.keys()
            if key in widget_defaults.keys()
        ]


        fixed["ax"] = ax
        fixed["figwidth"] = figwidth

        if len(self.sim_dict[self.model_keys[0]].survey.srcList) == 1:
            fixed["src_ind"] = 0
        else:
            widget_defaults["src_ind"] = 0

        if self._physics == "TDEM":
            widget_defaults["time_ind"] = 0
        else:
            fixed["time_ind"] = None

        if self._physics == "FDEM":
            widget_defaults["real_or_imag"] = "real"
        else:
            fixed["real_or_imag"] = "real"

        if self.primary_key is not None:
            widget_defaults["prim_sec"] = "total"
        else:
            fixed["prim_sec"] = "total"

        for key, val in defaults.iteritems():
            widget_defaults[key] = val

        widget_dict = {
            key: ipywidgets.fixed(value=val) for key, val in fixed.iteritems()
        }

        for key in ["max_r", "clim_min", "clim_max"]:
            if key in widget_defaults.keys():
                widget_dict[key] = ipywidgets.FloatText(
                    value=widget_defaults[key]
                )

        for key, option in zip(
            ["model_key", "view", "prim_sec", "real_or_imag"],
            ["model_keys", "fields_opts", "prim_sec_opts", ["real", "imag"]]
        ):
            if key in widget_defaults.keys():
                options = getattr(self, option, None)
                if options is None:
                    options = option
                if key == "model_key":
                    options = options + ["all"]
                widget_dict[key] = ipywidgets.ToggleButtons(
                    options=options,
                    value=widget_defaults[key]
                )

        for key, max_len in zip(
            ["z_ind", "src_ind", "time_ind"],
            [
                self.sim_dict[self.model_keys[0]].meshGenerator.mesh.vnC[2] - 1,
                len(self.sim_dict[self.model_keys[0]].survey.srcList) - 1,
                self.sim_dict[self.model_keys[0]].prob.nT if
                self._physics == "TDEM" else None
            ]
        ):
            if key in widget_defaults.keys():
                widget_dict[key] = ipywidgets.IntSlider(
                    min=0, max=max_len, value=widget_defaults[key]
                )

        for key in ["show_mesh", "use_aspect", "rotate"]:
            if key in widget_defaults.keys():
                widget_dict[key] = ipywidgets.Checkbox(
                    value=widget_defaults[key]
                )

        return ipywidgets.interact(
            self._depth_slice_widget_wrapper,
            **widget_dict
        )
