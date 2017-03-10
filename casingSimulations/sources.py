import numpy as np
import matplotlib.pyplot as plt

import properties
import discretize

from SimPEG import Utils
from SimPEG.EM import FDEM


class BaseCasingSrc(object):
    def __init__(self, cp, mesh, **kwargs):
        assert cp.src_a[1] == cp.src_b[1], (
            'non y-axis aligned sources have not been implemented'
        )
        self.mesh = mesh
        self.src_a = cp.src_a
        self.src_b = cp.src_b
        self.casing_a = cp.casing_a
        self.freqs = cp.freqs
        Utils.setKwargs(self, **kwargs)


class DownHoleTerminatingSrc(BaseCasingSrc):
    """
    A source that terminates down-hole. It is not coupled to the casing

    :param CasingSimulations.Model.CasingProperties cp: a casing properties instance
    :param discretize.BaseMesh mesh: a discretize mesh
    """

    def __init__(self, cp, mesh):
        super(DownHoleTerminatingSrc, self).__init__(cp, mesh)

    @property
    def wire_in_borehole(self):
        """
        Indices of the verically directed wire inside of the borehole. It goes
        through the center of the well
        """

        if getattr(self, '_wire_in_borehole', None) is None:
            mesh = self.mesh
            src_a = self.src_a
            src_b = self.src_b

            wire_in_boreholex = (mesh.gridFz[:, 0] < mesh.hx.min())
            wire_in_boreholez = (
                (mesh.gridFz[:, 2] >= src_a[2] - 0.5*mesh.hz.min()) &
                (mesh.gridFz[:, 2] < src_b[2] + 1.5*mesh.hz.min())
            )

            self._wire_in_borehole = wire_in_boreholex & wire_in_boreholez

            if getattr(mesh, 'isSymmetric', None) is not True:
                wire_in_boreholey = (
                    (mesh.gridFz[:, 1] > src_a[1] - mesh.hy.min()/2.) &
                    (mesh.gridFz[:, 1] < src_a[1] + mesh.hy.min()/2.)
                )
                self._wire_in_borehole = (
                    self._wire_in_borehole & wire_in_boreholey
                )

        return self._wire_in_borehole

    @property
    def surface_wire(self):
        """
        Horizontal part of the wire that runs along the surface
        (one cell above) from the center of the well to the return electrode
        """
        if getattr(self, '_surface_wire', None) is None:
            mesh = self.mesh
            src_a = self.src_a
            src_b = self.src_b

            # horizontally directed wire
            surface_wirex = (mesh.gridFx[:, 0] <= src_b[0])
            surface_wirez = (
                (mesh.gridFx[:, 2] > mesh.hz.min()) &
                (mesh.gridFx[:, 2] <= 1.75*mesh.hz.min())
            )
            self._surface_wire = surface_wirex & surface_wirez

            if getattr(mesh, 'isSymmetric', None) is not True:
                surface_wirey = (
                    (mesh.gridFx[:, 1] > src_b[1] - mesh.hy.min()/2.) &
                    (mesh.gridFx[:, 1] < src_b[1] + mesh.hy.min()/2.)
                )

                self._surface_wire = self._surface_wire & surface_wirey

        return self._surface_wire

    @property
    def surface_electrode(self):
        """
        Return electrode on the surface
        """
        if getattr(self, '_surface_electrode', None) is None:
            mesh = self.mesh
            src_a = self.src_a
            src_b = self.src_b

            # return electrode
            surface_electrodex = (
                (mesh.gridFz[:, 0] > src_b[0]*0.9) &
                (mesh.gridFz[:, 0] < src_b[0]*1.1)
            )
            surface_electrodez = (
                (mesh.gridFz[:, 2] >= src_b[2] - mesh.hz.min()) &
                (mesh.gridFz[:, 2] < src_b[2] + 1.75*mesh.hz.min())
            )
            self._surface_electrode = surface_electrodex & surface_electrodez

            if getattr(mesh, 'isSymmetric', None) is not True:
                surface_electrodey = (
                    (mesh.gridFz[:, 1] > src_b[1] - mesh.hy.min()/2.) &
                    (mesh.gridFz[:, 1] < src_b[1] + mesh.hy.min()/2.)
                )
                self._surface_electrode = (
                    self._surface_electrode & surface_electrodey
                )

        return self._surface_electrode

    @property
    def srcList(self):
        """
        Source List
        """
        if getattr(self, '_srcList', None) is None:
            # downhole source
            dg_x = np.zeros(self.mesh.vnF[0], dtype=complex)
            dg_y = np.zeros(self.mesh.vnF[1], dtype=complex)
            dg_z = np.zeros(self.mesh.vnF[2], dtype=complex)

            dg_z[self.wire_in_borehole] = -1.  # part of wire through borehole
            dg_x[self.surface_wire] = -1.  # horizontal part of wire along surface
            dg_z[self.surface_electrode] = 1.  # vertical part of return electrode

            # assemble the source (downhole grounded primary)
            dg = np.hstack([dg_x, dg_y, dg_z])
            srcList = [
                FDEM.Src.RawVec_e([], _, dg/self.mesh.area) for _ in self.freqs
            ]
            self._srcList = srcList
        return self._srcList

    def plot(self, ax=None):
        """
        Plot the source.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        mesh = self.mesh

        ax.plot(
            mesh.gridFz[self.wire_in_borehole, 0],
            mesh.gridFz[self.wire_in_borehole, 2], 'rv'
        )
        ax.plot(
            mesh.gridFz[self.surface_electrode, 0],
            mesh.gridFz[self.surface_electrode, 2], 'r^'
        )
        ax.plot(
            mesh.gridFx[self.surface_wire, 0],
            mesh.gridFx[self.surface_wire, 2], 'r<'
        )

    def validate(self):
        """
        Make sure that each segment of the wire is only going through a
        single face

        .. todo:: check that
        """
        # check the surface electrode only has one x and one y location
        surface_electrode = self.mesh.gridFz[self.surface_electrode, :]
        assert len(np.unique(surface_electrode[:, 0])) == 1, (
            'the surface electrode has more than one x-location'
        )
        assert len(np.unique(surface_electrode[:, 1])) == 1, (
            'the surface electrode has more than one y-location'
        )

        # check the surface wire only has one y and one z location
        surface_wire = self.mesh.gridFx[self.surface_wire, :]
        assert len(np.unique(surface_wire[:, 1])) == 1, (
            'the surface wire has more than one y-location'
        )
        assert len(np.unique(surface_wire[:, 2])) == 1, (
            'the surface wire has more than one z-location'
        )

        # check that the wire inside the borehole has only one x, y, location
        wire_in_borehole = self.mesh.gridFz[self.wire_in_borehole, :]
        assert len(np.unique(wire_in_borehole[:, 0])) == 1, (
            'the wire in borehole has more than one x-location'
        )
        assert len(np.unique(wire_in_borehole[:, 1])) == 1, (
            'the wire in borehole has more than one y-location'
        )
        return True


# Source Grounded on Casing
class DownHoleCasingSrc(DownHoleTerminatingSrc):
    """
    Source that is coupled to the casing down-hole and has a return electrode
    at the surface.

    :param CasingSimulations.Model.CasingProperties cp: a casing properties instance
    :param discretize.CylMesh mesh: a cylindrical mesh
    """

    def __init__(self, cp, mesh):
        super(DownHoleCasingSrc, self).__init__(cp, mesh)

    @property
    def downhole_electrode(self):
        """
        Down-hole horizontal part of the wire, coupled to the casing
        """
        if getattr(self, '_downhole_electrode', None) is None:
            mesh = self.mesh
            src_a = self.src_a
            src_b = self.src_b

            # couple to the casing downhole - top part
            downhole_electrode_indx = mesh.gridFx[:, 0] <= self.casing_a  # + mesh.hx.min()*2

            # couple to the casing downhole - bottom part
            downhole_electrode_indz2 = (
                (mesh.gridFx[:, 2] <= src_a[2]) &
                (mesh.gridFx[:, 2] > src_a[2] - mesh.hz.min())
            )

            self._downhole_electrode = (
                downhole_electrode_indx & downhole_electrode_indz2
            )

            if getattr(mesh, 'isSymmetric', None) is not True:
                dowhhole_electrode_indy = (
                    (mesh.gridFx[:, 1] > src_a[1] - mesh.hy.min()/2.) &
                    (mesh.gridFx[:, 1] < src_a[1] + mesh.hy.min()/2.)
                )
                self._downhole_electrode = (
                    self._downhole_electrode & dowhhole_electrode_indy
                )

        return self._downhole_electrode

    @property
    def srcList(self):
        """
        Source List
        """
        if getattr(self, '_srcList', None) is None:
            # downhole source
            dg_x = np.zeros(self.mesh.vnF[0], dtype=complex)
            dg_y = np.zeros(self.mesh.vnF[1], dtype=complex)
            dg_z = np.zeros(self.mesh.vnF[2], dtype=complex)

            dg_z[self.wire_in_borehole] = -1.  # part of wire through borehole
            dg_x[self.downhole_electrode] = 1.  # downhole hz part of wire
            dg_x[self.surface_wire] = -1.  # horizontal part of wire along surface
            dg_z[self.surface_electrode] = 1.  # vertical part of return electrode

            # assemble the source (downhole grounded primary)
            dg = np.hstack([dg_x, dg_y, dg_z])
            srcList = [
                FDEM.Src.RawVec_e([], _, dg/self.mesh.area) for _ in self.freqs
            ]
            self._srcList = srcList
        return self._srcList

    def plot(self, ax=None):
        """
        Plot the source.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        mesh = self.mesh
        super(DownHoleCasingSrc, self).plot(ax=ax)

        ax.plot(
            mesh.gridFx[self.downhole_electrode, 0],
            mesh.gridFx[self.downhole_electrode, 2], 'r>'
        )

        return ax

    def validate(self):
        """
        Make sure that each segment of the wire is only going through a
        single face

        .. todo:: check that
        """

        super(DownHoleCasingSrc, self).validate()

        # check that the down-hole electrode has only one y, one z location
        downhole_electrode = self.mesh.gridFx[self.downhole_electrode, :]
        assert len(np.unique(downhole_electrode[:, 1])) == 1, (
            'the downhole electrode has more than one y-location'
        )
        assert len(np.unique(downhole_electrode[:, 2])) == 1, (
            'the downhole electrode has more than one z-location'
        )
        return True


class TopCasingSrc(BaseCasingSrc):
    """
    Source that has one electrode coupled to the top of the casing, one return
    electrode and a wire in between. This source is set up to live on faces.

    :param discretize.CylMesh mesh: the cylindrical simulation mesh
    :param CasingSimulations cp: Casing parameters object
    """
    def __init__(self, cp, mesh):
        self.mesh = mesh
        self.src_a = cp.src_a
        self.src_b = cp.src_b
        self.casing_a = cp.casing_a
        self.freqs = cp.freqs

    @property
    def tophole_electrode(self):
        """
        Indices of the electrode that is grounded on the top of the casing
        """

        if getattr(self, '_tophole_electrode', None) is None:
            mesh = self.mesh
            src_a = self.src_a
            src_b = self.src_b

            tophole_electrodex = (
                (mesh.gridFz[:, 0] <= self.casing_a + mesh.hx.min()/2.) &
                (mesh.gridFz[:, 0] > self.casing_a - mesh.hx.min()/2.)
            )

            tophole_electrodez = (
                (mesh.gridFz[:, 2] < src_a[2] + 1.5*mesh.hz.min()) &
                (mesh.gridFz[:, 2] >= src_a[2] - 0.5*mesh.hz.min())
            )

            self._tophole_electrode = tophole_electrodex & tophole_electrodez

            if getattr(mesh, 'isSymmetric', None) is not True:
                tophole_electrodey = (
                    (mesh.gridFz[:, 1] > src_a[1] - mesh.hy.min()) &
                    (mesh.gridFz[:, 1] < src_a[1] + mesh.hy.min())
                )
                self._tophole_electrode = (
                    self._tophole_electrode & tophole_electrodey
                )

        return self._tophole_electrode

    @property
    def surface_wire(self):
        """
        indices of the wire that runs along the surface
        """
        if getattr(self, '_surface_wire', None) is None:
            mesh = self.mesh
            src_a = self.src_a
            src_b = self.src_b

            # horizontally directed wire
            surface_wirex = (
                (mesh.gridFx[:, 0] <= src_b[0]) &
                (mesh.gridFx[:, 0] > self.casing_a)
            )
            surface_wirez = (
                (mesh.gridFx[:, 2] > src_b[2] + mesh.hz.min()) &
                (mesh.gridFx[:, 2] <= src_b[2] + 1.75*mesh.hz.min())
            )
            self._surface_wire = surface_wirex & surface_wirez

            if getattr(mesh, 'isSymmetric', None) is not True:
                surface_wirey = (
                    (mesh.gridFx[:, 1] < src_b[1] + mesh.hy.min()/2.) &
                    (mesh.gridFx[:, 1] > src_b[1] - mesh.hy.min()/2.)
                )
                self._surface_wire = self._surface_wire & surface_wirey
        return self._surface_wire

    @property
    def surface_electrode(self):
        """
        indices of the return electrode at the surface
        """
        if getattr(self, '_surface_electrode', None) is None:
            mesh = self.mesh
            src_a = self.src_a
            src_b = self.src_b

            # return electrode
            surface_electrodex = (
                (mesh.gridFz[:, 0] > src_b[0]*0.9) &
                (mesh.gridFz[:, 0] < src_b[0]*1.1)
            )
            surface_electrodez = (
                (mesh.gridFz[:, 2] > -0.5*mesh.hz.min()) &
                (mesh.gridFz[:, 2] < 1.5*mesh.hz.min())
            )
            self._surface_electrode = surface_electrodex & surface_electrodez

            if getattr(mesh, 'isSymmetric', None) is not True:
                surface_electrodey = (
                    (mesh.gridFz[:, 1] < src_b[1] + mesh.hy.min()) &
                    (mesh.gridFz[:, 1] > src_b[1] - mesh.hy.min())
                )
                self._surface_electrode = (
                    self._surface_electrode & surface_electrodey
                )
        return self._surface_electrode

    @property
    def srcList(self):
        """
        source list
        """
        if getattr(self, '_srcList', None) is None:
            # downhole source
            th_x = np.zeros(self.mesh.vnF[0], dtype=complex)
            th_y = np.zeros(self.mesh.vnF[1], dtype=complex)
            th_z = np.zeros(self.mesh.vnF[2], dtype=complex)

            th_z[self.tophole_electrode] = -1.  # part of wire coupled to casing
            th_x[self.surface_wire] = -1.  # horizontal part of wire along surface
            th_z[self.surface_electrode] = 1.  # vertical part of return electrode

            # assemble the source (downhole grounded primary)
            th = np.hstack([th_x, th_y, th_z])
            srcList = [
                FDEM.Src.RawVec_e([], _, th/self.mesh.area) for _ in self.freqs
            ]
            self._srcList = srcList
        return self._srcList

    def plot(self, ax=None):
        """
        plot the source on the mesh.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        mesh = self.mesh

        ax.plot(
            mesh.gridFz[self.tophole_electrode, 0],
            mesh.gridFz[self.tophole_electrode, 2], 'rv'
        )
        ax.plot(
            mesh.gridFz[self.surface_electrode, 0],
            mesh.gridFz[self.surface_electrode, 2], 'r^'
        )
        ax.plot(
            mesh.gridFx[self.surface_wire, 0],
            mesh.gridFx[self.surface_wire, 2], 'r<'
        )

        return ax

    def validate(self):
        """
        Make sure that each segment of the wire is only going through a
        single face
        """
        # check the surface electrode only has one x and one y location
        surface_electrode = self.mesh.gridFz[self.surface_electrode, :]
        assert len(np.unique(surface_electrode[:, 0])) == 1, (
            'the surface electrode has more than one x-location'
        )
        assert len(np.unique(surface_electrode[:, 1])) == 1, (
            'the surface electrode has more than one y-location'
        )

        # check the top casing electrode only has one x and one y location
        tophole_electrode = self.mesh.gridFz[self.tophole_electrode, :]
        assert len(np.unique(tophole_electrode[:, 0])) == 1, (
            'the tophole electrode has more than one x-location'
        )
        assert len(np.unique(tophole_electrode[:, 1])) == 1, (
            'the tophole electrode has more than one y-location'
        )

        # check the surface wire only has one y and one z location
        surface_wire = self.mesh.gridFx[self.surface_wire, :]
        assert len(np.unique(surface_wire[:, 1])) == 1, (
            'the surface wire has more than one y-location'
        )
        assert len(np.unique(surface_wire[:, 2])) == 1, (
            'the surface wire has more than one z-location'
        )

        return True
