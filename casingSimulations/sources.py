import numpy as np
import matplotlib.pyplot as plt

import properties
import discretize
from discretize.utils import closestPoints

from SimPEG import Utils
from SimPEG.EM import FDEM


# TODO: This should be a serealizable class instead.
class BaseCasingSrc(object):
    """
    The base class for sources. Inherit this to attach properties.
    """
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


class HorizontalElectricDipole(BaseCasingSrc):
    """
    A source that terminates down-hole. It is not coupled to the casing

    :param CasingSimulations.Model.CasingProperties cp: a casing properties instance
    :param discretize.BaseMesh mesh: a discretize mesh
    """

    def __init__(self, cp, mesh):
        super(HorizontalElectricDipole, self).__init__(cp, mesh)
        assert self.src_a[2] == self.src_b[2], (
            'z locations must be the same for a HED'
        )

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
            surface_wirex = (
                (
                    mesh.gridFx[:, 0] <= np.max(
                        [self.src_a[0], self.src_b[0]]
                    )
                ) &
                (
                    mesh.gridFx[:, 0] >= np.min(
                        [self.src_a[0], self.src_b[0]]
                    )
                )
            )
            surface_wirez = (
                (mesh.gridFx[:, 2] > src_b[2] - self.mesh.hz.min()/2.) &
                (mesh.gridFx[:, 2] < src_b[2] + self.mesh.hz.min()/2.)
            )
            self._surface_wire = surface_wirex & surface_wirez

            if getattr(mesh, 'isSymmetric', False) is False:
                surface_wirey = (
                    (mesh.gridFx[:, 1] > src_b[1] - mesh.hy.min()/2.) &
                    (mesh.gridFx[:, 1] < src_b[1] + mesh.hy.min()/2.)
                )

                self._surface_wire = (
                    self._surface_wire & surface_wirey
                )

        return self._surface_wire

    @property
    def surface_wire_direction(self):
        # todo: extend to the case where the wire is not along the x-axis
        return [-1. if self.src_a[0] < self.src_b[0] else 1.][0]

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

            dg_x[self.surface_wire] = self.surface_wire_direction  # horizontal part of wire along surface

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
            mesh.gridFx[self.surface_wire, 0],
            mesh.gridFx[self.surface_wire, 2], 'r{}'.format(
                ['<' if self.surface_wire_direction == -1. else '>'][0]
            )
        )

    def validate(self):
        """
        Make sure that each segment of the wire is only going through a
        single face

        .. todo:: check that
        """
        # check the surface wire only has one y and one z location
        surface_wire = self.mesh.gridFx[self.surface_wire, :]
        assert len(np.unique(surface_wire[:, 1])) == 1, (
            'the surface wire has more than one y-location'
        )
        assert len(np.unique(surface_wire[:, 2])) == 1, (
            'the surface wire has more than one z-location'
        )


class DownHoleTerminatingSrc(BaseCasingSrc):
    """
    A source that terminates down-hole. It is not coupled to the casing

    :param CasingSimulations.Model.CasingProperties cp: a casing properties instance
    :param discretize.BaseMesh mesh: a discretize mesh
    """

    def __init__(self, cp, mesh):
        super(DownHoleTerminatingSrc, self).__init__(cp, mesh)

    @property
    def src_a_closest(self):
        """
        closest face to where we want the return current electrode
        """
        if getattr(self, '_src_a_closest', None) is None:
            # find the z location of the closest face to the src
            src_a_closest = (
                self.mesh.gridFz[closestPoints(self.mesh, self.src_a, 'Fz'), :]
            )
            assert(len(src_a_closest) == 1), 'multiple source locs found'
            self._src_a_closest = src_a_closest[0]
        return self._src_a_closest

    @property
    def src_b_closest(self):
        """
        closest face to where we want the return current electrode
        """
        if getattr(self, '_src_b_closest', None) is None:
            # find the z location of the closest face to the src
            src_b_closest = (
                self.mesh.gridFz[closestPoints(self.mesh, self.src_b, 'Fz'), :]
            )
            assert(len(src_b_closest) == 1), 'multiple source locs found'
            self._src_b_closest = src_b_closest[0]
        return self._src_b_closest

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

            wire_in_boreholex = (
                (mesh.gridFz[:, 0] < self.src_a_closest[0] + mesh.hx.min()/2.) &
                (mesh.gridFz[:, 0] > self.src_a_closest[0] - mesh.hx.min()/2.)
            )
            wire_in_boreholez = (
                (mesh.gridFz[:, 2] >= src_a[2] - 0.5*mesh.hz.min()) &
                (mesh.gridFz[:, 2] < src_b[2] + 1.5*mesh.hz.min())
            )

            self._wire_in_borehole = wire_in_boreholex & wire_in_boreholez

            if getattr(mesh, 'isSymmetric', False) is False:
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
            surface_wirex = (
                (
                    mesh.gridFx[:, 0] <= np.max(
                        [self.src_a_closest[0], self.src_b_closest[0]]
                    )
                ) &
                (
                    mesh.gridFx[:, 0] >= np.min(
                        [self.src_a_closest[0], self.src_b_closest[0]]
                    )
                )
            )
            surface_wirez = (
                (mesh.gridFx[:, 2] > mesh.hz.min()) &
                (mesh.gridFx[:, 2] <= 1.75*mesh.hz.min())
            )
            self._surface_wire = surface_wirex & surface_wirez

            if getattr(mesh, 'isSymmetric', False) is False:
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
                (mesh.gridFz[:, 0] > self.src_b_closest[0] - mesh.hx.min()/2.) &
                (mesh.gridFz[:, 0] < self.src_b_closest[0] + mesh.hx.min()/2.)
            )
            surface_electrodez = (
                (mesh.gridFz[:, 2] >= src_b[2] - mesh.hz.min()) &
                (mesh.gridFz[:, 2] < src_b[2] + 1.75*mesh.hz.min())
            )
            self._surface_electrode = surface_electrodex & surface_electrodez

            if getattr(mesh, 'isSymmetric', False) is False:
                surface_electrodey = (
                    (mesh.gridFz[:, 1] > src_b[1] - mesh.hy.min()/2.) &
                    (mesh.gridFz[:, 1] < src_b[1] + mesh.hy.min()/2.)
                )
                self._surface_electrode = (
                    self._surface_electrode & surface_electrodey
                )
        return self._surface_electrode

    @property
    def surface_wire_direction(self):
        # todo: extend to the case where the wire is not along the x-axis
        return [-1. if self.src_a[0] < self.src_b[0] else 1.][0]

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

            dg_z[self.wire_in_borehole] = -1.   # part of wire through borehole
            dg_x[self.surface_wire] = self.surface_wire_direction  # horizontal part of wire along surface
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
            mesh.gridFx[self.surface_wire, 2], 'r{}'.format(
                ['<' if self.surface_wire_direction == -1. else '>'][0]
            )
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

            if getattr(mesh, 'isSymmetric', False) is False:
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


class TopCasingSrc(DownHoleTerminatingSrc):
    """
    Source that has one electrode coupled to the top of the casing, one return
    electrode and a wire in between. This source is set up to live on faces.

    :param discretize.CylMesh mesh: the cylindrical simulation mesh
    :param CasingSimulations cp: Casing parameters object
    """
    def __init__(self, cp, mesh):
        super(TopCasingSrc, self).__init__(cp, mesh)
        # self.mesh = mesh
        # self.src_a = cp.src_a
        # self.src_b = cp.src_b
        # self.casing_a = cp.casing_a
        # self.freqs = cp.freqs

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
                (mesh.gridFz[:, 0] <= self.casing_a + mesh.hx.min()) &
                (mesh.gridFz[:, 0] > self.casing_a)
            )

            tophole_electrodez = (
                (mesh.gridFz[:, 2] < src_a[2] + 1.5*mesh.hz.min()) &
                (mesh.gridFz[:, 2] >= src_a[2] - 0.5*mesh.hz.min())
            )

            self._tophole_electrode = tophole_electrodex & tophole_electrodez

            if getattr(mesh, 'isSymmetric', False) is False:
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
                (mesh.gridFx[:, 0] <= self.src_b_closest[0]) &
                (mesh.gridFx[:, 0] > self.casing_a + mesh.hx.min()/2.)
            )
            surface_wirez = (
                (mesh.gridFx[:, 2] > src_b[2] + mesh.hz.min()) &
                (mesh.gridFx[:, 2] <= src_b[2] + 1.75*mesh.hz.min())
            )
            self._surface_wire = surface_wirex & surface_wirez

            if getattr(mesh, 'isSymmetric', False) is False:
                surface_wirey = (
                    (mesh.gridFx[:, 1] < src_b[1] + mesh.hy.min()/2.) &
                    (mesh.gridFx[:, 1] > src_b[1] - mesh.hy.min()/2.)
                )
                self._surface_wire = self._surface_wire & surface_wirey
        return self._surface_wire

    # @property
    # def surface_electrode(self):
    #     """
    #     indices of the return electrode at the surface
    #     """
    #     if getattr(self, '_surface_electrode', None) is None:
    #         mesh = self.mesh
    #         src_a = self.src_a
    #         src_b = self.src_b

    #         # return electrode
    #         closeFace = mesh.gridFz[closestPoints(mesh, src_b, 'Fz')]

    #         surface_electrodex = (
    #             (
    #                 mesh.gridFz[:, 0] >
    #                 self.src_b_closest[0] - mesh.hx.min()/2.
    #             ) & #src_b[0]*0.9) &
    #             (
    #                 mesh.gridFz[:, 0] <
    #                 self.src_b_closest[0] + mesh.hx.min()/2.
    #             ) #src_b[0]*1.1)
    #         )
    #         surface_electrodez = (
    #             (mesh.gridFz[:, 2] > -0.5*mesh.hz.min()) &
    #             (mesh.gridFz[:, 2] < 1.5*mesh.hz.min())
    #         )
    #         self._surface_electrode = surface_electrodex & surface_electrodez

    #         isSymmetric = getattr(mesh, 'isSymmetric', False)
    #         if isSymmetric is False or isSymmetric is None:
    #             surface_electrodey = (
    #                 (mesh.gridFz[:, 1] < src_b[1] + mesh.hy.min()) &
    #                 (mesh.gridFz[:, 1] > src_b[1] - mesh.hy.min())
    #             )
    #             self._surface_electrode = (
    #                 self._surface_electrode & surface_electrodey
    #             )
    #     return self._surface_electrode

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
