import numpy as np
import matplotlib.pyplot as plt

from SimPEG.EM import FDEM


# Source Grounded on Casing
class DownHoleCasingSrc(object):

    def __init__(self, mesh, src_a, src_b, casing_a, freqs):
        self.mesh = mesh
        self.src_a = src_a
        self.src_b = src_b
        self.casing_a = casing_a
        self.freqs = freqs

    @property
    def dgv_ind(self):
        # vertically directed wire in borehole
        # go through the center of the well
        mesh = self.mesh
        src_a = self.src_a
        src_b = self.src_b

        dgv_indx = (mesh.gridFz[:, 0] < mesh.hx.min())
        dgv_indz = ((mesh.gridFz[:, 2] >= src_a[2] - 0.5*mesh.hz.min())
                    & (mesh.gridFz[:, 2] < src_b[2] + 1.5*mesh.hz.min()))
        dgv_ind = dgv_indx & dgv_indz
        return dgv_ind

    @property
    def dgh_ind2(self):
        mesh = self.mesh
        src_a = self.src_a
        src_b = self.src_b

        # couple to the casing downhole - top part
        dgh_indx = mesh.gridFx[:, 0] <= self.casing_a  # + mesh.hx.min()*2

        # couple to the casing downhole - bottom part
        dgh_indz2 = ((mesh.gridFx[:, 2] <= src_a[2]) &
                     (mesh.gridFx[:, 2] > src_a[2] - mesh.hz.min()))
        return dgh_indx & dgh_indz2

    @property
    def sgh_ind(self):
        mesh = self.mesh
        src_a = self.src_a
        src_b = self.src_b

        # horizontally directed wire
        sgh_indx = (mesh.gridFx[:, 0] <= src_b[0])
        sgh_indz = (
            (mesh.gridFx[:, 2] > mesh.hz.min()) &
            (mesh.gridFx[:, 2] < 2*mesh.hz.min())
        )
        return sgh_indx & sgh_indz

    @property
    def sgv_ind(self):
        mesh = self.mesh
        src_a = self.src_a
        src_b = self.src_b

        # return electrode
        sgv_indx = (
            (mesh.gridFz[:, 0] > src_b[0]*0.9) &
            (mesh.gridFz[:, 0] < src_b[0]*1.1)
        )
        sgv_indz = (
            (mesh.gridFz[:, 2] >= -mesh.hz.min()) &
            (mesh.gridFz[:, 2] < 1.*mesh.hz.min())
        )
        return sgv_indx & sgv_indz

    @property
    def s_e(self):
        if getattr(self, '_s_e', None) is None:
            # downhole source
            dg_x = np.zeros(self.mesh.vnF[0], dtype=complex)
            dg_y = np.zeros(self.mesh.vnF[1], dtype=complex)
            dg_z = np.zeros(self.mesh.vnF[2], dtype=complex)

            dg_z[self.dgv_ind] = -1.  # part of wire through borehole
            dg_x[self.dgh_ind2] = 1.  # downhole hz part of wire
            dg_x[self.sgh_ind] = -1.  # horizontal part of wire along surface
            dg_z[self.sgv_ind] = 1.  # vertical part of return electrode

            # assemble the source (downhole grounded primary)
            dg = np.hstack([dg_x, dg_y, dg_z])
            s_e = [
                FDEM.Src.RawVec_e([], _, dg/self.mesh.area) for _ in self.freqs
            ]
            self._s_e = s_e
        return self._s_e

    def plotSrc(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        mesh = self.mesh

        ax.plot(
            mesh.gridFz[self.dgv_ind, 0], mesh.gridFz[self.dgv_ind, 2], 'rv'
        )
        ax.plot(
            mesh.gridFx[self.dgh_ind2, 0], mesh.gridFx[self.dgh_ind2, 2], 'r>'
        )
        ax.plot(
            mesh.gridFz[self.sgv_ind, 0], mesh.gridFz[self.sgv_ind, 2], 'r^'
        )
        ax.plot(
            mesh.gridFx[self.sgh_ind, 0], mesh.gridFx[self.sgh_ind, 2], 'r<'
        )

        return ax


class TopCasingSource(object):

    def __init__(self, mesh, src_a, src_b, casing_a, freqs):
        self.mesh = mesh
        self.src_a = src_a
        self.src_b = src_b
        self.casing_a = casing_a
        self.freqs = freqs

    @property
    def th_ind(self):
        mesh = self.mesh
        src_a = self.src_a
        src_b = self.src_b

        th_indx = (
            (mesh.gridFz[:, 0] <= self.casing_a + mesh.hx.min()/2.) &
            (mesh.gridFz[:, 0] > self.casing_a - mesh.hx.min()/2.)
        )

        th_indz = (
            (mesh.gridFz[:, 2] < src_b[2] + 1.5*mesh.hz.min()) &
            (mesh.gridFz[:, 2] >= src_a[2] - 0.5*mesh.hz.min())
        )

        return th_indx & th_indz

    @property
    def sgh_ind(self):
        mesh = self.mesh
        src_a = self.src_a
        src_b = self.src_b

        # horizontally directed wire
        sgh_indx = (
            (mesh.gridFx[:, 0] <= src_b[0]) &
            (mesh.gridFx[:, 0] > self.casing_a)
        )
        sgh_indz = (
            (mesh.gridFx[:, 2] > mesh.hz.min()) &
            (mesh.gridFx[:, 2] < 2*mesh.hz.min())
        )
        return sgh_indx & sgh_indz

    @property
    def sgv_ind(self):
        mesh = self.mesh
        src_a = self.src_a
        src_b = self.src_b

        # return electrode
        sgv_indx = (
            (mesh.gridFz[:, 0] > src_b[0]*0.9) &
            (mesh.gridFz[:, 0] < src_b[0]*1.1)
        )
        sgv_indz = (
            (mesh.gridFz[:, 2] >= -mesh.hz.min()) &
            (mesh.gridFz[:, 2] < 1.5*mesh.hz.min())
        )
        return sgv_indx & sgv_indz

    @property
    def s_e(self):
        if getattr(self, '_s_e', None) is None:
            # downhole source
            th_x = np.zeros(self.mesh.vnF[0], dtype=complex)
            th_y = np.zeros(self.mesh.vnF[1], dtype=complex)
            th_z = np.zeros(self.mesh.vnF[2], dtype=complex)

            th_z[self.th_ind] = -1.  # part of wire coupled to casing
            th_x[self.sgh_ind] = -1.  # horizontal part of wire along surface
            th_z[self.sgv_ind] = 1.  # vertical part of return electrode

            # assemble the source (downhole grounded primary)
            th = np.hstack([th_x, th_y, th_z])
            s_e = [
                FDEM.Src.RawVec_e([], _, th/self.mesh.area) for _ in self.freqs
            ]
            self._s_e = s_e
        return self._s_e

    def plotSrc(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        mesh = self.mesh

        ax.plot(
            mesh.gridFz[self.th_ind, 0], mesh.gridFz[self.th_ind, 2], 'rv'
        )
        ax.plot(
            mesh.gridFz[self.sgv_ind, 0], mesh.gridFz[self.sgv_ind, 2], 'r^'
        )
        ax.plot(
            mesh.gridFx[self.sgh_ind, 0], mesh.gridFx[self.sgh_ind, 2], 'r<'
        )

        return ax
