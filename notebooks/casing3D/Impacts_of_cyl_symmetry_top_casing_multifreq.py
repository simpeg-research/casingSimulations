
# coding: utf-8

# In[50]:

import time
import discretize
from discretize import utils
import numpy as np
import scipy.sparse as sp

import sympy
from scipy.constants import mu_0

from SimPEG.EM import FDEM
from SimPEG import Utils, Maps

import CasingSimulations

from pymatsolver import Pardiso

import matplotlib.pyplot as plt


# In[57]:

sigma_back = 1e-1 # wholespace

# top casing source
cp = CasingSimulations.CasingParameters(
    casing_l = 1000.,
    src_a = np.r_[0., np.pi, 0.], # the source fcts will take care of coupling it to the casing
    src_b = np.r_[1e3, np.pi, 0.], # return electrode
    freqs = np.r_[0.25, 0.5, 1., 2.],
    sigma_back = sigma_back, # wholespace
    sigma_layer = sigma_back,
    sigma_air = sigma_back,

)
print('Casing Parameters: ', cp.serialize())


# In[58]:

print('skin depths in casing: ', cp.skin_depth(sigma=cp.sigma_casing, mu=cp.mur_casing*mu_0))
print('casing thickness: ',  cp.casing_t)


# In[59]:

print('skin depths in background: ', cp.skin_depth())


# # Set up meshes

# In[71]:

npadx, npadz = 9, 20
dx2 = 200.
csz = 0.25

mesh2D = CasingSimulations.CasingMesh(
    cp=cp, npadx=npadx, npadz=npadz, dx2=dx2, csz=csz
).mesh

print(mesh2D.vectorNx.max(), mesh2D.vectorNz.min(), mesh2D.vectorNz.max())


# In[72]:

ncy = 1
nstretchy = 3
stretchfact = 1.6
hy = utils.meshTensor([(1, nstretchy, -stretchfact), (1, ncy), (1, nstretchy, stretchfact)])
hy = hy * 2*np.pi/hy.sum()


# In[73]:

mesh3D = CasingSimulations.CasingMesh(
    cp=cp, npadx=npadx, npadz=npadz, dx2=dx2, hy=hy, csz=csz
).mesh


# In[74]:

print(mesh2D.nC, mesh3D.nC)


# In[24]:

mesh2D.vectorCCx.max()


# In[25]:

# TODO: this should go into the cyl mesh view
self = mesh3D
NN = utils.ndgrid(self.vectorNx, self.vectorNy, np.r_[0])[:,:2]
NN = NN.reshape((self.vnN[0], self.vnN[1], 2), order='F')
NN = [NN[:,:,0], NN[:,:,1]]


# In[26]:

print(NN[1].shape, self.nCx)


# In[27]:

from discretize.utils import mkvc


# In[28]:

X1 = np.c_[mkvc(NN[0][0, :]), mkvc(NN[0][self.nCx, :]), mkvc(NN[0][0, :])*np.nan].flatten()
Y1 = np.c_[mkvc(NN[1][0, :]), mkvc(NN[1][self.nCx, :]), mkvc(NN[1][0, :])*np.nan].flatten()


# In[36]:

ax = plt.subplot(111, projection='polar')
ax.plot(Y1, X1, 'b-')
n = 100
xy2 = [ax.plot(np.linspace(0., np.pi*2, n), r*np.ones(n), '-b') for r in self.vectorNx]
ax.set_rlim([0., 1000.])


# In[37]:

src2D = CasingSimulations.Sources.TopCasingSource(mesh2D, cp)


# In[38]:

src3D = CasingSimulations.Sources.TopCasingSource(mesh3D, cp)


# In[39]:

fig, ax = plt.subplots(1,1)
mesh2D.plotGrid(ax=ax)
src2D.plot(ax=ax)
ax.set_xlim([-10,1010.])
ax.set_ylim(0.5*np.r_[-1., 1.])


# In[40]:

fig, ax = plt.subplots(1,1)
mesh2D.plotGrid(ax=ax)
src3D.plot(ax=ax)
ax.set_xlim([-10,1010.])
ax.set_ylim(0.5*np.r_[-1., 1.])


# In[41]:

# validate the source terms
src3D.validate()
src2D.validate()


# In[42]:

ax = plt.subplot(111, projection='polar')
ax.plot(Y1, X1, 'b-')
n = 100
xy2 = [ax.plot(np.linspace(0., np.pi*2, n), r*np.ones(n), '-b') for r in self.vectorNx[1:]]
ax.plot(mesh3D.gridFx[src3D.surface_wire,1], mesh3D.gridFx[src3D.surface_wire,0], 'ro')
ax.set_rlim([0., 1500])


# # Look at physical properties on mesh

# In[43]:

physprops2D = CasingSimulations.PhysicalProperties(mesh2D, cp)
physprops3D = CasingSimulations.PhysicalProperties(mesh3D, cp)


# In[44]:

xlim = [-1., 1]
ylim = [-1200., 100.]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
plt.colorbar(
    mesh2D.plotImage(np.log10(physprops2D.sigma), ax=ax[0], mirror=True)[0], ax=ax[0]
)
ax[0].set_xlim(xlim)
ax[0].set_ylim(ylim)

sigmaplt = physprops3D.sigma.reshape(mesh3D.vnC, order='F')

plt.colorbar(mesh2D.plotImage(np.log10(utils.mkvc(sigmaplt[:,0,:])), ax=ax[1], mirror=True)[0], ax=ax[1])
ax[1].set_xlim(xlim)
ax[1].set_ylim(ylim)

plt.tight_layout()


# In[45]:

xlim = [-1., 1]
ylim = [-1200., 100.]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
plt.colorbar(
    mesh2D.plotImage(physprops2D.mur, ax=ax[0], mirror=True)[0], ax=ax[0]
)
ax[0].set_xlim(xlim)
ax[0].set_ylim(ylim)

murplt = physprops3D.mur.reshape(mesh3D.vnC, order='F')

plt.colorbar(mesh2D.plotImage(utils.mkvc(murplt[:,0,:]), ax=ax[1], mirror=True)[0], ax=ax[1])
ax[1].set_xlim(xlim)
ax[1].set_ylim(ylim)

plt.tight_layout()


# # set up the forward simulation

# In[46]:

prb2D = FDEM.Problem3D_h(
    mesh2D, sigmaMap=physprops2D.wires.sigma, muMap=physprops2D.wires.mu, Solver=Pardiso
)
prb3D = FDEM.Problem3D_h(
    mesh3D, sigmaMap=physprops3D.wires.sigma, muMap=physprops3D.wires.mu, Solver=Pardiso
)


# In[47]:

survey2D = FDEM.Survey(src2D.srcList)
survey3D = FDEM.Survey(src3D.srcList)


# In[48]:

prb2D.pair(survey2D)
prb3D.pair(survey3D)


# In[51]:

t = time.time()
fields2D = prb2D.fields(physprops2D.model)
np.save('fields2DMultiFreqtopCasing', fields2D[:, 'hSolution'])
print('Elapsed time for 2D: {}'.format(time.time()-t))


# In[28]:

# %%time
t = time.time()
fields3D = prb3D.fields(physprops3D.model)
np.save('fields2DMultiFreqtopCasing', fields3D[:, 'hSolution'])
print('Elapsed time for 3D: {}'.format(time.time()-t))


# In[29]:

h2D = np.load('fields2DMultiFreqtopCasing.npy')
h3D = np.load('fields3DMultiFreqtopCasing.npy')


# In[30]:

prb2D.model = physprops2D.model
fields2D = prb2D.fieldsPair(mesh2D, survey2D)
fields2D[:, 'hSolution'] = h2D


# In[31]:

prb3D.model = physprops3D.model
fields3D = prb3D.fieldsPair(mesh3D, survey3D)
fields3D[:, 'hSolution'] = h3D

