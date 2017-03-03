
# coding: utf-8

# In[1]:

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
from discretize.utils import mkvc

plotIt = False

if plotIt:
    import matplotlib.pyplot as plt
# get_ipython().magic(u'matplotlib inline')


# In[2]:

sigma_back = 1e-1 # wholespace

# top casing source
cp = CasingSimulations.CasingParameters(
    casing_l = 1000.,
    src_a = np.r_[0., np.pi, 0.], # the source fcts will take care of coupling it to the casing
    src_b = np.r_[1e3, np.pi, 0.], # return electrode
    freqs = np.r_[0.5],
    sigma_back = sigma_back, # wholespace
    sigma_layer = sigma_back,
    sigma_air = sigma_back,

)
cp.serialize()


# In[3]:

cp.skin_depth()


# # Set up meshes

# In[4]:

npadx, npadz = 8, 19
dx2 = 200.
csz = 0.25

mesh2D = CasingSimulations.CasingMesh(
    cp=cp, npadx=npadx, npadz=npadz, dx2=dx2, csz=csz
).mesh


# In[5]:

print(mesh2D.vectorNz.min(), mesh2D.vectorNz.max(), mesh2D.vectorNx.max())


# In[6]:

ncy = 1
nstretchy = 3
stretchfact = 1.6
hy = utils.meshTensor([(1, nstretchy, -stretchfact), (1, ncy), (1, nstretchy, stretchfact)])
hy = hy * 2*np.pi/hy.sum()


# In[7]:

mesh3D = CasingSimulations.CasingMesh(
    cp=cp, npadx=npadx, npadz=npadz, dx2=dx2, hy=hy, csz=csz
).mesh


# In[8]:

print(mesh2D.nC, mesh3D.nC)


# In[9]:

mesh2D.vectorCCx.max()


# In[10]:

# TODO: this should go into the cyl mesh view
self = mesh3D
NN = utils.ndgrid(self.vectorNx, self.vectorNy, np.r_[0])[:,:2]
NN = NN.reshape((self.vnN[0], self.vnN[1], 2), order='F')
NN = [NN[:,:,0], NN[:,:,1]]


# In[11]:

print(NN[1].shape, self.nCx)


# In[12]:




# In[13]:

X1 = np.c_[mkvc(NN[0][0, :]), mkvc(NN[0][self.nCx, :]), mkvc(NN[0][0, :])*np.nan].flatten()
Y1 = np.c_[mkvc(NN[1][0, :]), mkvc(NN[1][self.nCx, :]), mkvc(NN[1][0, :])*np.nan].flatten()


# In[14]:

if plotIt:
    ax = plt.subplot(111, projection='polar')
    ax.plot(Y1, X1, 'b-')
    n = 100
    xy2 = [ax.plot(np.linspace(0., np.pi*2, n), r*np.ones(n), '-b') for r in self.vectorNx]
    ax.set_rlim([0., 1000.])


# In[15]:

src2D = CasingSimulations.Sources.TopCasingSource(mesh2D, cp)


# In[16]:

src3D = CasingSimulations.Sources.TopCasingSource(mesh3D, cp)


# In[17]:

if plotIt:
    fig, ax = plt.subplots(1,1)
    mesh2D.plotGrid(ax=ax)
    src2D.plot(ax=ax)
    ax.set_xlim([-10,1010.])
    ax.set_ylim(0.5*np.r_[-1., 1.])


# In[18]:

if plotIt:
    fig, ax = plt.subplots(1,1)
    mesh2D.plotGrid(ax=ax)
    src3D.plot(ax=ax)
    ax.set_xlim([-10,1010.])
    ax.set_ylim(0.5*np.r_[-1., 1.])


# In[19]:

# validate the source terms
src3D.validate()
src2D.validate()


# In[20]:

if plotIt:
    ax = plt.subplot(111, projection='polar')
    ax.plot(Y1, X1, 'b-')
    n = 100
    xy2 = [ax.plot(np.linspace(0., np.pi*2, n), r*np.ones(n), '-b') for r in self.vectorNx[1:]]
    ax.plot(mesh3D.gridFx[src3D.surface_wire,1], mesh3D.gridFx[src3D.surface_wire,0], 'ro')
    ax.set_rlim([0., 1500])


# # Look at physical properties on mesh

# In[21]:

physprops2D = CasingSimulations.PhysicalProperties(mesh2D, cp)
physprops3D = CasingSimulations.PhysicalProperties(mesh3D, cp)


# In[22]:

if plotIt:
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


    # In[23]:

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

# In[24]:

prb2D = FDEM.Problem3D_h(
    mesh2D, sigmaMap=physprops2D.wires.sigma, muMap=physprops2D.wires.mu, Solver=Pardiso
)
prb3D = FDEM.Problem3D_h(
    mesh3D, sigmaMap=physprops3D.wires.sigma, muMap=physprops3D.wires.mu, Solver=Pardiso
)


# In[25]:

survey2D = FDEM.Survey(src2D.srcList)
survey3D = FDEM.Survey(src3D.srcList)


# In[26]:

prb2D.pair(survey2D)
prb3D.pair(survey3D)


# In[27]:

fields2D = prb2D.fields(physprops2D.model)



# In[30]:

np.save('fields2DtopCasing', fields2D[:, 'hSolution'])


# In[ ]:

fields3D = prb3D.fields(physprops3D.model)


# # In[ ]:

np.save('fields3DtopCasing', fields3D[:, 'hSolution'])

