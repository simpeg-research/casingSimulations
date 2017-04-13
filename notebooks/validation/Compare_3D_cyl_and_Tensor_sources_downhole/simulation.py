import casingSimulations

# Set up the simulations

# Cyl Simulation
simCyl = casingSimulations.run.SimulationFDEM(
    cp='cyl_cp.json',
    meshGenerator='cyl_mesh.json',
    src='DownHoleTerminatingSrc',
    fields_filename='fieldsCyl.npy'
)

# Tensor Simulation
simTensor = casingSimulations.run.SimulationFDEM(
    cp='tensor_cp.json',
    meshGenerator='tensor_mesh.json',
    src='DownHoleTerminatingSrc',
    fields_filename='fieldsTensor.npy'
)
# run the simulations
fieldsCyl = simCyl.run()
fieldsTensor = simTensor.run()
