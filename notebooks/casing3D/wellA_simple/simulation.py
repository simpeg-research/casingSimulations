import casingSimulations

# Set up the simulations

# Cyl Simulation
simCyl = casingSimulations.run.SimulationFDEM(
    cp='CasingParameters.json',
    meshGenerator='MeshParameters.json',
    srcType='TopCasingSrc',
    fields_filename='fieldsCyl.npy',
    num_threads=12
)

# run the simulations
fieldsCyl = simCyl.run()
