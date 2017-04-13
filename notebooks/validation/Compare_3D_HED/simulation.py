import casingSimulations

# Set up the simulations

# Cyl Simulation
simCyl = casingSimulations.run.SimulationFDEM(
    cp='cyl_cp.json',
    meshGenerator='cyl_mesh.json',
    src='HorizontalElectricDipole',
    fields_filename='fieldsCyl.npy'
)

# run the simulations
fieldsCyl = simCyl.run()
