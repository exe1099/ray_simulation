import lmfit as fit
from fast_simulation import run_fast_simulation_for_fitting
import simulation as sim

# for logging
fit_number = str.lower(time.asctime()[4:-5]).replace(' ', '_').replace(':', '_')

# creating model
model = fit.Model(run_fast_simulation_for_fitting)

# parameters
params = fit.Parameters()
params.add('n1', value=1.5, min=1.4, max=1.6, brute_step=0.1)

# importing data to fit to
data = np.genfromtxt('data/run30_minus_run34_data.csv')
data_to_fit = data[8:-2, [0,1]]  # no total reflection, no error calculation
data_to_fit = np.concatenate((data_to_fit[0:60, :], data_to_fit[81:, :]))  # no middle bump
data_to_fit.T[1] /= np.sum(data_to_fit.T[1])  # normalizing

# fitting
fit_result = model.fit(data_to_fit.T[1], params, xdata=data_to_fit.T[0], method='brute')

# saving results
file_path = 'fitting/fit_results.txt'
sim.log("", file_path)
sim.log("", file_path)
sim.log(f"Fitting {fit_number}", file_path)
sim.log(f"{fit_result.candidates}")
