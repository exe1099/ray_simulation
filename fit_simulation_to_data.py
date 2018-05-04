import lmfit as fit
from fast_simulation import run_fast_simulation_for_fitting
import simulation_classes as sim
import time
import numpy as np


# for logging
fit_number = str.lower(time.asctime()[4:-5]).replace(' ', '_').replace(':', '_')

# creating model
model = fit.Model(run_fast_simulation_for_fitting)

# parameters
params = fit.Parameters()
params.add('n1', value=1.5, min=1.4, max=1.6, brute_step=0.1)
params.add('scaling', value=1, min=0.9, max=1.1, brute_step=0.1, vary=False)

# importing data to fit to
data = np.genfromtxt('data/run30_minus_run34_data.csv')
data_to_fit = data[8:-2, [0,1]]  # leaving out total reflection and errors
# leaving out bump in the middle
data_to_fit = np.concatenate((data_to_fit[0:60, :], data_to_fit[81:, :]))
data_to_fit.T[1] /= np.sum(data_to_fit.T[1])  # normalizing with sum

# fitting
fit_result = model.fit(data_to_fit.T[1], params, xdata=data_to_fit.T[0], method='brute')

# getting numerical results
results = []
for candidate in fit_result.candidates:
    results.append([candidate.params['n1'].value, candidate.params['scaling'].value, candidate.score])
results = np.array(results)

# saving results
np.savetxt(f'fits/fit_results_{fit_number}.csv', results)

print('Fitting done!')
