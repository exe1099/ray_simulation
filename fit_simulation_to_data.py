import lmfit as fit
from fast_simulation import run_fast_simulation_for_fitting
import simulation_classes as sim
import time
import numpy as np


# for logging
fit_number = str.lower(time.asctime()[4:-5]).replace(' ', '_').replace(':', '_')

# creating residual function
def residual(params, xdata, data_to_fit):
    # extracting parameter values
    n1 = params['n1'].value
    scaling = params['scaling'].value
    repeat_sim = params['repeat_sim'].value
    n_rays = params['n_rays'].value
    print(params)

    # run simulation, returns [data, d_data]
    model_data = run_fast_simulation_for_fitting(xdata, n1, scaling, repeat_sim, n_rays)

    # calculating weighted residuals
    resi = model_data[0] - data_to_fit[0]
    d_resi = np.sqrt(model_data[1]**2 + data_to_fit[1]**2)
    return resi / d_resi


# parameters
params = fit.Parameters()
params.add('n1', value=1.5, min=1.48, max=1.5, brute_step=0.02)
params.add('scaling', value=1, min=0.9, max=1.1, brute_step=0.1, vary=False)
params.add('repeat_sim', value=20, vary=False)
params.add('n_rays', value=8*10**7, vary=False)

# importing data to fit to
data = np.genfromtxt('data/run138_minus_run137_data.csv')
data_to_fit = data[17:-11, :]  # leaving out total reflection
# leaving out bump in the middle
data_to_fit = np.concatenate((data_to_fit[0:56, :], data_to_fit[116:, :]))  # no middle bump
data_to_fit.T[1:3] /= np.sum(data_to_fit.T[1])  # normalizing with sum

# fitting
minimizer_result = fit.minimize(residual, params, method='brute',
                                kws={'xdata': data_to_fit.T[0],
                                     'data_to_fit': data_to_fit.T[1:3, :]})
print(fit.fit_report(minimizer_result))

# getting numerical results
results = []
for candidate in minimizer_result.candidates:
    results.append([candidate.params['n1'].value, candidate.score])
results = np.array(results)

# saving results
np.savetxt(f'fits/fit_results_{fit_number}.csv', results)

with open(f'fits/fit_results_{fit_number}.csv', 'a') as file:
    file.write(f"\n {params['repeat_sim']} \n {params['n_rays']} \n")

print('Fitting done!')
