import numpy as np
import matplotlib
#  matplotlib.use('agg')
import matplotlib.pyplot as plt
import simulation as sim  # self-written classes, functions (detector etc.)
import time
from scipy.interpolate import interp1d

RAD_TO_DEG = 180 / np.pi
DEG_TO_RAD = np.pi / 180

def run_fast_simulation(xdata=np.array([0,1]), n1=1.5, n2=1, scaling=1, detector_distance=7.5,
                        detector_window=0.3, detector_steps=100, n_rays=1*10**7, logging=False):
    """Run fast simulation.

    - xdata [numpy.ndarray]: x-values at which to return simulated y-values
    - n1/n2: refractive index of first/second medium
    - detector_distance/detector_window [cm]: diameter
    - detector_steps: positions detector will scan
    - n_rays: number of rays to simulate
    - logging: whether to write to log file and create pdf or not
    """

    detector_window=3.4  # smoothing by increasing detector_window
    print(f"n1: {n1}")
    print(f"scaling: {scaling}")
    # logging
    if logging:
        start_time = time.time()
        sim_number = str.lower(time.asctime()[4:-5]).replace(' ', '_').replace(':', '_')
        sim.log("")
        sim.log("")
        sim.log(f"Simulation {sim_number}")
        sim.log(f"n1: {n1}")
        sim.log(f"n2: {n2}")
        sim.log(f"detector_distance: {detector_distance}")
        sim.log(f"detector_window: {detector_window}")
        sim.log(f"detector_steps: {detector_steps}")
        sim.log(f"n_rays: {n_rays: e}")

    # creating instances
    detector = sim.Detector(detector_distance, detector_window)
    detector.steps = detector_steps
    ray = sim.Ray([0, 0, 0], [1, 1, 1])
    # generating initial raya
    print("Generating initial rays and throwing unimportant rays away...")
    # maximal y-value ray can have and still hit detector
    # noinspection PyProtectedMember
    max_y = detector._window_diameter / 2 / detector._distance
    rays = np.random.randn(3, n_rays)  # generating rays
    rays[2] = np.abs(rays[2])  # mirror z < 0 rays to z > 0
    rays /= np.linalg.norm(rays, axis=0)
    # only pick rays with y < max_y
    rays = rays[:, np.abs(rays[1]) < max_y]
    print("Generating initial rays and throwing unimportant rays away... done")

    # refract rays
    print("Refract rays...")
    N = np.array([0, 0, 1])  # surface normal
    theta1 = np.arccos(np.dot(rays.T, N))  # angle of incident, in rad
    # case: ray hits surface
    hits_surface_bool = theta1 < 90 * DEG_TO_RAD
    # case: total reflection
    sin_theta2 = n1 / n2 * np.sin(theta1)
    total_reflection_bool = sin_theta2 > 1
    # choose rays which are worth refracting
    good_rays_bool = np.logical_and(hits_surface_bool, np.logical_not(total_reflection_bool))
    good_rays = rays.T[good_rays_bool]
    theta1 = theta1[good_rays_bool]
    theta2 = np.arcsin(sin_theta2[good_rays_bool])
    # refracting
    A = n1 / n2 * (good_rays.T - (N[:, None] * np.cos(theta1)))
    B = N[:, None] * np.cos(theta2)
    T = A + B
    refracted_rays = T
    # Fresnel equations
    F_R_parallel = ((n2 * np.cos(theta1) - n1 * np.cos(theta2)) / (n2 * np.cos(
                theta1) + n1 * np.cos(theta2))) ** 2
    F_R_perp = ((n1 * np.cos(theta2) - n2 * np.cos(theta1)) / (n1 * np.cos(
                theta2) + n2 * np.cos(theta1))) ** 2
    F_R = 1 / 2 * (F_R_parallel + F_R_perp)
    F_T = 1 - F_R
    # create 4 dimensional vectors with last dimension intensity
    # refracted_rays = np.array([T, F_T])
    print("Refract rays... done")

    # check if rays hit detector
    print("Check if rays hit detector...")
    bins_intensity = []
    angular_positions = []
    # iterating detector positions
    for angular_position, detector_direction in detector:
        angular_positions.append(angular_position)
        angle = np.arccos(np.dot(refracted_rays.T, detector_direction)) * RAD_TO_DEG
        # check if detector got hit by comparing angle between ray and detector direction
        # with opening angle of detector
        hits_detector_bool = angle <= detector.d_angle
        bins_intensity.append(np.sum(F_T[hits_detector_bool]))
    print("Check if rays hit detector... done")

    # plotting and logging
    if logging:
        fig = plt.figure(figsize=(15, 10))
        plt.grid()
        plt.plot(angular_positions, bins_intensity, label="Intensity")
        plt.legend()
        plt.xlabel(r"$\mathtt{Angle\/\/[°]}$", fontsize=15)
        plt.ylabel(r"$\mathtt{Intensity/Counts\/\/}$", fontsize=15)
        plt.savefig(f'runs/{sim_number}.pdf')

        np.savetxt(f"runs/{sim_number}.csv", np.array([angular_positions, bins_intensity]).T, '%.5f',
                   header="angular_position  intensity")
        sim.log("Successful run!")
        sim.log(f"Ran for: {(time.time() - start_time) / 60: .1f}min")

    print("Successful run!")

    # normalizing and return data at xdata points
    interpolation = interp1d(angular_positions, bins_intensity, kind='cubic')  # A / W
    return_data = interpolation(xdata)
    return_data /= np.sum(return_data)
    return return_data * scaling


def run_fast_simulation_for_fitting(xdata, n1,):
    return run_fast_simulation(xdata=xdata, n1=n1, n_rays=3*10**7)


if __name__ == "__main__":
    run_fast_simulation()

