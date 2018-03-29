import numpy as np
import matplotlib.pyplot as plt
import simulation as sim  # self-written classes, functions
import time

# options
n1 = 1.4
n2 = 1
detector_distance = 5  # distance to detector
detector_window = 0.5  # diameter of detector window
detector_steps = 100  # positions detector will scan
n_rays = 5 * 10 ** 5  # number of rays to simulate

# logging
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
surface = sim.RefractiveSurface(n1, n2)
detector = sim.Detector(detector_distance,
                        detector_window)
detector.steps = detector_steps
ray = sim.Ray([0, 0, 0], [1, 1, 1])
# generating initial ray directions
print("Generating initial ray directions...")
directions = sim.get_random_directions(n_rays)
print("Generating initial ray directions... done")
# throwing unimportant rays away to increase speed
print("Throwing unimportant rays away...")
good_directions = []
# maximal y-value ray can have and still hit detector
# noinspection PyProtectedMember
max_y = detector._window_diameter / 2 / detector._distance
for direction in directions:
    # discard if z-value <= 0
    if direction[2] <= 0:
        continue
    # discard if y-value outside detector even before refraction,
    # since refraction will increase y-value
    if np.abs(direction[1]) > max_y:
        continue
    good_directions.append(direction)
directions = np.array(good_directions)
print("Throwing unimportant rays away... done")

# refract rays
print("Refract rays...")
refracted_rays = []
for direction in directions:
    ray.intensity = 1  # reset intensity
    ray.direction = direction
    ray.refract(surface)
    # discard if z-value <= 0
    if ray.direction[2] <= 0:
        continue
    # create 4 dimensional vectors with last dimension intensity
    refracted_rays.append(np.append(ray.direction, ray.intensity))
refracted_rays = np.array(refracted_rays)
print("Refract rays... done")
print(f"Rays that could hit detector: {len(refracted_rays)}")

# check if rays hit detector
print("Check if rays hit detector...")
bins_intensity = []
bins_histo = []
angular_positions = []
# iterating detector positions
for angular_position, detector_direction in detector:
    intensities = []
    angular_positions.append(angular_position)
    for ray0 in refracted_rays:
        # check if detector got hit by comparing angle between ray and detector direction
        # with opening angle of detector
        angle = np.arccos(ray0[:-1] @ detector_direction) * sim.CONVERT_TO_DEG
        if angle <= detector.d_angle:
            intensities.append(ray0[3])
    bins_intensity.append(sum(intensities))
    bins_histo.append(len(intensities))
print("Check if rays hit detector... done")

# plotting
fig = plt.figure(figsize=(15, 10))
plt.grid()
plt.plot(angular_positions, bins_intensity, label="Intensity")
plt.plot(angular_positions, bins_histo, label="Counts")
plt.legend()
plt.xlabel(r"$\mathtt{Angle\/\/[°]}$", fontsize=15)
plt.ylabel(r"$\mathtt{Intensity/Counts\/\/}$", fontsize=15)
plt.savefig(f'runs/{sim_number}.pdf')

# logging
np.savetxt(f"runs/{sim_number}.csv", np.array([angular_positions, bins_intensity,
                                               bins_histo]).T, '%.5f',
           header="angular_position  intensity  counts")
sim.log("Successful run!")
sim.log(f"Ran for: {(time.time() - start_time) / 60: .1f}min")

print("Successful run!")
