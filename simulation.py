import numpy as np

CONVERT_TO_DEG = 180 / np.pi
CONVERT_TO_RAD = np.pi / 180


class RefractiveSurface:

    def __init__(self, n1: float, n2: float, surface_normal=(0, 0, 1)):
        """Create a refractive surface.

        - position can be neglected in this simulation.
        - surface_normal [list of length 3]: doesn't have to be normalized,
            must point in direction of detector
        - n1/n2: refractive index of incoming/outgoing light medium,
            n1 > n2
        - surface_normal/n1/n2 shouldn't be changed later on
        """

        # normalizing
        self.surface_normal = np.array(surface_normal) / np.linalg.norm(surface_normal)
        self.n1 = n1
        self.n2 = n2
        self.total_reflection = np.arcsin(n2 / n1)
        print(f'Surface total reflection: {self.total_reflection * CONVERT_TO_DEG}°')


class Ray:

    def __init__(self, position, direction):
        """Create a ray.

        - position [list of length 3]
        - direction [list of length 3]: doesn't have to be normalized
        - returns: nothing
        """
        self._direction = None
        self._position = None
        self.position = position
        self.direction = direction
        self.intensity = 1

    def refract(self, refractive_surface: RefractiveSurface):
        """Change direction of ray according to refraction law.

        - to make sense, position of ray should be on surface, this has to be done manual
        - calculation from https://www.scratchapixel.com/lessons/3d-basic-rendering/
                            introduction-to-shading/reflection-refraction-fresnel
        """
        N = refractive_surface.surface_normal
        I = self.direction
        n1 = refractive_surface.n1
        n2 = refractive_surface.n2

        # angle of incident
        theta1 = np.arccos(I @ N)  # in rad

        # case: ray doesn't hit surface (theta1 >= 90°)
        # direction stays as before
        if theta1 >= 90 * CONVERT_TO_RAD:
            # print("Ray doesn't hit surface.")
            # intensity stays the same
            return self.direction

        # case: total reflection (theta2 > 90°), which can be checked by sin_theta2 > 1
        # new direction doesn't matter for our simulation,
        # roughly along surface -> delete z component
        # intensity stays as before
        sin_theta2 = n1 / n2 * np.sin(theta1)
        if sin_theta2 > 1:
            # print(f'Total reflection at {theta1 * CONVERT_TO_DEG}° > '
            #       f'{refractive_surface.total_reflection * CONVERT_TO_DEG}°.')
            # self.direction automatic normalized via @property
            self.direction = [self.direction[0], self.direction[1], 0]
            return self.direction

        theta2 = np.arcsin(sin_theta2)  # in rad
        # print(f'Angle in: {theta1 * CONVERT_TO_DEG}, '
        #       f'Angle out: {theta2 * CONVERT_TO_DEG}')

        # checking if inverse worked
        # should never happen, all cases checked already
        # if np.isnan(theta1) or np.isnan(theta2):
        #     raise Exception("Angles are outside range!")

        # case: refraction happens
        # direction
        A = n1 / n2 * (I - N * np.cos(theta1))
        B = N * np.cos(theta2)
        T = A + B
        # intensity
        self.intensity *= self.get_fresnel_refracted(n1, n2, theta1, theta2)
        # print(f"Fraction refracted: {self.intensity}")
        self.direction = T
        return T, self.intensity

    @staticmethod
    def get_fresnel_refracted(n1: float, n2: float, theta1: float, theta2: float):
        """Calculates how much percent of light gets refracted according to Fresnel
        equations.

        - theta1/theta2: in rad
        - returns: value in [0, 1], part of the intensity that gets refracted"""

        F_R_parallel = ((n2 * np.cos(theta1) - n1 * np.cos(theta2)) / (n2 * np.cos(
            theta1) + n1 * np.cos(theta2))) ** 2

        F_R_perp = ((n1 * np.cos(theta2) - n2 * np.cos(theta1)) / (n1 * np.cos(
            theta2) + n2 * np.cos(theta1))) ** 2

        F_R = 1 / 2 * (F_R_parallel + F_R_perp)
        F_T = 1 - F_R
        return F_T

    def shift(self, length: float):
        """Shift ray to new position along direction.

        - length: length of which to shift ray
        - not need for current simulation
        """

        self.position = self.position + self.direction * length

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position):
        self._position = np.array(position)

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, direction):
        self._direction = np.array(direction) / np.linalg.norm(direction)


class Detector:

    def __init__(self, distance: float, window_diameter: float, steps=100):
        """Create detector.

        - distance: distance from point source to detector
        - window_size: detector window diameter
        - steps: number of steps detector makes when scanning"""

        self.angular_position = -90  # in deg, initial position
        self.steps = steps
        # private variables, shouldn't be changed directly
        self._distance = distance
        self._window_diameter = window_diameter

        # d_angle: angle between ray and detector direction must be < d_angle for ray
        #       to hit detector
        self.d_angle = np.arctan(
            window_diameter / 2 / distance) * CONVERT_TO_DEG  # in deg
        print(f"d_angle: {self.d_angle}")

    def get_direction(self):
        """Calculate the detectors direction from the angular position (in the z-y-plane).

        - right-handed coordinate system with z-axis towards detector at 0°
        """

        x = np.sin(self.angular_position * CONVERT_TO_RAD)
        y = 0
        z = np.cos(self.angular_position * CONVERT_TO_RAD)
        direction = np.array([x, y, z])
        return direction

    def __iter__(self):
        """Create iterator which iterates detector position.

        steps: number of steps to make"""

        self.step_size = 180 / (self.steps - 1)  # correcting for 0
        # starting position
        self.angular_position = -90
        return self

    def __next__(self):
        if self.angular_position <= 90:
            direction = self.get_direction()
            # print(f"angular position: {self.angular_position}")
            self.angular_position += self.step_size
            return direction
        else:
            raise StopIteration()


def get_random_directions(n_directions: int):
    """Generate random directions/points on the unit sphere.

    - last method from http://mathworld.wolfram.com/SpherePointPicking.html
    - n_directions: number of directions to generate
    """

    vectors = np.random.randn(3, n_directions)
    vectors /= np.linalg.norm(vectors, axis=0)
    return vectors.T
