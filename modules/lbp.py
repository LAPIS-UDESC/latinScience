from skimage import feature
import numpy as np


class LocalBinaryPatterns:
    def __init__(self, n_points, radius):
        self.n_points = n_points
        self.radius = radius

    def describe(self, image):
        lbp = feature.local_binary_pattern(
            image, self.n_points, self.radius, method="uniform"
        )

        (hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, self.n_points + 3),
			range=(0, self.n_points + 2))

        return hist
