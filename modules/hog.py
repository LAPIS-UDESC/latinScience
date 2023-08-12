import numpy as np
from skimage import feature


class HistogramOfOrientedGradients:
    def __init__(
        self,
        orientations,
        pixels_per_cell,
        cells_per_block,
        block_norm,
        transform_sqrt,
        channel_axis,
    ):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
        self.channel_axis = channel_axis
        self.transform_sqrt = transform_sqrt

    def describe(self, image):

        hog = feature.hog(image, orientations=self.orientations, pixels_per_cell=self.pixels_per_cell, cells_per_block=self.cells_per_block,
                          visualize=False, channel_axis=self.channel_axis, transform_sqrt=self.transform_sqrt)
        
        (hist, _) = np.histogram(hog.ravel(),bins=self.orientations)
        
        return hist
