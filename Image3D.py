import numpy as np
from scipy.interpolate import interpn

class Image3D:
    def __init__(self, image: np.ndarray, voxel_dimensions: tuple):
        """
        Constructor for the Image3D class.

        Parameters:
        image (np.ndarray): A 3D numpy array representing the image.
        voxel_dimensions (tuple): A tuple of three numerical items for voxel dimensions.
        """
        self.image = image
        self.voxel_dimensions = voxel_dimensions

        # Define a local image coordinate system
        # The origin is at the center of the image, and the unit is the voxel size.
        self.coordinates = np.indices(self.image.shape).T - np.array(self.image.shape) / 2

    def volume_resize(self, resize_ratio: tuple):
        """
        Resize the volume of the image.

        Parameters:
        resize_ratio (tuple): A three-item tuple specifying the resize ratio.

        Returns:
        Image3D: An object of the Image3D class.
        """
        # Compute the new shape of the image
        new_shape = tuple(int(dim * ratio) for dim, ratio in zip(self.image.shape, resize_ratio))

        # Compute the new voxel dimensions
        new_voxel_dimensions = tuple(dim / ratio for dim, ratio in zip(self.voxel_dimensions, resize_ratio))

        # Compute the new coordinates
        new_coordinates = np.indices(new_shape).T - np.array(new_shape) / 2

        # Interpolate the image
        new_image = interpn(self.coordinates, self.image, new_coordinates, method='linear')

        return Image3D(new_image, new_voxel_dimensions)