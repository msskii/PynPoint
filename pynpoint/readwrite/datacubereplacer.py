"""
Module for replacing the data cube for a database entry with an inputted array.
"""

from pynpoint.core.processing import ProcessingModule

import numpy as np
import warnings


class DataCubeReplacer(ProcessingModule):
    """
    Pipeline module for replacing the data cube with an inputted array. The header
    data will be taken from the original image_in_tag database entry.
    
    """

    __author__ = "Gian Rungger"

    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 image_out_tag: str,
                 new_cube: np.ndarray) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry with the science header and original data 
            that should be replaced with the array read as input under new_cube.
        image_out_tag : str
            Tag of the database entries of the replaced image.
        new_cube : np.ndarray
            Array with new data.
        """

        super().__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
        self.new_data = new_cube

    def run(self) -> None:
        old_shape = self.m_image_in_port.get_shape()
        new_shape = self.new_data.shape
        # Warnings if wavelength dimension is preserved in new array
        if len(old_shape) == 3:
            if len(new_shape) == 3:
                if old_shape[0] != new_shape[0]:
                    warnings.warn("The wavelength dimension length was changed "
                                  f"from {old_shape[0]} to {new_shape[0]}. \nNote "
                                  "that some of the attributes including \ne.g. "
                                  "wavelength array, channel array, …, possibly "
                                  "require alterations.")
            elif len(new_shape) == 2:
                warnings.warn("The wavelength dimension has been reduced to one frame."
                              "A few attributes might need to be updated.")
            else:
                raise ValueError("The given array dimension is unexpected.")

        if old_shape[-2:] != new_shape[-2:]:
            warnings.warn("The spatial dimension shape was changed "
                          f"from {old_shape[-2:]} to {new_shape[-2:]}. \nNote "
                          "that some of the attributes including e.g. "
                          "NAXIS1, NAXIS2, …, possibly "
                          "require alterations.")

        self.m_image_out_port.set_all(self.new_data)
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.close_port()
