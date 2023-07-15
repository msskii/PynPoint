"""
Module for replacing the data cube for a database entry.
"""

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.apply_func import subtract_psf
from typing import List, Optional, Tuple, Union

import numpy as np
import pdb


class DataCubeReplacer(ProcessingModule):
    """
    Pipline module for replacing the data cube with an inputted array
    
    """
    
    __author__ = "Gian Rungger, Keanu Gleixner"

    def __init__(self,
                 name_in: str,
                 image_in_tag: str,
                 image_out_tag: str,
                 new_cube: np.array) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : List[str]
            Tag of the database entries with the various science images that are read as input.
        image_out : str
            Tag suffix for the database entries of the padded images. It will copy the in_tags and add the suffix with an _ before it
        """

        super().__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
        self.new_data = new_cube
    
    def run(self) -> None:
        self.m_image_out_port.set_all(self.new_data)
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.close_port()