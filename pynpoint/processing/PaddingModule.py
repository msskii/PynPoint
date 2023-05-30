from pynpoint.core.processing import ProcessingModule
from pynpoint.util.apply_func import subtract_psf
from typing import List, Optional, Tuple, Union

import numpy as np
import pdb


class PaddingModule(ProcessingModule):
    """
    Pipline module for padding sets of data to the largest image size.
    The padding is performed only in the "last two" axes/dimensions (in case of
    wavelength data or time data in the first axis)
    
    """
    
    __author__ = "Gian Rungger, Keanu Gleixner"

    def __init__(self,
                 name_in: str,
                 image_in_tags: List[str],
                 image_out_suff: str) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : List[str]
            Tags of the database entries with the various science images that are read as input.
        image_out_suff : str
            Tag suffix for the database entries of the padded images. It will copy the in_tags and add the suffix with an _ before it
        """

        super().__init__(name_in)

        self.m_image_in_port_tags_list = image_in_tags
        self.m_image_out_suff = image_out_suff

        self.m_image_in_port_arr = []
        self.m_image_out_port_arr = []

        for tag in image_in_tags:
            self.m_image_in_port_arr.append(self.add_input_port(tag))
            self.m_image_out_port_arr.append(self.add_output_port(tag+"_"+image_out_suff))
        
        
    def padding(self,
                data: np.ndarray,
                shape: Tuple) -> np.ndarray:
        
        assert data.shape[1]<=shape[0] and data.shape[2]<=shape[1], f'The shape for padding {shape} has to be larger than the given data which has shape {data.shape}.'
        
        wav,a,b = data.shape
        X,Y = shape
        Shape = (wav,X,Y)
        
        paddington = np.zeros(Shape)
        # the frames are entered at the center of each new frame and the surplus is full of zeros
        paddington[:,int((X-a)/2):int((X+a)/2),int((Y-b)/2):int((Y+b)/2)] = data
        
        return paddington
    
    def run(self) -> None:
        # pdb.set_trace()
        shape = [0,0]
        for port in self.m_image_in_port_arr:
            port_shape = port.get_shape()
            print(port_shape)
            assert len(port_shape) == 3, "The provided data is not 3dimensional, this module allows only 3d."
            shape[0] = max(shape[0], port_shape[1])
            shape[1] = max(shape[1], port_shape[2])

        for index in range(len(self.m_image_in_port_arr)):
            data = self.m_image_in_port_arr[index].get_all()
            data_padded = self.padding(data,shape)
            self.m_image_out_port_arr[index].set_all(data_padded)
            self.m_image_out_port_arr[index].copy_attributes(self.m_image_in_port_arr[index])
            print(f'The image {self.m_image_in_port_tags_list[index]} was padded and saved under the tag {self.m_image_in_port_tags_list[index] + "_" + self.m_image_out_suff}.')
            history = f'zeros padding from {self.m_image_in_port_arr[index].get_shape()} to {shape}'
            self.m_image_out_port_arr[index].add_history('Padding', history)

        for port in self.m_image_out_port_arr:
            port.close_port()
