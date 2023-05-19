from pynpoint.core.processing import ProcessingModule
from pynpoint.util.apply_func import subtract_psf
from typing import List, Optional, Tuple, Union

import numpy as np
import pdb


class PaddingModule(ProcessingModule):
    """
    Pipline module for padding two sets of data to the largest size.
    The padding is performed only in the "last two" axes/dimensions (in case of
    wavelength data or time data in the first axis)
    
    """
    
    __author__ = "Gian Rungger"

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
        
        self.m_image_in_port_sci = self.add_input_port(image_in_tags[0])
        self.m_image_in_port_ref = self.add_input_port(image_in_tags[1])
        
        
        self.m_image_out_suff = image_out_suff
        
        self.m_image_out_port_sci = self.add_output_port(image_in_tags[0]+"_"+image_out_suff)
        self.m_image_out_port_ref = self.add_output_port(image_in_tags[1]+"_"+image_out_suff)
        
        
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
        shape = (0,0)
        
        sci = self.m_image_in_port_sci.get_all()
        assert len(sci.shape) == 3, "The provided data is not 3dimensional, this module allows only 3d."
        if shape[0] < sci.shape[1]:
            better0 = sci.shape[1]
        else:
            better0 = shape[0]
        if shape[1] < sci.shape[2]:
            better1 = sci.shape[2]
        else:
            better1 = shape[1]
        shape = (better0,better1)
        ref = self.m_image_in_port_ref.get_all()
        assert len(ref.shape) == 3, "The provided data is not 3dimensional, this module allows only 3d."
        if shape[0] < ref.shape[1]:
            better0 = ref.shape[1]
        else:
            better0 = shape[0]
        if shape[1] < ref.shape[2]:
            better1 = ref.shape[2]
        else:
            better1 = shape[1]
        shape = (better0,better1)
        
        data = self.padding(sci,shape)
        self.m_image_out_port_sci.set_all(data)
        self.m_image_out_port_sci.copy_attributes(self.m_image_in_port_sci)
        print(f'The image {self.m_image_in_port_tags_list[0]} was padded and saved under the tag {self.m_image_in_port_tags_list[0]+"_"+self.m_image_out_suff}.')
        history = f'zeros padding from {self.m_image_in_port_sci.get_all().shape} to {shape}'
        self.m_image_out_port_sci.add_history('Padding', history)
        # self.m_image_out_port_sci.close_port()
        
        data = self.padding(ref,shape)
        self.m_image_out_port_ref.set_all(data)
        self.m_image_out_port_ref.copy_attributes(self.m_image_in_port_ref)
        print(f'The image {self.m_image_in_port_tags_list[0]} was padded and saved under the tag {self.m_image_in_port_tags_list[0]+"_"+self.m_image_out_suff}.')
        history = f'zeros padding from {self.m_image_in_port_ref.get_all().shape} to {shape}'
        self.m_image_out_port_ref.add_history('Padding', history)
        self.m_image_out_port_ref.close_port()
        self.m_image_out_port_sci.close_port()