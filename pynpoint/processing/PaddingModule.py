from pynpoint.core.processing import ProcessingModule
from typing import List, Optional, Tuple

import numpy as np


class PaddingModule(ProcessingModule):
    """
    Pipline module for padding sets of data to the largest image size.
    The padding is performed only in the last two axes/dimensions (in case of
    wavelength data or time data in the first axes). The module accepts only
    3-dimensional datasets
    
    """

    __author__ = "Gian Rungger, Keanu Gleixner"

    def __init__(self,
                 name_in: str,
                 image_in_tags: List[str],
                 image_out_suff: str = "pad",
                 squaring: Optional[bool] = False) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : List[str]
            Tags of the database entries with the various science images that are read as input.
        image_out_suff : str
            Tag suffix for the database entries of the padded images. It will copy the in_tags 
            and add the suffix with an '_' before it. I.e. the out_tag will be of the
            form "image_in_tag"+"_"+"image_out_suff".
        squaring : bool
            If set to True it will not only pad all input tags to the same - largest -
            size, but also make sure that all output images are square in shape.
        """

        super().__init__(name_in)

        self.m_image_in_port_tags_list = image_in_tags
        self.m_image_out_suff = image_out_suff
        self.squaring = squaring

        self.m_image_in_port_arr = []
        self.m_image_out_port_arr = []

        for tag in image_in_tags:
            self.m_image_in_port_arr.append(self.add_input_port(tag))
            self.m_image_out_port_arr.append(
                self.add_output_port(tag+"_"+image_out_suff))

    def padding(self,
                data: np.ndarray,
                shape: Tuple) -> np.ndarray:

        if self.squaring:
            shape = (max(shape[0], shape[1]), max(shape[0], shape[1]))

        assert data.shape[-2] <= shape[0] and data.shape[-1] <= shape[
            1], f'The shape for padding {shape} has to be larger than the given data which has shape {data.shape}.'

        X, Y = shape
        if len(data.shape) == 3:
            wav, a, b = data.shape
            Shape = (wav, X, Y)
        elif len(data.shape) == 2:
            a, b = data.shape
            Shape = (X, Y)

        paddington = np.zeros(Shape)
        # the frames are entered at the center of each new frame and the surplus is full of zeros
        if len(data.shape) == 3:
            paddington[:, int((X-a)/2):int((X+a)/2),
                       int((Y-b)/2):int((Y+b)/2)] = data
        if len(data.shape) == 2:
            paddington[int((X-a)/2):int((X+a)/2),
                       int((Y-b)/2):int((Y+b)/2)] = data

        return paddington

    def run(self) -> None:
        shape = [0, 0]
        for port in self.m_image_in_port_arr:
            port_shape = port.get_shape()
            assert len(port_shape) == 3 or len(port_shape) == 2, "The provided data\
                is not the right shape (only 2 or 3 dimensional datasets are accepted)."
            shape[0] = max(shape[0], port_shape[-2])
            shape[1] = max(shape[1], port_shape[-1])

        for index in range(len(self.m_image_in_port_arr)):
            data = self.m_image_in_port_arr[index].get_all()
            data_padded = self.padding(data, shape)
            self.m_image_out_port_arr[index].set_all(data_padded)
            self.m_image_out_port_arr[index].copy_attributes(
                self.m_image_in_port_arr[index])
            print(
                f'The image {self.m_image_in_port_tags_list[index]} was padded and saved under the tag {self.m_image_in_port_tags_list[index] + "_" + self.m_image_out_suff}.')
            history = f'zeros padding from {self.m_image_in_port_arr[index].get_shape()} to {shape}'
            self.m_image_out_port_arr[index].add_history('Padding', history)

        for port in self.m_image_out_port_arr:
            port.close_port()
