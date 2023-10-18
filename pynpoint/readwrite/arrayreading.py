"""
Module for reading arrays into the database.
"""
from typing import List, Optional, Union

import numpy as np
from typeguard import typechecked

from pynpoint.core.processing import ReadingModule
from pynpoint.util.attributes import set_static_attr, set_nonstatic_attr, set_extra_attr


class ArrayReadingModule(ReadingModule):
    """
    Module for manually reading both a dataset and header information into the pypeline database.
    """

    __author__ = 'Gian Rungger'

    @typechecked
    def __init__(self,
                 name_in: str,
                 array_star: np.ndarray,
                 header: dict,
                 input_dir: Optional[str] = None,
                 image_tag: str = 'im_arr',
                 overwrite: bool = True,
                 check: bool = True,
                 filenames: Optional[Union[str, List[str]]] = None,
                 ifs_data: bool = False) -> None:
        """

        Parameters
        ----------
        name_in : str
            DESCRIPTION.
        array_star : np.ndarray
            DESCRIPTION.
        header : dict
            DESCRIPTION.
        input_dir : Optional[str], optional
            DESCRIPTION. The default is None.
        image_tag : str, optional
            DESCRIPTION. The default is 'im_arr'.
        overwrite : bool, optional
            DESCRIPTION. The default is True.
        check : bool, optional
            DESCRIPTION. The default is True.
        filenames : Optional[Union[str, List[str]]], optional
            DESCRIPTION. The default is None.
        ifs_data : bool, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None
            DESCRIPTION.

        """

        super().__init__(name_in, input_dir=input_dir)

        self.m_image_out_port = self.add_output_port(image_tag)
        self.array_star = array_star
        self.header = header
        self.m_overwrite = overwrite
        self.m_check = check
        self.m_filenames = filenames
        self.m_ifs_data = ifs_data

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Looks for all FITS files in the input directory and imports the
        images into the database. Note that previous database information is overwritten if
        ``overwrite=True``. The filenames are stored as attributes.

        Returns
        -------
        NoneType
            None
        """

        array_star = self.array_star
        if self.m_overwrite:
            dim = len(array_star.shape)
            self.m_image_out_port.set_all(array_star, data_dim=dim)
            self.m_image_out_port.del_all_attributes()
        
        else:
            self.m_image_out_port.append(array_star, data_dim=dim)
            
                
            
        # self.m_header_out_port = self.add_output_port('fits_header/'+self.m_img_tag)
        # header_out_port = self.m_header_out_port
        # header_out_port.set_all(fits_header)
        shape = array_star.shape
        if len(shape) == 2:
            nimages = 1

        elif len(shape) == 3:
            if self.m_ifs_data:
                nimages = 1
            else:
                nimages = shape[0]

        elif len(shape) == 4:
            nimages = shape[1]

        else:
            raise ValueError('Data read from FITS file has an invalid shape.')
        
        set_static_attr(fits_file="Read-in Array", 
                        header=self.header, 
                        config_port=self._m_config_port, 
                        image_out_port=self.m_image_out_port,
                        instrument_key="MIRI")
        
        set_nonstatic_attr(header=self.header, 
                           config_port=self._m_config_port, 
                           image_out_port=self.m_image_out_port, 
                           instrument_key="MIRI")
        
        set_extra_attr(fits_file="Read-in Array", 
                       nimages=nimages, 
                       config_port=self._m_config_port, 
                       image_out_port=self.m_image_out_port, 
                       first_index=0, 
                       instrument_key="MIRI")
        
        # (header=self.header,
        #                    config_port=self._m_config_port,
        #                    image_out_port=self.m_image_out_port,
        #                    check=False)
        

        self.m_image_out_port.flush()

        self.m_image_out_port.close_port()
