"""
Module for reading FITS files.
"""

import os
import time
import warnings

from typing import List, Optional, Tuple, Union

import numpy as np

from astropy.io import fits
from typeguard import typechecked

from pynpoint.core.processing import ReadingModule
from pynpoint.util.attributes import set_static_attr, set_nonstatic_attr_JWST, set_extra_attr
from pynpoint.util.module import progress


class ArrayReadingModule(ReadingModule):
    """
    Reads a numpy array into the database. Header information can be added to the database entry.
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
            Unique name of the module instance.
        input_dir : str, None
            Input directory where the FITS files are located. If not specified the Pypeline default
            directory is used.
        array_star: np.ndarray
        image_tag : str
            Tag of the read data in the HDF5 database. Non static header information is stored with
            the tag: *header_* + image_tag / header_entry_name.
        overwrite : bool
            Overwrite existing data and header in the central database.
        check : bool
            Print a warning if certain attributes from the configuration file are not present in
            the FITS header. If set to `False`, attributes are still written to the dataset but
            there will be no warning if a keyword is not found in the FITS header.
        filenames : str, list(str, ), None
            If a string, then a path of a text file should be provided. This text file should
            contain a list of FITS files. If a list, then the paths of the FITS files should be
            provided directly. If set to None, the FITS files in the `input_dir` are read. All
            paths should be provided either relative to the Python working folder (i.e., the folder
            where Python is executed) or as absolute paths.
        ifs_data : bool
            Import IFS data which is stored as a 4D array with the wavelength and temporal
            dimensions as first and second dimension, respectively. If set to ``False`` (default),
            the data is imported as a 3D array with the temporal dimension as first dimension.

        Returns
        -------
        NoneType
            None
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

        overwrite_tags = [] 
        array_star = self.array_star
        if self.m_overwrite and self.m_image_out_port.tag not in overwrite_tags:
            overwrite_tags.append(self.m_image_out_port.tag)
        
            dim = len(array_star.shape)
            self.m_image_out_port.set_all(array_star, data_dim=dim)
            self.m_image_out_port.del_all_attributes()
        
        else:
            self.m_image_out_port.append(array_star, data_dim=dim)
            
                
        # self.m_header_out_port = self.add_output_port('fits_header/'+self.m_img_tag)
        # header_out_port = self.m_header_out_port
        # header_out_port.set_all(fits_header)
        
        set_nonstatic_attr_JWST(header=self.header,
                           config_port=self._m_config_port,
                           image_out_port=self.m_image_out_port,
                           check=False)
        

        self.m_image_out_port.flush()

        self.m_image_out_port.close_port()
