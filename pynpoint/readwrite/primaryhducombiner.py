"""
Module for preparing JWST MIRI data for Pynpoint pipeline
"""

import os
import time
import warnings

from typing import List, Optional, Tuple, Union

import numpy as np

from astropy.io import fits
from typeguard import typechecked

from pynpoint.core.processing import ReadingModule
from pynpoint.util.module import progress


class PrimaryHDUCombiner(ReadingModule):
    """
    Combines headers from primary HDU and secondary HDU of a fits file, if there is only data saved
    in one of the two HDU's but there is header data saved in both. If
    Reads FITS files from the given *input_dir* or the default directory of the Pypeline. The FITS
    files need to contain either single images (2D) or cubes of images (3D). Individual images
    should have the same shape and type. The header of the FITS is scanned for the required static
    attributes (should be identical for each FITS file) and non-static attributes. Static entries
    will be saved as HDF5 attributes while non-static attributes will be saved as separate data
    sets in a subfolder of the database named *header_* + image_tag. If the FITS files in the input
    directory have changing static attributes or the shape of the input images is changing a
    warning appears. FitsReadingModule overwrites by default all existing data with the same tags
    in the central database.
    """
    
    __author__ = 'Gian Rungger'

    @typechecked
    def __init__(self,
                 name_in: str,
                 input_dir: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 image_tag: str = 'im_arr',
                 filenames: Optional[Union[str, List[str]]] = None,
                 overwrite: bool = False) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        input_dir : str, None
            Input directory where the FITS files are located. If not specified the Pypeline default
            directory is used.
        output_dir : str, None
            Output directory where the collapsed FITS files are written to. If not specified 
            the Pypeline default directory is used.
        image_tag : str
            Tag of the read data in the HDF5 database. Non static header information is stored with
            the tag: *header_* + image_tag / header_entry_name.
        filenames : str, list(str, ), None
            If a string, then a path of a text file should be provided. This text file should
            contain a list of FITS files. If a list, then the paths of the FITS files should be
            provided directly. If set to None, the FITS files in the `input_dir` are read. All
            paths should be provided either relative to the Python working folder (i.e., the folder
            where Python is executed) or as absolute paths.
        overwrite : bool
            Overwrite existing data.

        Returns
        -------
        NoneType
            None
        """

        super().__init__(name_in, input_dir=input_dir)

        self.m_image_out_port = self.add_output_port(image_tag)
        
        self.m_output_dir = output_dir
        
        self.m_filenames = filenames
        self.m_overwrite = overwrite
        
    @typechecked
    def _txt_file_list(self) -> list:
        """
        Internal function to import a list of FITS files from a text file.
        """

        with open(self.m_filenames) as file_obj:
            files = file_obj.readlines()

        # remove newlines
        files = [x.strip() for x in files]

        # remove of empty lines
        files = filter(None, files)

        return list(files)
    
    
    
    @typechecked
    def collapse_to_PrimaryHDU(self,
                               fits_file: str,
                               out_path: Optional[str] = None,
                               overwrite: bool = False) -> None:
        """
        Function that reads in a 
        

        Parameters
        ----------
        fits_file : str
            Absolute path and filename of the fits file
        out_path: Optional[str]
            Absolute path where the collapsed files are stored.
        overwrite: bool
            Data in the out_path with the same name are appended if overwrite 

        Returns
        -------
        hdr
            Collapsed header is returned.

        """
        
        hdu_list = fits.open(fits_file)
        filename = "col_" + os.path.basename(fits_file)
        hdr = fits.Header()
        
        hdr.extend(hdu_list[0].header)
        hdr.extend(hdu_list[1].header)
        
        data = hdu_list[1].data.byteswap().newbyteorder()
        
        hdu_list.close()
        
        assert os.path.isdir(out_path), f'The data out path: {out_path} , does not exist. '
        
        fits.writeto(os.path.join(out_path, filename), 
                     data = data,
                     header = hdr,
                     overwrite=overwrite)
        
    
    def run(self) -> None:
        """
        Run method of the module.

        Returns
        -------
        None
        
        """
        
        files = []

        if isinstance(self.m_filenames, str):
            files = self._txt_file_list()

            for item in files:
                if not os.path.isfile(item):
                    raise ValueError(f'The file {item} does not exist. Please check that the '
                                     f'path is correct.')

        elif isinstance(self.m_filenames, list):
            files = self.m_filenames

            for item in files:
                if not os.path.isfile(item):
                    raise ValueError(f'The file {item} does not exist. Please check that the '
                                     f'path is correct.')

        elif isinstance(self.m_filenames, type(None)):
            for filename in os.listdir(self.m_input_location):
                if filename.endswith('.fits') and not filename.startswith('._'):
                    files.append(os.path.join(self.m_input_location, filename))

            assert files, 'No FITS files found in %s.' % self.m_input_location
        
        files.sort()

        first_index = 0

        start_time = time.time()
        for i, fits_file in enumerate(files):
            progress(i, len(files), 'Reading FITS files...', start_time)

            self.collapse_to_PrimaryHDU(fits_file, self.m_output_dir, self.m_overwrite)

            first_index += 1

            self.m_image_out_port.flush()

        self.m_image_out_port.close_port()
        
        
        
        