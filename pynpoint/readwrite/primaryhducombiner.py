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
    in one of the two HDU's but there is header data saved in both. If the data is not located in 
    the PrimaryHDU a warning can be displayed by setting primarywarn to True. The data is written to
    a FITS file with the data in the PrimaryHDU and the header information from the PrimaryHDU and
    the HDU containing the data also saved in the PrimaryHDU. The generated FITS file is written to
    output_dir with the identical name as the input and optionally with "col_" prepended to the 
    file name if change_name is set to True; 
    old files with the same name are overwritten if overwrite is set to True.
    """
    
    __author__ = 'Gian Rungger'

    @typechecked
    def __init__(self,
                 name_in: str,
                 input_dir: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 filenames: Optional[Union[str, List[str]]] = None,
                 change_name: Optional[bool] = True,
                 overwrite: bool = False,
                 primarywarn: Optional[bool] = False) -> None:
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
        filenames : str, list(str, ), None
            If a string, then a path of a text file should be provided. This text file should
            contain a list of FITS files. If a list, then the paths of the FITS files should be
            provided directly. If set to None, the FITS files in the `input_dir` are read. All
            paths should be provided either relative to the Python working folder (i.e., the folder
            where Python is executed) or as absolute paths.
        change_name : bool
            Prepend "col_" to the filename when saving it to the output directory.
        overwrite : bool
            Overwrite existing data.
        primarywarn : bool
            Warn if no data in PrimaryHDU.

        Returns
        -------
        NoneType
            None
        """

        super().__init__(name_in, input_dir=input_dir)
        
        self.m_output_location = output_dir
        
        self.m_filenames = filenames
        self.m_change_name = change_name
        self.m_overwrite = overwrite
        self.m_primarywarn = primarywarn
        
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
        Function that reads in a single FITS file and collapses the data and headers
        to the primary HDU of a new FITS file, which it then saves to out_path
        

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
        n = len(hdu_list)
        if self.m_change_name:
            filename = "col_" + os.path.basename(fits_file)
        else:
            filename = os.path.basename(fits_file)
        
        assert n>0, f'The file {fits_file} did not contain any data. '
        
        hdr = fits.Header()
        
        if hdu_list[0].data is not None:
            images = hdu_list[0].data.byteswap().newbyteorder()
            theone = 0
            
        elif len(hdu_list) > 1:
            for i, item in enumerate(hdu_list[1:]):
                if isinstance(item, fits.hdu.image.ImageHDU):
                    if self.m_primarywarn:
                        warnings.simplefilter('always', UserWarning)
    
                        warnings.warn(f"No data was found in the PrimaryHDU "
                                      f"so reading data from the ImageHDU "
                                      f"at number {i+1} instead.")

                    images = hdu_list[i+1].data.byteswap().newbyteorder()
                    theone = i+1

                    break
        
        if theone == 0:
            k = 1
        else:
            k = 0
        if len(hdu_list[theone].header) > 0:
            hdr.extend(hdu_list[theone].header,unique=True)
        if len(hdu_list) > 1 and len(hdu_list[k].header) > 0:
            hdr.extend(hdu_list[k].header,unique=True)
        if len(hdr) == 0:
            raise ValueError(f'No header information found in {fits_file}.')
        
        hdu_list.close()
        
        assert os.path.isdir(out_path), f'The data out path: {out_path} , does not exist. '
        
        fits.writeto(os.path.join(out_path, filename), 
                     data = images,
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
            progress(i, len(files), 'Collapsing FITS files...', start_time)

            self.collapse_to_PrimaryHDU(fits_file, self.m_output_location, self.m_overwrite)

            first_index += 1
        
        
        
        