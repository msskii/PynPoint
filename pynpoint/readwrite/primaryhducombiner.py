"""
Module for preparing JWST MIRI data for Pynpoint pipeline
"""

import os
import time
import warnings

from typing import List, Optional, Union

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
                 file_list: Union[str, List[str], None] = [],
                 change_name: Optional[bool] = True,
                 overwrite: bool = False,
                 primarywarn: Optional[bool] = False,
                 only2hdus: Optional[bool] = True) -> None:
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
            the input_dir will be used.
        file_list : str, list(str, ), None
            A path to one or a list of paths to several FITS files should be provided.
            If set to [] or None it will simply read in all files within input_dir.
        change_name : bool
            Prepend "col_" to the filename when saving it to the output directory.
        overwrite : bool
            Overwrite existing data.
        primarywarn : bool
            Warn if no data in PrimaryHDU.
        only2hdus: bool
            If it is known that there are only two HDU's this may improve efficiency.
            If set to False it will forage through all HDU's and look for header data.
            The pixel data will be taken as the first one to appear when looking
            through the HDU's.

        Returns
        -------
        NoneType
            None
        """

        super().__init__(name_in, input_dir=input_dir)

        if not output_dir:
            self.m_output_location = input_dir
        else:
            self.m_output_location = output_dir

        self.m_filedirs = file_list
        self.m_change_name = change_name
        self.m_overwrite = overwrite
        self.m_primarywarn = primarywarn
        self.m_only2hdus = only2hdus

    @typechecked
    def collapse_to_PrimaryHDU(self,
                               fits_file: str,
                               out_path: str,
                               overwrite: bool = False) -> None:
        """
        Function that reads in a single FITS file and collapses the data and headers
        to the primary HDU of a new FITS file, which it then saves to out_path
        

        Parameters
        ----------
        fits_file : str
            Absolute path and filename of the fits file
        out_path: str
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

        assert n > 0, f'The file {fits_file} did not contain any data. '

        hdr = fits.Header()

        if hdu_list[0].data is not None:
            images = hdu_list[0].data.byteswap().newbyteorder()

        elif n > 1:
            for i, item in enumerate(hdu_list[1:]):
                if isinstance(item, fits.hdu.image.ImageHDU) or hdu_list[i+1].data is not None:
                    if self.m_primarywarn:
                        warnings.simplefilter('always', UserWarning)

                        warnings.warn(f"No data was found in the PrimaryHDU "
                                      f"so reading data from the ImageHDU "
                                      f"at number {i+1} instead.")

                    images = hdu_list[i+1].data.byteswap().newbyteorder()
                    break

        for i, n_hdu in enumerate(hdu_list):
            if self.m_only2hdus and i > 1:
                break
            if len(n_hdu.header) > 0:
                hdr.extend(n_hdu.header, unique=True)

        if len(hdr) == 0:
            raise ValueError(f'No header information found in {fits_file}.')

        hdu_list.close()

        assert os.path.isdir(
            out_path), f'The data out path: {out_path} , does not exist. '

        fits.writeto(os.path.join(out_path, filename),
                     data=images,
                     header=hdr,
                     overwrite=overwrite)

    def run(self) -> None:
        """
        Run method of the module.

        Returns
        -------
        None
        
        """
        files = self.m_filedirs
        if not files:  # if no files are given, FITS files will be searched for in the input directory
            if files == None:
                files = []
            for filename in os.listdir(self.m_input_location):
                if filename.endswith('.fits') and not filename.startswith('._'):
                    files.append(os.path.join(self.m_input_location, filename))

            assert files, 'No FITS files were provided and none were found in %s.' % self.m_input_location

        files.sort()

        start_time = time.time()
        for i, fits_file in enumerate(files):
            progress(i, len(files), 'Collapsing FITS files...', start_time)

            self.collapse_to_PrimaryHDU(
                fits_file, self.m_output_location, self.m_overwrite)
