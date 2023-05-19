"""
Module for reading a data cube dispersed over several channels saved as FITS files.
An attribute describing the channel of each part is added.
Could be adjusted to read in several FITS files under one tag into data base.
"""

import os
import shutil
import time

from typing import List, Optional, Tuple, Union, Dict
import string
import warnings

import numpy as np

from astropy.io import fits
from typeguard import typechecked

from pynpoint.core.processing import ReadingModule
from pynpoint.util.attributes import set_static_attr_JWST, set_nonstatic_attr_JWST, set_extra_attr_JWST
from pynpoint.readwrite.primaryhducombiner import PrimaryHDUCombiner
from pynpoint.util.module import progress

class MultiChannelReader(ReadingModule):
    """
    Reads several FITS files within *input dir* and saves them in the database under
    the same tag.
    """
    
    @typechecked
    def __init__(self,
                 name_in: str,
                 input_dir: Optional[str] = None,
                 image_tag: str = 'im_arr',
                 overwrite: bool = True,
                 check: bool = True,
                 filenames: Optional[Union[str, List[str]]] = None,
                 collapser: bool = True,
                 ifs_data: bool = False) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        input_dir : str, None
            Input directory where the FITS files are located. If not specified the Pypeline default
            directory is used.
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
        collapser : bool
            The data is collapsed to the PrimaryHDU if set to True using the PrimaryHDUCombiner module.
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
        
        self.m_img_tag = image_tag
        self.m_overwrite = overwrite
        self.m_check = check
        self.m_filenames = filenames
        self.m_collapse = collapser
        self.m_ifs_data = ifs_data


    def FillCubesZeros(self,imgs,shape,ch_len):
        """
        Concatenates the entries in the first axis of the imgs array into an array
        of sufficient size (specifically shape) where if the imgs[i] entry is smaller
        in the two dimensions it will "pad" the rest of the array with zeros

        Parameters
        ----------
        imgs : array(dtype=object)
        data cube with correct order with fourth dimension being channel in correct order
        shape : 3-Tuple 
        total shape of final cube
        ch_len : array
        list of the lengths of the individual channels

        Returns
        -------
        images : 3d array

        """
        N = len(imgs)
        Nwav = shape[0]
        X = shape[1]
        Y = shape[2]
        
        cube = np.zeros(shape)
        ch_len_ind = np.copy(ch_len)
        for i in np.arange(N):
            ch_len_ind[i] = ch_len[:i+1].sum()
        
        for i in np.arange(N):
            wav,a,b = imgs[i].shape
            s = int(ch_len[0:i].sum())
            e = int(ch_len[0:i+1].sum())
            # the frames are entered at the center of each new frame and the surplus is full of zeros
            cube[s:e,int((X-a)/2):int((X+a)/2),int((Y-b)/2):int((Y+b)/2)] = imgs[i]
            
        cube = np.nan_to_num(cube)
        
        return cube


    @typechecked
    def read_files(self,
                         files: list,
                         overwrite_tags: list) -> Tuple[List[str],Dict, tuple, np.ndarray, np.ndarray, np.ndarray]:
        """
        Function which reads a single FITS file and appends it to the database. The function gets
        a list of *overwriting_tags*. If a new key (header entry or image data) is found that is
        not on this list the old entry is overwritten if *self.m_overwrite* is active. After
        replacing the old entry the key is added to the *overwriting_tags*. This procedure
        guaranties that all previous database information, that does not belong to the new data
        set that is read by FitsReadingModule is replaced and the rest is kept.

        Parameters
        ----------
        files : str
            Absolute path and filename of the FITS files.
        overwrite_tags : list(str, )
            The list of database tags that will not be overwritten.

        Returns
        -------
        List[str]
            header as list
        Dict
            header as dictionary
        tuple(int, )
            Image shape.
        np.ndarray
            wavelength array
        np.ndarray
            channel array
        """
        hdu_lists = []
        imgs = []
        headers = []
        N = len(files)
        # find largest dimension:
        maxNM = np.zeros(2)
        start_time = time.time()
        for i in np.arange(N):
            progress(int(i), int(N), 'Reading FITS files...', start_time)
            hdu_lists.append(fits.open(files[i]))
            hdu_list = hdu_lists[i]
            
            if hdu_list[0].data is not None:
                imgs.append(hdu_list[0].data.byteswap().newbyteorder())
                headers.append(hdu_list[0].header)
            else:
                raise RuntimeError(f"No data was found in {files[i]}.")
                
            if imgs[i].ndim == 4 and not self.m_ifs_data:
                raise ValueError('The input data is 4D but ifs_data is set to False. Reading in 4D '
                                 'data is only possible by setting the argument to True.')
    
            if imgs[i].ndim < 3 and self.m_ifs_data:
                raise ValueError('It is only possible to read 3D or 4D data when ifs_data is set to '
                                 'True.')
            if imgs[i].shape[1] > maxNM[0]:
                maxNM[0] = imgs[i].shape[1]
            if imgs[i].shape[2] > maxNM[1]:
                maxNM[1] = imgs[i].shape[2]
        
        
        # We add the JWST specific order of bands, 1 short, 1 medium, 1 long, 2 short, etc
        index_order = np.zeros(N,dtype=int)
        # the data channel/band located at images[i] should be saved at position j = index_order[i]
        
        
        for i in np.arange(N):
            header = headers[i]
            ch = int(header["CHANNEL"])
            bd = header["BAND"]
            if bd == "SHORT":
                bdnr = 0
            elif bd == "MEDIUM":
                bdnr = 1
            elif bd == "LONG":
                bdnr = 2
            else:
                raise ValueError(f"The name of the band {bd} is unexpected.")
            index = int(3*(ch-1)+ bdnr)
            index_order[i] = index
            
        # we have to account for the case where not all bands are inputted:
        if np.array([i not in index_order for i in np.arange(N)]).sum() != 0:
            # make the indexes ascending
            indeces = np.sort(index_order)
            for j in np.arange(N):
                s = int(np.where(index_order==indeces[j])[0][0])
                index_order[s] = j
        # Now index_order is an array of size N with an index between 0 and N-1 that tells you
        # that if index_order[i] == j then images[i] will be saved at array position j in the
        # concatenated data cube.
        
        # array with the length of channel i at index i in the wavelength dimension
        channel_lengths = np.zeros(N)
        
        # Create Data Cube with correct order:
        data_cube = np.zeros(N,dtype=object)
        
        for i in np.arange(N):
            j = np.where(index_order==i)[0][0]
            data_cube[i] = imgs[j]
            channel_lengths[i] = headers[j]["NAXIS3"]
        # Total wavelength dimension size of data cube
        N_wave = channel_lengths.sum()
        # array with the wavelengths of each frame saved at the corresponding location
        wavelength_arr = np.zeros(int(N_wave))
        # array with the channel and band of each frame saved at the corresponding location
        channel_arr = np.zeros(int(N_wave),dtype=str)
        band_arr =  np.zeros(int(N_wave),dtype=np.dtype('<U6'))
        
        # Determine wavelengths:
        channel_lengths_ind = np.copy(channel_lengths)
        for i in np.arange(N):
            channel_lengths_ind[i] = channel_lengths[:i+1].sum()
            if i == 0:
                low = 0
            else:
                low = int(channel_lengths_ind[i-1])
            high = int(channel_lengths_ind[i])
            j = np.where(index_order==i)[0][0]
            start_wav = headers[j]["CRVAL3"]
            wav_incr = headers[j]["CDELT3"]
            stop_wav = start_wav + (channel_lengths[i]-1) * wav_incr
            
            channel = headers[j]["CHANNEL"]
            band = headers[j]["BAND"]
            
            wavelength_arr[low:high] = np.linspace(start_wav,stop_wav,int(channel_lengths[i]))
            for c in np.arange(high-low):
                channel_arr[low:high][c] = channel
                band_arr[low:high][c] = band
        
        shape = (int(N_wave), int(maxNM[0]), int(maxNM[1]))
        data_cube = self.FillCubesZeros(data_cube,shape,channel_lengths)
        

        fits_header = []
        fits_dict = {}
        hdr_template = headers[0]
        for key in list(hdr_template.keys()):
            if '\n' in key:
                continue
            tobeornottobe = True
            # if all keys have the same value we leave 'tobeornottobe' on True
            for i in np.arange(N):
                hdr_prev = headers[i-1]
                hdr = headers[i]
                if hdr[key] != hdr_prev[key]:
                    tobeornottobe = False
            if key == 'NAXIS3':
                fits_header.append(f'{key} = {N_wave}')
                fits_dict[key] = N_wave
                continue
            
            if tobeornottobe:
                fits_header.append(f'{key} = {headers[0][key]}')
                fits_dict[key] = headers[0][key]
            else:
                card = np.zeros(N,dtype=object)
                for i in np.arange(N):
                    card[i] = headers[i][key]
                card = card.astype('S') # convert to numpy bytes array which is compatible with HDF5 database
                fits_header.append(f'{key} = {card}')
                fits_dict[key] = card
        # New attributes:
        fits_header.append(f'WAV_ARR = {wavelength_arr}')
        fits_dict['WAV_ARR'] = wavelength_arr
        fits_header.append(f'CHAN_ARR = {channel_arr}')
        fits_dict['CHAN_ARR'] = channel_arr
        fits_header.append(f'BAND_ARR = {band_arr}')
        fits_dict['BAND_ARR'] = band_arr
        
        # now add the data cube to the database
        if data_cube.ndim == 4 and not self.m_ifs_data:
            raise ValueError('The input data is 4D but ifs_data is set to False. Reading in 4D '
                             'data is only possible by setting the argument to True.')

        if data_cube.ndim < 3 and self.m_ifs_data:
            raise ValueError('It is only possible to read 3D or 4D data when ifs_data is set to '
                             'True.')

        if self.m_overwrite and self.m_image_out_port.tag not in overwrite_tags:
            overwrite_tags.append(self.m_image_out_port.tag)

            if self.m_ifs_data:
                self.m_image_out_port.set_all(data_cube, data_dim=4)
            else:
                self.m_image_out_port.set_all(data_cube, data_dim=3)

            self.m_image_out_port.del_all_attributes()

        else:
            if self.m_ifs_data:
                self.m_image_out_port.append(data_cube, data_dim=4)
            else:
                self.m_image_out_port.append(data_cube, data_dim=3)

        hdu_list.close()
        
        
        self.m_header_out_port = self.add_output_port('fits_header/'+self.m_img_tag)
        header_out_port = self.m_header_out_port
        header_out_port.set_all(fits_header)

        return fits_header, fits_dict, data_cube.shape, wavelength_arr, channel_arr, band_arr

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
        
        if self.m_collapse:
            tempout = os.path.join(self.m_input_location,"temp")
            if os.path.isdir(tempout):
                warnings.warn(f'The temporary folder {tempout} already exists in the input directory.')
                for char in string.printable:
                    if os.path.isdir(os.path.join(self.m_input_location,"temp"+char)):
                        warnings.warn(f'The temporary folder {os.path.join(self.m_input_location,"temp"+char)} already exists in the input directory.')
                        continue
                    else:
                        tempout = os.path.join(self.m_input_location,"temp"+char)
                        break
            os.mkdir(tempout) # create a temporary folder for the collapsed data
            corecollapsesupernova = PrimaryHDUCombiner(name_in = "neutronstar",
                                                       input_dir = self.m_input_location,
                                                       output_dir = tempout,
                                                       filenames = files,
                                                       change_name = False,
                                                       overwrite = True)
        
            corecollapsesupernova.run()
            files = []
            for filename in os.listdir(tempout):
                if filename.endswith('.fits') and not filename.startswith('._'):
                    files.append(os.path.join(tempout, filename))
        
            print(f'FITS files have been collapsed to PrimaryHDU and saved to {tempout}.')
        files.sort()
        overwrite_tags = []
        first_index = 0
        print("Reading all data and headers into Pynpoint.")
        header,header_dict, shape, wav_arr, chan_arr, band_arr = self.read_files(files,overwrite_tags)
        
        if self.m_collapse:
            shutil.rmtree(tempout) # remove the temporary folder and the collapsed data
        
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

        print("Setting attributes in database.")
        set_static_attr_JWST(fits_file=files[0],
                        header=header_dict,
                        config_port=self._m_config_port,
                        image_out_port=self.m_image_out_port,
                        check=self.m_check)

        set_nonstatic_attr_JWST(header=header_dict,
                           config_port=self._m_config_port,
                           image_out_port=self.m_image_out_port,
                           check=self.m_check)

        wavandchan = [('WAV_ARR',wav_arr),('CHAN_ARR',chan_arr),('BAND_ARR',band_arr)]
        set_extra_attr_JWST(fits_file=files[0],
                       nimages=nimages,
                       config_port=self._m_config_port,
                       image_out_port=self.m_image_out_port,
                       first_index=first_index,
                       optional_attrs=wavandchan)

        first_index += nimages

        self.m_image_out_port.flush()

        self.m_image_out_port.close_port()