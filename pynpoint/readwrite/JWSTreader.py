"""
Module for reading a data cube dispersed over several channels saved as FITS files.
An attribute describing the channel of each part is added.
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
from pynpoint.util.attributes import set_static_attr, set_nonstatic_attr, set_extra_attr
from pynpoint.readwrite.primaryhducombiner import PrimaryHDUCombiner
from pynpoint.util.module import progress


class MultiChannelReader(ReadingModule):
    """
    Reads several FITS files within *input dir* and saves them in the database under
    the same tag. Note that all the data and header information need to be stored
    in the PrimaryHDU of the FITS files; if they aren't, collapser = True can be used
    to use the module PrimaryHDUCombiner which creates new temporary files with all 
    data now saved in the PrimaryHDU.
    """

    __author__ = 'Gian Rungger'

    @typechecked
    def __init__(self,
                 name_in: str,
                 input_dir: Optional[str] = None,
                 image_tag: str = 'im_arr',
                 overwrite: bool = True,
                 check: bool = False,
                 filenames: Optional[Union[str, List[str]]] = None,
                 collapser: bool = True,
                 ifs_data: bool = False,
                 only2hdus: Optional[bool] = True) -> None:
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
        only2hdus: bool
            If the data and header information in the FITS files are known to only be stored
            in a primary and a secondary HDU and no more, this can be set to True, improving 
            performance slightly

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
        self.m_only2hdus = only2hdus

    @typechecked
    def FillCubesZeros(self,
                       imgs: np.ndarray,
                       shape: Tuple,
                       ch_len: np.ndarray) -> np.ndarray:
        """
        Concatenates the entries of imgs in the first axis into an array
        of sufficient size (specifically shape) where if the imgs[i] entry is smaller
        in the two dimensions than given by shape it will "pad" 
        the rest of the array with zeros.

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
        cube : 3d array
            The concatenated and padded data cube.

        """
        N = len(imgs)
        X = shape[1]
        Y = shape[2]

        cube = np.zeros(shape)
        ch_len_ind = np.copy(ch_len)
        for i in np.arange(N):
            ch_len_ind[i] = ch_len[:i+1].sum()

        for i in np.arange(N):
            wav, a, b = imgs[i].shape
            s = int(ch_len[0:i].sum())
            e = int(ch_len[0:i+1].sum())
            # the frames are entered at the center of each new frame and the surplus is full of zeros
            cube[s:e, int((X-a)/2):int((X+a)/2), int((Y-b)/2):int((Y+b)/2)] = imgs[i]

        cube = np.nan_to_num(cube)

        return cube

    @typechecked
    def index_order_setter(self,
                           header_list: list,
                           ordering_keyword: str = "JWST") -> np.ndarray:
        """
        Function which creates the desired index order. For "JWST" data this order
        is by ascending wavelength. The individual channels and bands of the 
        MIRI IFU have to be sorted in order channel 1 band SHORT, MEDIUM, LONG,
        then channel 2 band SHORT, MEDIUM, LONG, etc.
        Another implementation with keyword "read" will simply leave the order 
        of the files just as they were read into header_list.
        The output index_order is an array of size N = len(header_list) with an index between 
        0 and N-1 that tells you that if index_order[i] == j then images[i] will be 
        saved at array position j in the concatenated data cube.

        Parameters
        ----------
        header_list : list
            List with the headers of all FITS files that will be concatenated
        ordering_keyword: str
            Keyword used to determine in what way the files will be ordered.
            "read" will leave the order of header_list,
            "JWST" will sort by channel and band in ascending wavelength order.

        Returns
        -------
        index_order: np.ndarray
            index_order is an array of size N with an index between 0 and N-1 that tells you
            that if index_order[i] == j then images[i] will be saved at array position j in the
            concatenated data cube.

        """
        N = len(header_list)
        if ordering_keyword == "JWST":
            # We add the JWST specific order of bands, 1 short, 1 medium, 1 long, 2 short, etc
            index_order = np.zeros(N, dtype=int)
            # the data channel/band located at images[i] should be saved at position j = index_order[i]
            for i in np.arange(N):
                header = header_list[i]
                ch = int(header["CHANNEL"])
                bd = header["BAND"]
                if bd == "SHORT":
                    bdnr = 0
                elif bd == "MEDIUM":
                    bdnr = 1
                elif bd == "LONG":
                    bdnr = 2
                else:
                    raise ValueError(
                        f"The name of the band {bd} is unexpected.")
                index = int(3*(ch-1) + bdnr)
                index_order[i] = index

            # we have to account for the case where not all bands are inputted:
            if np.array([i not in index_order for i in np.arange(N)]).sum() != 0:
                # make the indeces ascending
                indeces = np.sort(index_order)
                for j in np.arange(N):
                    s = int(np.where(index_order == indeces[j])[0][0])
                    # we rename the indeces; instead of using 3*(ch-1)+bdnr which might be out of bounds we use an index less than N.
                    index_order[s] = j
        elif ordering_keyword == "read":
            index_order = np.arange(N)
        else:
            index_order = None
            raise ValueError("The ordering keyword is not valid.")
        return index_order

    @typechecked
    def wave_chan_bd_arr_generator(self,
                                   index_order: np.ndarray,
                                   channel_lengths: np.ndarray,
                                   header_list: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Function which generates three arrays with length such that there is an entry for each wavelength
        frame in the datacube. The wavelength_arr denotes the wavelength in um of the data frame at the
        same index, the channel_arr the detector channel and band_arr the band.

        Parameters
        ----------
        index_order : np.ndarray
            The order in which header_list will be concatenated into the final data cube.
        channel_lengths : np.ndarray
            Array with an entry for each FITS file that denotes how many frames each file contains.
        header_list : list
            All header information, one entry for each FITS file.

        Returns
        -------
        wavelength_arr : np.ndarray
            wavelengths for final data cube
        channel_arr : np.ndarray
            channels for final data cube
        band_arr : np.ndarray
            bands for final data cube

        """
        N = len(header_list)
        # Total wavelength dimension size of data cube
        N_wave = channel_lengths.sum()

        # array with the wavelengths of each frame saved at the corresponding location
        wavelength_arr = np.zeros(int(N_wave))
        # array with the channel and band of each frame saved at the corresponding location
        channel_arr = np.zeros(int(N_wave))
        band_arr = np.zeros(int(N_wave), dtype=object)

        # Determine wavelengths:
        channel_lengths_ind = np.copy(channel_lengths)
        for i in np.arange(N):
            channel_lengths_ind[i] = channel_lengths[:i+1].sum()
            if i == 0:
                low = 0
            else:
                low = int(channel_lengths_ind[i-1])
            high = int(channel_lengths_ind[i])
            j = np.where(index_order == i)[0][0]
            start_wav = header_list[j]["CRVAL3"]
            wav_incr = header_list[j]["CDELT3"]
            stop_wav = start_wav + (channel_lengths[i]-1) * wav_incr

            channel = header_list[j]["CHANNEL"]
            band = header_list[j]["BAND"]

            for j in np.arange(low, high):
                wavelength_arr[low:high] = np.linspace(
                    start_wav, stop_wav, int(channel_lengths[i]))
            for c in np.arange(high-low):
                channel_arr[low:high][c] = channel
                band_arr[low:high][c] = band

        return wavelength_arr, channel_arr, band_arr

    @typechecked
    def read_headers(self,
                     header_list: list,
                     index_order: np.ndarray,
                     N_wave: int) -> Tuple[List[str], Dict]:
        """
        Function that reads all header information into both a list with entries 
        of the form 'key = value' and a dictionary. It upholds the same order as
        the data cube ordering.

        Parameters
        ----------
        header_list : list
            List with headers of each read FITS file.
        index_order : np.ndarray
            The order in which the data cube is produced from the hdu_list order.
        N_wave : int
            Size of wavelength dimension.

        Returns
        -------
        (Tuple[List[str], Dict])
            The header data both as a list and as a dictionary.

        """
        N = len(header_list)
        fits_header = []
        fits_dict = {}
        hdr_template = header_list[0]
        for key in list(hdr_template.keys()):
            if '\n' in key:
                continue
            tobeornottobe = True
            # if all keys have the same value we leave 'tobeornottobe' on True
            for i in np.arange(N):
                hdr_prev = header_list[i-1]
                hdr = header_list[i]
                if hdr[key] != hdr_prev[key]:
                    tobeornottobe = False
            if key == 'NAXIS3':
                fits_header.append(f'{key} = {N_wave}')
                fits_dict[key] = N_wave
                continue

            if tobeornottobe:
                fits_header.append(f'{key} = {header_list[0][key]}')
                fits_dict[key] = header_list[0][key]
            else:
                typ = type(header_list[0][key])
                stringy = False
                if typ == str:
                    stringy = True
                    card = np.zeros(N, dtype=object)
                else:
                    card = np.zeros(N, dtype=typ)
                for i in np.arange(N):
                    j = np.where(index_order == i)[0][0]
                    card[i] = header_list[j][key]
                if stringy:
                    # convert to numpy bytes array which is compatible with HDF5 database
                    card = card.astype('S')
                fits_header.append(f'{key} = {card}')
                fits_dict[key] = card
        return fits_header, fits_dict

    @typechecked
    def read_files(self,
                   files: list) -> Tuple[Dict,
                                         tuple, np.ndarray, np.ndarray, np.ndarray]:
        """
        Function which reads all FITS file and appends each data set and header info to the database. The function gets
        a list of *overwriting_tags*. If a new key (header entry or image data) is found that is
        not on this list the old entry is overwritten if *self.m_overwrite* is active. After
        replacing the old entry the key is added to the *overwriting_tags*. This procedure
        guarantees that all previous database information, that does not belong to the new data
        set that is read by FitsReadingModule is replaced and the rest is kept.

        Parameters
        ----------
        files : str
            Absolute path and filename of the FITS files.

        Returns
        -------
        Dict
            header as dictionary
        tuple(int, )
            Image shape.
        list
            wavelength array
        list
            channel array
        list
            band array
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

            # the data is expected to be stored in the primary HDU
            if hdu_list[0].data is not None:
                imgs.append(hdu_list[0].data.byteswap().newbyteorder())
                headers.append(hdu_list[0].header)
            else:
                raise RuntimeError(
                    f"No data was found in {files[i]} (or the collapser argument should be set to True).")

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

        index_order = self.index_order_setter(header_list=headers,
                                              ordering_keyword="JWST")

        # array with the nr of (wavelength) frames in channel i saved at index i
        channel_lengths = np.zeros(N)

        # Create Data Cube with correct order:
        data_cube = np.zeros(N, dtype=object)

        for i in np.arange(N):
            j = np.where(index_order == i)[0][0]
            data_cube[i] = imgs[j]

            channel_lengths[i] = headers[j]["NAXIS3"]

        # Total wavelength dimension size of data cube
        N_wave = channel_lengths.sum()

        wavelength_arr, channel_arr, band_arr = self.wave_chan_bd_arr_generator(index_order=index_order,
                                                                                channel_lengths=channel_lengths,
                                                                                header_list=headers)

        shape = (int(N_wave), int(maxNM[0]), int(maxNM[1]))
        data_cube = self.FillCubesZeros(data_cube, shape, channel_lengths)

        fits_header, fits_dict = self.read_headers(
            header_list=headers, index_order=index_order, N_wave=int(N_wave))

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

        if self.m_overwrite:
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

        self.m_header_out_port = self.add_output_port(
            'fits_header/'+self.m_img_tag)
        header_out_port = self.m_header_out_port
        header_out_port.set_all(fits_header)

        return fits_dict, data_cube.shape, wavelength_arr, channel_arr, band_arr

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

        # files is now a list of directory paths to each FITS file that should be read in

        if self.m_collapse:  # if the header information is saved spread across multiple HDU's it will be collapsed to the primary one
            tempout = os.path.join(self.m_input_location, "temp")
            if os.path.isdir(tempout):
                warnings.warn(
                    f'The temporary folder {tempout} already exists in the input directory.')
                for char in string.printable:  # create an unused temp folder
                    if os.path.isdir(os.path.join(self.m_input_location, "temp"+char)):
                        warnings.warn(
                            f'The temporary folder {os.path.join(self.m_input_location,"temp"+char)} already exists in the input directory.')
                        continue
                    else:
                        tempout = os.path.join(
                            self.m_input_location, "temp"+char)
                        break
            # create a temporary folder for the collapsed data
            os.mkdir(tempout)
            corecollapsesupernova = PrimaryHDUCombiner(name_in="neutronstar",
                                                       input_dir=self.m_input_location,
                                                       output_dir=tempout,
                                                       file_list=files,
                                                       change_name=False,
                                                       overwrite=True,
                                                       only2hdus=self.m_only2hdus)

            corecollapsesupernova.run()
            files = []
            # the filenames are read again in case names were changed
            for filename in os.listdir(tempout):
                if filename.endswith('.fits') and not filename.startswith('._'):
                    files.append(os.path.join(tempout, filename))

            print(
                f'FITS files have been collapsed to PrimaryHDU and saved to {tempout}.')
        files.sort()
        print("Reading all data and headers into Pynpoint.")
        header_dict, shape, wav_arr, chan_arr, band_arr = self.read_files(
            files)

        if self.m_collapse:
            # remove the temporary folder and the collapsed data
            shutil.rmtree(tempout)

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
        set_static_attr(fits_file=files[0],
                        header=header_dict,
                        config_port=self._m_config_port,
                        image_out_port=self.m_image_out_port,
                        check=self.m_check,
                        instrument_key="MIRI")

        set_nonstatic_attr(header=header_dict,
                           config_port=self._m_config_port,
                           image_out_port=self.m_image_out_port,
                           check=self.m_check,
                           instrument_key="MIRI")

        wavandchan = [('WAV_ARR', wav_arr), ('CHAN_ARR',
                                             chan_arr), ('BAND_ARR', band_arr)]
        set_extra_attr(fits_file=files[0],
                       nimages=nimages,
                       config_port=self._m_config_port,
                       image_out_port=self.m_image_out_port,
                       first_index=0,
                       optional_attrs=wavandchan,
                       instrument_key="MIRI")

        self.m_image_out_port.flush()

        self.m_image_out_port.close_port()
