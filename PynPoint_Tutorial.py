"""
PynPoint Example Tutorial
"""

import os
import sys
import urllib
import matplotlib.pyplot as plt

from pynpoint import Pypeline, Hdf5ReadingModule, PSFpreparationModule, PcaPsfSubtractionModule


urllib.request.urlretrieve('https://home.strw.leidenuniv.nl/~stolker/pynpoint/betapic_naco_mp.hdf5',
                           './betapic_naco_mp.hdf5')

pipeline = Pypeline(working_place_in='./',
                    input_place_in='./',
                    output_place_in='./')

module = Hdf5ReadingModule(name_in='read',
                           input_filename='betapic_naco_mp.hdf5',
                           input_dir=None,
                           tag_dictionary={'stack': 'stack'})

pipeline.add_module(module)

module = PSFpreparationModule(name_in='prep',
                              image_in_tag='stack',
                              image_out_tag='prep',
                              mask_out_tag=None,
                              norm=False,
                              resize=None,
                              cent_size=0.15,
                              edge_size=1.1)

pipeline.add_module(module)

module = PcaPsfSubtractionModule(pca_numbers=[20, ],
                                 name_in='pca',
                                 images_in_tag='prep',
                                 reference_in_tag='prep',
                                 res_median_tag='residuals')

pipeline.add_module(module)

pipeline.run()