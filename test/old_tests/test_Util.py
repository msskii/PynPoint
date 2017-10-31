# Copyright (C) 2014 ETH Zurich, Institute for Astronomy
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/.


"""
Tests for `Util` module.
"""
#from __future__ import print_function, division, absolute_import, unicode_literals
import os
import numpy as np

import PynPoint


limit1 = 1.0e-10
limit2 = 1.0e-7

class TestUtil(object):

    def setup(self):
        print("setting up " + __name__)
        self.num_files = 4
        self.test_data = (os.path.dirname(__file__))+'/test_data/'
        self.file_images_hdf5 = self.test_data + 'test_data_images_v001.hdf5'
        self.file_basis_hdf5 = self.test_data + 'test_data_basis_v001.hdf5'
        self.files = [self.test_data+'Cube_000_Frame_0002_zoom_2.0.fits_shift.fits_planet.fits',
        self.test_data+'Cube_001_Frame_0130_zoom_2.0.fits_shift.fits_planet.fits',
        self.test_data+'Cube_000_Frame_0166_zoom_2.0.fits_shift.fits_planet.fits',
        self.test_data+'Cube_003_Frame_0160_zoom_2.0.fits_shift.fits_planet.fits']
        self.para_test = np.array([-17.3261 , -17.0143, -17.172, -16.6004])
        
        self.eg_array1 = np.arange(100.).reshape(4,5,5)
        self.ave_eg_array1 = np.array([[ 37.5,  38.5,  39.5,  40.5,  41.5],
                        [ 42.5,  43.5,  44.5,  45.5,  46.5],
                        [ 47.5,  48.5,  49.5,  50.5,  51.5],
                        [ 52.5,  53.5,  54.5,  55.5,  56.5],
                        [ 57.5,  58.5,  59.5,  60.5,  61.5]])
        
        pass

        
    def test_print_attributes(self):
        PynPoint._Util.print_attributes(self)
        pass
    
    
    def test_rd_fits1(self):
        PynPoint._Util.rd_fits(self)#,avesub=False,para_sort=True,inner_pix=False)        
        assert len(self.para) == 4
        assert np.allclose(self.para, self.para_test,rtol=limit1)
        assert len(self.im_arr.shape) == 3
        assert self.im_arr.shape[0] == 4
        assert self.im_arr.shape[1] == 146
        assert self.im_arr.shape[2] == 146

        assert self.im_arr[0,100,100] == 136.0
        assert self.im_arr[1,80,70] == 1418.280029296875
        assert self.im_arr[2,40,10] == 26.0
        assert self.im_arr[3,110,110] == 53.319999694824219


    def test_rd_fits2(self):
        PynPoint._Util.rd_fits(self)#,avesub=True,para_sort=False,inner_pix=False)        
        assert len(self.para) == 4
        assert self.para[0] == self.para_test[0]
        assert self.para[1] == self.para_test[1]
        assert self.para[2] == self.para_test[2]
        assert self.para[3] == self.para_test[3]

        assert len(self.im_arr.shape) == 3
        assert self.im_arr.shape[0] == 4
        assert self.im_arr.shape[1] == 146
        assert self.im_arr.shape[2] == 146
        assert self.im_arr[0,100,100] == 136.0
        assert self.im_arr[2,80,70] == 1649.0
        assert self.im_arr[1,40,10] == 28.639999389648438
        assert self.im_arr[3,110,110] == 53.319999694824219
        
                
    def test_gaussian(self):
        gauss = PynPoint._Util.gaussian(1., 100., 100., 20., 10.)
        assert gauss(100.,90.).size == 1
        assert gauss(100.,90.) == np.exp(-0.5)        
        assert gauss(120.,100.) == np.exp(-0.5)        
            
    def test_mk_circle(self):
        circ = PynPoint._Util.mk_circle(10.,20.)
        assert circ(10.,30.).size == 1
        assert circ(10.,30.) == 10.  
        assert circ(20.,30.) == np.sqrt(2)*10.  
        assert circ(20.,20.) == 10.  
        
        
    def test_moments(self):
        test_data = np.array([[1.,1.],[1.,1.]])
        temp = PynPoint._Util.moments(test_data)
        assert len(temp) == 2
        assert temp[0] == 0.5
        assert temp[1] == 0.5
        
        
    def test_mk_gauss2D(self):
        gauss2d = PynPoint._Util.mk_gauss2D(201,101,20.,xcent=None,ycent=None)
        assert gauss2d.shape[0] == 201
        assert gauss2d.shape[1] == 101
        assert gauss2d[100,50] == 1.0
        assert gauss2d[80,50] == np.exp(-0.5)
        assert gauss2d[100,70] == np.exp(-0.5)
        

    def test_gausscent(self):
        test_data = PynPoint._Util.mk_gauss2D(101,101,20.,xcent=None,ycent=None)
        temp = PynPoint._Util.gausscent(test_data,gauss_width=20.,itnum=5)
        assert len(temp) == 2
        assert np.allclose(temp[0],50.0,rtol=limit1)
        assert np.allclose(temp[1],50.0,rtol=limit1)
               
    def test_mk_resize(self):
        test_data = PynPoint._Util.mk_gauss2D(101,101,20.,xcent=None,ycent=None)
        temp = PynPoint._Util.mk_resize(test_data,303,303)
        assert temp.shape[0] == 303
        assert temp.shape[1] == 303
        assert np.allclose(temp[151,151], 1.0,rtol=1e-3)
        assert np.allclose(temp[151,211], 0.61054736022966827,rtol=1e-5)
        assert np.allclose(temp[91,151], 0.61054736022966827,rtol=1e-5)
        
    def test_mk_recent(self):
        test_data = PynPoint._Util.mk_gauss2D(101,101,20.,xcent=None,ycent=None)
        temp = PynPoint._Util.mk_recent(test_data,10,20)
        assert np.allclose(temp[40:101,30:91], test_data[20:81,20:81],rtol=limit1)
        
    def test_mk_rotate(self):
        delta = 100/np.sqrt(2.)
        test_data1 = PynPoint._Util.mk_gauss2D(401,401,30.,xcent=100,ycent=200) + PynPoint._Util.mk_gauss2D(401,401,30.,xcent=200,ycent=100)
        test_data2 = PynPoint._Util.mk_gauss2D(401,401,30.,xcent=200,ycent=100) + PynPoint._Util.mk_gauss2D(401,401,30.,xcent=300,ycent=200)
        test_data3 = PynPoint._Util.mk_gauss2D(401,401,30.,xcent=200-delta,ycent=200-delta) + PynPoint._Util.mk_gauss2D(401,401,30.,xcent=200+delta,ycent=200-delta)
        
        rot1 = PynPoint._Util.mk_rotate(test_data1,90.)
        rot2 = PynPoint._Util.mk_rotate(test_data1,45.)
        print((rot2- test_data3).max())
        print((rot2- test_data3).min())
        
        
        assert np.allclose(test_data2, rot1,rtol=limit1)
        assert np.allclose(test_data3, rot2,atol=4.e-3)
        
    def test_mk_resizerecent(self):
        test_data = PynPoint._Util.mk_gauss2D(101,101,20.,xcent=45,ycent=48)
        test_data2 = PynPoint._Util.mk_gauss2D(404,404,80.,xcent=None,ycent=None)
        im_arr = np.array([test_data,test_data])
        im_arr2 = PynPoint._Util.mk_resizerecent(im_arr,2.,4.)
        assert np.allclose(test_data2[50:350,50:350],im_arr2[0,50:350,50:350],atol = 7e-3) 
        assert np.allclose(im_arr2[1,50:350,50:350],im_arr2[0,50:350,50:350],rtol = limit1)
        
    def test_mk_resizeonly(self):
        test_data = PynPoint._Util.mk_gauss2D(101,101,20.,xcent=None,ycent=None)
        test_data2 = PynPoint._Util.mk_gauss2D(404,404,80.,xcent=None,ycent=None)
        im_arr = np.array([test_data,test_data])
        im_arr2 = PynPoint._Util.mk_resizeonly(im_arr,4.)
        assert np.allclose(test_data2,im_arr2[0,],atol = 6e-3) 
        assert np.allclose(im_arr2[0,],im_arr2[1,],rtol = limit1)
        
    def test_mk_avesub(self):

        im_arr,im_ave = PynPoint._Util.mk_avesub(self.eg_array1)

        assert np.array_equal(im_ave,self.ave_eg_array1)
        im_arr2 = im_arr.copy()
        for i in range(0,len(im_arr[:,0,0])):
            im_arr2[i,] += im_ave
        assert np.array_equal(self.eg_array1,im_arr2)#im_ave,self.ave_eg_array1)
        
    def test_file_list(self):
        files_temp = PynPoint._Util.file_list(self.test_data,ran_sub=False)
        print('-------')
        print(files_temp)
        print('-------')
        print(self.files)
        for i in range(0,4):
            assert files_temp[i] in self.files

    def test_check_type(self):
        print(self.file_images_hdf5)
        print(self.file_basis_hdf5)
        
        im_type = PynPoint._Util.check_type(self.file_images_hdf5)
        basis_type = PynPoint._Util.check_type(self.file_basis_hdf5)
        assert(im_type == 'PynPoint_images')
        assert(basis_type == 'PynPoint_basis')

    def test_filename4mdir(self):
        temp1 = PynPoint._Util.filename4mdir(self.test_data,filetype='images')
        temp2 = PynPoint._Util.filename4mdir(self.test_data,filetype='junk')
        assert(temp1[-21:] == '_PynPoint_images.hdf5') 
        assert(temp2[-19:] == '_PynPoint_temp.hdf5') 

    def test_conv_dirfits2hdf5(self):
        PynPoint._Util.conv_dirfits2hdf5(self.test_data,outputfile = None)
        

    def test_peak_find(self):
        res_null = PynPoint._Util.dummyclass()#('','','',intype='empty')
        dim = [400,600]
        cents = np.array([[100,200],[50,350],[215,79]])
        test_im = np.zeros(shape=dim)
        for i in range(0,cents.shape[0]): 
            test_im += PynPoint._Util.mk_gauss2D(dim[0],dim[1],30.,xcent=cents[i,0],ycent=cents[i,1]) 
        res_null.res_clean_mean_clip = test_im
        x_peaks,y_peaks, h_peaks,sig,num_peaks = PynPoint._Util.peak_find(test_im,limit=0.8,printit=True)
        
        x = np.array(x_peaks)
        x.sort()
        x_cents = cents[:,1].copy()
        x_cents.sort()

        y = np.array(y_peaks)
        y.sort()
        y_cents = cents[:,0].copy()
        y_cents.sort()


    def test_str2bool(self):
        for s in ("yes", "true", "t", "1", "Yes", "True", "T"):
            assert PynPoint._Util.str2bool(s)
            
        for s in ("no", "false", "f", "0", "No", "False", "F" ):
            assert not PynPoint._Util.str2bool(s)
        

    def teardown(self):
        '''
        #tidy up
        print("tearing down " + __name__)
        tempfilename = PynPoint._Util.filename4mdir(self.test_data,filetype='convert')
        if os.path.isfile(tempfilename):
            os.remove(tempfilename)
        pass'''
