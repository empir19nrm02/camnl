# -*- coding: utf-8 -*-
"""
stand-alone version for LMK access

See  /TechnoTeam/LabSoft/doc/activexserver/class_l_m_k_ax_server.html

@author: Christian Schrader
@email: christian.schrader@ptb.de
"""

import win32com.client
import numpy as np

class LMK_Error(Exception):
    def __init__(self, code, message, function=""):
        self.code = code
        self.message = message
        self.function = function

class LMK(object):
    _handle = None
    
    def __init__(self):
        self._handle = win32com.client.Dispatch("lmk4.LMKAxServer.1")

    def _cmd(self, func_name, func, args):
        res = None
        err = None
        err_res = func(*args)
        if type(err_res) == int:
            res=None
            err=err_res
        else:
            err=err_res[0]
            res=err_res[1:]
            if len(res) == 1:
                res=res[0]
        if err==0 or err==None:
            return res
        else:
            msg=self._handle.iGetErrorInformation()[1]
            raise LMK_Error(err, msg, function=func_name)

    def SetNewCamera2(self, _qNameCamera, _qNameLens):
        '''
        Set new camera calibration data.

        Parameters
        ______
        _qNameCamera	Name of camera, for example "dxm2134"
        _qNameLens	Name of lens, for example "oC202517f25"

        If both strings are empty, a currently existing
        camera connection is finished.

        Returns
        ______
        0=ok, !=0 Error code
        '''
        func_name = 'iSetNewCamera2'
        func = self._handle.iSetNewCamera2
        args = [_qNameCamera, _qNameLens]
        return self._cmd(func_name, func, args)

    def Camera_GetParameter(self, _qname):
        '''
        Read some additional camera properties.\n
        Parameter
        ________
        _qName: Name of the parameter wished. Currently are supported
        by some camera types:\n
        \t Gain: Gain value of the camera chip\n
        \t Temperature: Temperature of the camera chip\n
        \t VARI_SPEC_TEMPERATURE: Temperature of LCDF\n
        \t VARI_SPEC_WAVELENGTH: Wavelength of LCDF\n
        Retur
        _______
        _qrValue Return value of the function. The interpretation of
        the string depends on the parameter itself. For example the gain
        is an integer value, the temperature is a floating point.

        '''
        func_name = 'iCamera_GetParameter'
        func = self._handle.iCamera_GetParameter
        args = [_qname]
        return self._cmd(func_name, func, args)

    def GetIntegrationTime(self):
        '''
        Return
        ________
        double &_drCurrentTime,
        double &_drPreviousTime,
        double &_drNextTime,
        double &_drMinTime,
        double &_drMaxTime
        '''
        func_name = 'iGetIntegrationTime'
        func = self._handle.iGetIntegrationTime
        args = []
        return self._cmd(func_name, func, args)

    def AutoScanTime(self):
        '''
        autoset exposure time
        '''
        func_name = 'iAutoScanTime'
        func = self._handle.iAutoScanTime
        args = []
        return self._cmd(func_name, func, args)

    def SetIntegrationTime(self, _dWishedTime):
        '''
        Set new exposure time.

        Parameters
        __________

        _dWishedTime	Wished integration time

        _drRealizedTime	Realized integration time
        '''
        func_name = 'iSetIntegrationTime'
        func = self._handle.iSetIntegrationTime
        args = [_dWishedTime]
        return self._cmd(func_name, func, args)

    def Camera_GetSaturation(self, _iWholeImage=1, _iTop=0, _iLeft=0,
                             _iBottom=0, _iRight=0, _drSaturation=0):
        '''
        Parameters
        __________
        _iWholeImage	- 1: Use whole image (_iTop, _iLeft,
        _iBottom and _iRight are meaningless)
        0: Use given region coordinates
        _iTop	First line of region
        _iLeft	First column of region
        _iBottom	Last line of region
        _iRight	Last column of region
        Return
        ____
        _drSaturation	Saturation of image or region in percent
        (values between 0.0 and 100.0, 100.0 = overdriven image or region)
        '''
        func_name = 'iCamera_GetSaturation'
        func = self._handle.iCamera_GetSaturation
        args = [_iWholeImage, _iTop, _iLeft, _iBottom, _iRight]
        return self._cmd(func_name, func, *args)

    def Grab(self):
        '''
        Capturing a camera image
        '''
        func_name = 'iGrab'
        func = self._handle.iGrab
        args = []
        return self._cmd(func_name, func, args)

    def SinglePic2(self, _dExposureTime):
        '''
        Capturing a luminance image with SinglePic algorithm

        Parameters
        ____
        dExposureTime Exposure time to use
        '''
        func_name = 'iSinglePic2'
        func = self._handle.iSinglePic2
        args = [_dExposureTime]
        return self._cmd(func_name, func, args)

    def SaveImage(self, _iNumber, _qFileName):
        '''
        Save Image

        Parameters
        _____

        _iNumber index of image to save
        -3	 Camera image
        -2	 Luminance image
        -1 Color image
        0 or larger eval image
        _qFileName Destination file name
        '''
        func_name = 'iSaveImage'
        func = self._handle.iSaveImage
        args = [_iNumber, _qFileName]
        return self._cmd(func_name, func, args)

    def Open(self):
        '''
        öffnet LabSoft
        '''
        func_name = 'iOpen'
        func = self._handle.iOpen
        args = []
        return self._cmd(func_name, func, args)

    def Close(self, _iQuestion=0):
        '''
        Closes the Lmk4 application.

        Parameters
        ____
        _iQuestion	!=0: Opens a dialogue window in the application.
        The user can choose, wether he wish to save the current state
        or not or cancel th closing of the program.\n
        =0: No dialogue window

        Returns
        ____
        0=ok, 1=User do not want to close the application, >1 Error code

        '''
        func_name = 'iClose'
        func = self._handle.iClose
        args = [_iQuestion]

        return self._cmd(func_name, func, args)

    def ImageGetDumpToMemory(self, nr, rowstart, rowend, colstart, colend):
        # very slow! Better use SaveImage() to temporary file and 
        # open it with lmk_image
        func_name = 'iImageGetDumpToMemory'
        func = self._handle.iImageGetDumpToMemory
        args = [nr, rowstart, rowend, colstart, colend]
        # FIXME: color images?
        buff =  self._cmd(func_name, func, args)
        img = np.frombuffer(buff, dtype=np.float32).reshape(rowend-rowstart+1,colend-colstart+1)
        return img

    def SetFilterWheel(self, nr):
        func_name = 'iSetFilterWheel'
        func = self._handle.iSetFilterWheel
        args = [nr]
        return self._cmd(func_name, func, args)
