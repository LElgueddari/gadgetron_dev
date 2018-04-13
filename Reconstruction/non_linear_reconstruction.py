# -*- coding: utf-8 -*-
##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Non-cartesian reconstruction for Sparkling Trajectorie.
"""

# Package import
import ismrmrd
import numpy as np
from scipy.io import loadmat
from gadgetron import Gadget
from pysap.plugins.mri.reconstruct.reconstruct import sparse_rec_condatvu


class AccumulateAndRecon(Gadget):
    """ Accumulator and reconstrcutor class.

    Attributes
    ----------
    myBuffer: np.ndarray
        The Buffer that accumulate the acquired data
    myCounter: int
        Counts the number of images
    mySeries: int
        Counts the number of Series (5 images = 1 series)
    header: ismrmrd.xsd.ismrmrdHeader Object
        Gives all the metadata of the acquisition
    enc: ismrmrd.xsd.encoding
        Gives all the information of the encoding parameters used
    protocol: string
        The name of the protocol (Gradient File), this name will be used to
        load the k_space sample position
    NEX: int
        Number of excitation (to average the signal)
    z_size: int
        The number of slices
    y_size: int
        The number of spokes used to acquire the NMR signal
    x_size: int
        The number of samples per spokes
    image_x: int
        The 1 dimension of the output image size
    image_y: int
        The 2 dimension of the output image size
    image_z: int
        The 3 dimension of the output image size
    """
    def __init__(self, next_gadget=None):
        """ Initilize the 'AccumulateAndRecon' class.

        Parameters
        ----------
        next_gadget: Gadget
            The next Gadget in the pipeline process
        """
        Gadget.__init__(self, next_gadget)
        self.myBuffer = None
        self.myCounter = 1
        self.mySeries = 1
        self.header = None
        self.enc = None

    def process_config(self, conf):
        """ This method configures the attributes given the xml header.

        Parameters
        ----------
        conf: string
            The description of the xml header
        """
        self.header = ismrmrd.xsd.CreateFromDocument(conf)
        self.protocol = str(self.header.measurementInformation.protocolName)
        self.enc = self.header.encoding[0]
        self.NEX = int(self.enc.encodingLimits.average.maximum) + 1
        self.z_size = int(self.enc.encodedSpace.matrixSize.z)
        self.y_size = int(
                self.enc.encodingLimits.kspace_encoding_step_1.maximum) + 1
        self.x_size = int(self.enc.encodedSpace.matrixSize.x)
        self.image_x = int(
                    self.header.userParameters.userParameterLong[2].value_)
        self.image_y = self.image_x
        for _ in range(15):
            self.image_x
        self.image_z = 1  # int(self.enc.reconSpace.matrixSize.z)

    def _reconstruct(self, data):
        """ This method reconstruct the final image given the kspace data.

        Parameters
        ----------
        data: np.ndarray
            The kspace value in the Buffer

        Returns
        -------
        x_final: np.ndarray
            The solution of the iterative reconstruction algorithm
        """
        samples = loadmat(self.protocol)
        samples = samples[samples.keys()[0]]
        samples /= 2*np.max(np.abs(samples).flatten())
        max_iter = 500
        if self.image_z == 1:
            image_shape = (self.image_x, self.image_y)
        else:
            image_shape = (self.image_x, self.image_y, self.image_z)
        data = np.squeeze(data.astype('complex128'))
        data = np.mean(data, axis=len(data.shape)-1)
        data = np.transpose(data).flatten()
        x_final, _ = sparse_rec_condatvu(
            data=data,
            wavelet_name="UndecimatedDiadicWaveletTransform",
            samples=samples,
            nb_scales=4,
            std_est=None,
            std_est_method=None,
            std_thr=2.,
            mu=1e-5,
            tau=None,
            sigma=None,
            relaxation_factor=1.0,
            nb_of_reweights=None,
            max_nb_of_iter=max_iter,
            add_positivity=False,
            atol=1e-24,
            non_cartesian=True,
            uniform_data_shape=image_shape,
            verbose=1)
        return x_final

    def process(self, acq, data, *args):
        """
        Parameters
        ----------
        acq:
        data: np.ndarray
            The kspace value in the Buffer

        Returns
        -------

        """
        if self.myBuffer is None:
            channels = acq.active_channels
            if self.enc.encodingLimits.slice is not None:
                nslices = self.enc.encodingLimits.slice.maximum + 1
            else:
                nslices = 1

            self.myBuffer = np.zeros((self.x_size, self.y_size, self.z_size,
                                      nslices, channels, self.NEX),
                                     dtype=np.complex64)

        line_offset = self.enc.encodedSpace.matrixSize.y/2 - \
            self.enc.encodingLimits.kspace_encoding_step_1.center
        self.myBuffer[:,
                      acq.idx.kspace_encode_step_1 + line_offset,
                      acq.idx.kspace_encode_step_2,
                      acq.idx.slice,
                      :,
                      acq.idx.average] = data

        if (acq.flags & (1 << 7)):  # Is this the last scan in slice
            image = self._reconstruct(self.myBuffer)
            # Scaling for the scanner
            image = image * np.product(image.shape)*100
            # Create a new image header and transfer value
            img_head = ismrmrd.ImageHeader()
            img_head.channels = acq.active_channels
            img_head.slice = acq.idx.slice
            img_head.matrix_size[0] = self.image_x
            img_head.matrix_size[1] = self.image_y
            img_head.matrix_size[2] = self.image_z
            img_head.position = acq.position
            img_head.read_dir = acq.read_dir
            img_head.phase_dir = acq.phase_dir
            img_head.slice_dir = acq.slice_dir
            img_head.patient_table_position = acq.patient_table_position
            img_head.acquisition_time_stamp = acq.acquisition_time_stamp
            img_head.image_index = self.myCounter
            img_head.image_series_index = self.mySeries
            img_head.data_type = ismrmrd.DATATYPE_CXFLOAT
            self.myCounter += 1
            if self.myCounter > 5:
                    self.mySeries += 1
                    self.myCounter = 1

            # Return image to Gadgetron
            self.put_next(img_head, image.astype('complex64'), *args)

        # print "Returning to Gadgetron"
        return 0  # Everything OK
