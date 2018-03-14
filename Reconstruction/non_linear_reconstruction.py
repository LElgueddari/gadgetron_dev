import numpy as np
import pisap
from scipy.io import loadmat
from ismrmrdtools import transform
from gadgetron import Gadget
import ismrmrd
import ismrmrd.xsd
from pisap.plugins.mri.reconstruct.reconstruct import sparse_rec_condatvu


class AccumulateAndRecon(Gadget):
    def __init__(self, next_gadget=None):
        Gadget.__init__(self, next_gadget)
        self.myBuffer = None
        self.myCounter = 1
        self.mySeries = 1
        self.header = None
        self.enc = None

    def process_config(self, conf):
        self.header = ismrmrd.xsd.CreateFromDocument(conf)
        self.protocol = str(self.header.measurementInformation.protocolName)
        self.enc = self.header.encoding[0]
        self.NEX = int(self.enc.encodingLimits.average.maximum) + 1
        self.z_size = int(self.enc.encodedSpace.matrixSize.z)
        self.y_size = int(self.enc.encodingLimits.kspace_encoding_step_1.maximum) + 1
        self.x_size = int(self.enc.encodedSpace.matrixSize.x)
        self.image_x = int(self.enc.reconSpace.matrixSize.x)
        self.image_y = int(self.enc.reconSpace.matrixSize.y)
        self.image_z = int(self.enc.reconSpace.matrixSize.z)

    def _reconstruct(self, data):
        samples = loadmat(self.protocol)
        samples = samples[samples.keys()[0]]
        samples /= 2*np.max(np.abs(samples).flatten())
        max_iter = 150
        image_shape = (512, 512)
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
            mu=0.0,
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
