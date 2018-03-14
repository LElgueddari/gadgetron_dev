import os
import ismrmrd
import ismrmrd.xsd
import numpy as np
from ismrmrdtools import show
from non_linear_reconstruction import AccumulateAndRecon


gadget = AccumulateAndRecon()

# load file

filename = 'meas_MID151_CSGRE_std_OS2_sparkling_nc43_BR1536_FID12233.h5'
if not os.path.isfile(filename):
    print("%s is not a valid file" % filename)
    raise SystemExit
dset = ismrmrd.Dataset(filename, 'dataset', create_if_needed=False)

gadget.process_config(dset.read_xml_header())

for acqnum in range(0,dset.number_of_acquisitions()):
    acq = dset.read_acquisition(acqnum)
    # print(acq.getHead().idx.kspace_encode_step_1)
    # print(acq.getHead().idx.average)
    gadget.process(acq.getHead(),np.transpose(acq.data))

#Get result and display
res = gadget.get_results()
show.imshow(np.squeeze(abs(res[0][1])))
