import ismrmrd
import numpy as np


# def change_header_file(input_filename):
#     dset = ismrmrd.Dataset(input_filename, 'dataset', create_if_needed=False)
#     header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
#     enc = header.encoding[0]
#     enc.reconSpace.matrixSize.x = 512L
#     enc.reconSpace.matrixSize.y = 512L
#     print(header.encoding[0].reconSpace.matrixSize.y)


def get_ADC_samples_coordinates(gradient_path, nb_ADC_samples, dwelltime=5000,
                                ndim=2, verbose=0):
    """This function extracts the 2D and 3D sampling scheme from the gradient
        file, using the	dwelltime and the number of samples.

        Parameters
        ----------
        gradient_path: string
            Path to the gradient.txt file that contains the points of the
            gradient
        nb_ADC_samples: int
            Number of points taken for this acquisition
        dwelltime: float
            dwell time extracted from the header of the acquisition, in
            nanoseconds (default 5000µs)
        ndim: int
            Number of dimension of the gradient 2D or 3D acquisition
        verbose: int
            Verbosity level (default 0: silent)
        """
    # Load gardient file
    file_header = np.loadtxt(gradient_path, delimiter='\n', comments='\t')
    spokes_number = int(file_header[0])
    samples_per_spoke = int(file_header[1])
    file_content = np.loadtxt(gradient_path, delimiter='\t', skiprows=2,
                              ndmin=0)
    k0 = file_content[:spokes_number]
    grad = file_content[spokes_number:]

    # Over-sampling factor
    OS_factor = int(nb_ADC_samples*1.0 / (samples_per_spoke*spokes_number))
    real_samples_nb_per_spoke = OS_factor*samples_per_spoke

    # Compute gradient constants
    gradient_duration = samples_per_spoke * 0.01
    gamma = 42.576*1e6
    dt = 10e-6
    dt_ns = dt * 1e9  # Gradient time step in nanoseconds
    if verbose > 0:
        print('Number of Spokes                      ', spokes_number)
        print('Number of samples / Spokes            ', samples_per_spoke)
        print('Dimension                             ', k0.shape[-1])
        print('Number of samples / Spokes in the ADC ',
              real_samples_nb_per_spoke)
        print('Gradient duration was                 ', gradient_duration)
        print('Gradient shape                        ', grad.shape)

    grad = grad*1e-3   # Conversion from mT/m to T/m

    # Start calcul of the ADC samples coordinates

    ADC_samples = []
    for k in range(spokes_number):
        gradient = grad[k*samples_per_spoke:(k+1)*samples_per_spoke, :]
        ADC_samples_k = np.zeros((real_samples_nb_per_spoke+1, ndim))
        ADC_samples_k[0] = k0[k]
        cnt = 1
        for j in range(1, nb_ADC_samples):
            ADC_time = dwelltime * j
            q = int(np.floor(ADC_time/dt_ns))
            r = ADC_time - (q)*dt_ns*1.0
            cnt = 1 + cnt
            if q < samples_per_spoke:
                gradient_to_sum = gradient[:q]
                ADC_samples_k[j] = ADC_samples_k[0] + (np.sum(
                    gradient_to_sum,
                    axis=0) * dt_ns + gradient[q, :] * r) * gamma * 1e-9
            elif q == samples_per_spoke and (r == 0):
                gradient_to_sum = gradient[:q, :]
                ADC_samples_k[j, :] = ADC_samples_k[0, :] + (np.sum(
                    gradient_to_sum,
                    axis=0) * dt_ns) * gamma * 1e-9
            else:
                ADC_samples_k = ADC_samples_k[:cnt - 1, :]
                break
        ADC_samples.append(ADC_samples_k)

    return ADC_samples


def extract_dwell_time_from_datfile(filename):
    import csv
    import re

    line_idx = 0
    line_dwelltime = -1

    dwelltime_matrices = []
    found = False
    with open(filename, "rb") as myfile, open('x.txt', 'w', newline='') as fw:
        writer = csv.DictWriter(fw, fields, delimiter='|')
        record = {}
        for line in myfile:
            line = str(line)

            if line_idx > 94967:
                line = str(line)
                result_k = line.find('ParamLong."alDwellTime"')
                if result_k >= 0:
                    line_dwelltime = line_idx
                    result_k_1 = result_k
                    print(result_k, line, line_idx)

                if line_dwelltime > 0:
                    if line.find('}') >= 0:
                        found = True
                    dwelltime_matrices.append(line.strip("'b"))
                    print(dwelltime_matrices[-1])
            if found:
                break

            line_idx += 1
