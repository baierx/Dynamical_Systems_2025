def read_patient_data(file_name, number_of_seizures, path, ):
    
    all_segs_fromfile = loadtxt(file_name)
    
    data_all_segs_read = asarray(all_segs_fromfile)
    
    dim_r1, dim_r2 = data_all_segs_read.shape

    print(dim_r1, dim_r2, number_of_seizures)
    
    data_all_segments = data_all_segs_read.reshape(number_of_seizures, dim_r1, dim_r2//number_of_seizures)
    
    label_dict_chars =  path + 'All_onsets_label_dict.txt'
    
    with open(label_dict_chars, "r") as dict_file:
    
        labels_dict_read = json.load(dict_file)
    
    labels_chars =  path + 'All_onsets_labels.txt'
    
    all_labels_list = []
    
    with open(labels_chars, 'r') as file:
            
        for line in file:
    
            label = line[:-1]
            all_labels_list.append(label)

    return labels_dict_read, data_all_segments, all_labels_list



def eeg_plot_by_electrode(data, offset, labels_dict, normalise=True):
    """
    Plot date columns in EEG style
    data:      two-dimensional array
    offset:    scaling factor
    normalise: normalisation of amplitudes to variance 1
    """
    from matplotlib.pyplot import subplots, yticks, axis
    from numpy import zeros, linspace, arange
    
    start = 0
    samples    = data.shape[0]
    electrodes = data.shape[1]

    dataset = data[start:start+samples, :electrodes]
    means   = data[start:start+samples, :electrodes].mean(axis=0)
    devs    = data[start:start+samples, :electrodes].std(axis=0)


    from matplotlib.colors import BASE_COLORS
    
    color_array = list()
    
    farben = list(BASE_COLORS.keys())*3

    farben = [farb for farb in farben if farb != 'w']

    
    for index, key in enumerate(labels_dict):
    
        for index2 in arange(labels_dict[key][1] - labels_dict[key][0]):
        
            color_array.append(farben[index])
            
    
    fig, ax = subplots(figsize=(8, 8))

    if not normalise:
        ax.set_prop_cycle(color=color_array)
        ax.plot((dataset - means)      + offset*arange(electrodes-1,-1,-1), linewidth=1);
        
    else:
        ax.set_prop_cycle(color=color_array)
        ax.plot((dataset - means)/devs + offset*arange(electrodes-1,-1,-1), linewidth=1);
    
    ax.plot(zeros((samples, electrodes)) + offset*arange(electrodes-1,-1,-1),'--',color='gray');
    ax.set(ylabel='Voltage')

    yticks([]);

    axis('tight')

    return fig, ax


def plot_electrodes(data_all_segments, labels_dict_read, all_labels_list,
                    elec, segment, freq_bands, sr,
                    dims, order, plot_dim, onset_time, start, stop,
                    offset, vmin, vmax):

    no_of_bands = len(freq_bands)

    rows_seg = stop - start
    
    data_elec = data_all_segments[segment, :, labels_dict_read[elec][0]:labels_dict_read[elec][1]]
    
    elec_chans = labels_dict_read[elec][1] - labels_dict_read[elec][0]
    
    elec_label_names = all_labels_list[labels_dict_read[elec][0]:labels_dict_read[elec][1]]
    
    
    data_elec_embedded = zeros((data_elec[:, :].shape[0], data_elec[:, :].shape[1], dims))
        
    data_elec_embedded[:, :, 0] = data_elec[:, :]
    
    for dim in arange(dims-1):
    
        data_elec_embedded[:, :, dim+1] = gradient(data_elec_embedded[:, :, dim], axis=0)
    
    
    data_elec_bands_embedded = zeros((len(freq_bands), data_elec.shape[0], data_elec.shape[1], dims))
        
    for dim in arange(dims):
    
        for index, band in enumerate(freq_bands):
            
            data_elec_bands_embedded[index, :, :, dim] = data_band_pass_filter_2d(data_elec_embedded[ :, :, dim], 
                                                      band[0], band[1], order, sr)
    

    ###################################################
    fig, ax = subplots(nrows=no_of_bands, ncols=2, figsize=(8, 8))
        
    for index in arange(no_of_bands):
            
        ax[index, 0].plot(data_elec_bands_embedded[index, start:stop, :, plot_dim] + offset*arange(elec_chans-1,-1,-1), 
                linewidth=1, color='b');
    
        ax[index, 0].set_yticks(offset*arange(elec_chans))
        ax[index, 0].set_yticklabels(flip(arange(elec_chans)), fontsize=8)
        ax[index, 0].margins(x=0)
        ax[index, 0].set_xticks([])
        ax[index, 0].set_yticks(offset*arange(len(elec_label_names)));
        ax[index, 0].set_yticklabels(flip(elec_label_names), fontsize=6);
    
    
        if onset_time*sr > start and onset_time*sr < stop:
            
            ax[index, 0].vlines(onset_time*sr-start, -0.5*offset, offset*elec_chans-.5, color="k", linewidth=1);
    
    
        ax[index, 1].imshow(data_elec_bands_embedded[index, start:stop, :, plot_dim].T, aspect='auto', 
                  cmap='bwr', vmin=vmin/((index+1)**2), vmax=vmax/(((index+1)**2)));
        ax[index, 1].set_xticks(linspace(0, rows_seg-1, 3))
        ax[index, 1].set_xticklabels([])
        ax[index, 1].set_yticks(arange(len(elec_label_names)));
        ax[index, 1].set_yticklabels([]);
    
    
        ax2 = ax[index, 1].twinx() 
    
        ax2.tick_params(axis='y')
        ax2.set_yticks(arange(len(elec_label_names)));
        ax2.set_yticklabels(flip(elec_label_names[:]), fontsize=6);
    
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    
        textstr = str(freq_bands[index][0]) + ' - ' + str(freq_bands[index][1])
        
        ax[index, 1].text(-.16, 0.95, textstr, transform=ax[index, 1].transAxes, fontsize=6,
        verticalalignment='top', bbox=props)
    
        if onset_time*sr > start and onset_time*sr < stop:
            
            ax[index, 1].vlines(onset_time*sr-start, -0.5, elec_chans-.5, color="k", linewidth=1);
    
    
    ax[index, 0].set_xticks(linspace(0, rows_seg-1, 3))
    elec_labl = around(linspace(start/sr, stop/sr, 3), 2)
    ax[index, 0].set_xticklabels(elec_labl, fontsize=10)
    
    ax[index, 1].set_xticks(linspace(0, rows_seg-1, 3))
    x_labl = around(linspace(start/sr, stop/sr, 3), 2)
    ax[index, 1].set_xticklabels(x_labl, fontsize=10)
    ax[index, 1].set_xlabel('Time (seconds)', fontsize=10);

    return fig, ax



def data_band_pass_filter_2d(data_2d, cut_low, cut_high, order, sr):


    from scipy.signal import butter, sosfiltfilt
    from numpy import arange, zeros, asarray

    sos = butter(order, (cut_low, cut_high), btype='bandpass', fs=sr, output='sos')

    data_filtered = sosfiltfilt(sos, data_2d, padlen=100, axis=0)
        
    return data_filtered


def plot_hippocampus_channels(data_elecs_band, elecs_label_names, start, stop, sr, segment, offset, onset_time, freq_band):

    from numpy import linspace, arange
    
    chans = data_elecs_band.shape[1]

    rows_seg = data_elecs_band.shape[0]

    # ytick_pos = chans // len(elecs_label_names)

###################################################
    fig, ax = subplots(ncols=2, figsize=(6, 4))
                
    ax[0].plot(data_elecs_band + offset*arange(chans-1,-1,-1), 
            linewidth=1, color='b');
    
    ax[0].margins(x=0)
    # ax[0].set_yticks(1.1*offset*arange(len(elecs_label_names)));
    # ax[0].set_yticklabels(flip(elecs_label_names), fontsize=6);
    ax[0].set_xticks(linspace(0, rows_seg-1, 3))
    x_labl = around(linspace(start/sr, stop/sr, 3), 2)
    ax[0].set_xticklabels(x_labl, fontsize=10)
    ax[0].set_xlabel('Time (seconds)', fontsize=10);

    
    if onset_time*sr > start and onset_time*sr < stop:
        
        ax[0].vlines((onset_time*sr-start), -0.5*offset, offset*chans-.5, color="k", linewidth=1);
    
    
    ax[1].imshow(data_elecs_band.T, aspect='auto', 
              cmap='bwr', vmin=vmin, vmax=vmax);
    ax[1].set_xticks(linspace(0, rows_seg-1, 3))
    x_labl = around(linspace(start/sr, stop/sr, 3), 2)
    ax[1].set_xticklabels(x_labl, fontsize=10)
    # ax[1].set_yticks(1.17*arange(len(elecs_label_names)));
    # ax[1].set_yticklabels(elecs_label_names, fontsize=6);
    ax[1].set_xlabel('Time (seconds)', fontsize=10);

    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    textstr = 'Sz ' + str(segment+1) + ', ' + str(freq_band[0]) + ' - ' + str(freq_band[1]) + ' Hz'
    # ax[1].text(-.16, 0.95, textstr, transform=ax[1].transAxes, fontsize=6,
    # verticalalignment='top', bbox=props)
    ax[0].set_title(textstr)
    
    fig.tight_layout()

    return fig, ax


def plot_hippocampus_channels_original(data_elecs_band_embedded, elecs_chans, elecs_label_names, start, stop, segment, onset_time, freq_band, sr):

    rows_seg = stop - start

###################################################
    fig, ax = subplots(nrows=4, figsize=(6, 8))
                
    ax[0].plot(data_elecs_band_embedded[start:stop, :] + offset*arange(elecs_chans-1,-1,-1), 
            linewidth=1, color='b');
    
    ax[0].margins(x=0)
    ax[0].set_yticks(offset*arange(len(elecs_label_names)));
    ax[0].set_yticklabels(flip(elecs_label_names), fontsize=6);
    ax[0].set_xticks(linspace(0, rows_seg-1, 3))
    ax[0].set_xticklabels([])
    
    if onset_time*sr > start and onset_time*sr < stop:
        
        ax[0].vlines(onset_time*sr-start, -0.5*offset, offset*elecs_chans-.5, color="k", linewidth=1);
    
    
    ax[1].imshow(data_elecs_band_embedded[start:stop, :].T, aspect='auto', 
              cmap='bwr', vmin=vmin, vmax=vmax);
    ax[1].set_xticks(linspace(0, rows_seg-1, 3))
    # elec_labl = around(linspace(start/sr, stop/sr, 3), 2)
    # ax[1].set_xticklabels(elec_labl, fontsize=10)
    ax[1].set_xticklabels([])

    ax[1].set_yticks(arange(len(elecs_label_names)));
    ax[1].set_yticklabels(elecs_label_names[:], fontsize=6);


    data_sample_mean     = data_elecs_band_embedded[start:stop, :].mean(axis=1)
    data_sample_mean_abs = abs(data_elecs_band_embedded[start:stop, :]).mean(axis=1)

    # fig, ax = subplots(nrows=2)

    ax[2].plot(data_sample_mean, color='tomato', label='mean voltage')
    ax[3].plot(data_sample_mean_abs, color='r', label='mean abs voltage')
    ax[2].margins(x=0)
    ax[3].margins(x=0)
    ax[2].legend(), ax[3].legend()

    # ax[2].set_xlabel('Time (seconds)', fontsize=10);
    ax[2].set_xticks(linspace(0, rows_seg-1, 3))
    elec_labl = around(linspace(start/sr, stop/sr, 3), 2)
    ax[2].set_xticklabels(elec_labl, fontsize=10)
    
    ax[3].set_xlabel('Time (seconds)', fontsize=10)
    ax[3].set_xticks(linspace(0, rows_seg-1, 3))
    elec_labl = around(linspace(start/sr, stop/sr, 3), 2)
    ax[3].set_xticklabels(elec_labl, fontsize=10)
    

    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    
    textstr = 'Sz ' + str(segment+1) + ', ' + str(freq_band[0]) + ' - ' + str(freq_band[1]) + ' Hz'
    
    # ax[1].text(-.16, 0.95, textstr, transform=ax[1].transAxes, fontsize=6,
    # verticalalignment='top', bbox=props)
    ax[0].set_title(textstr)
    
    if onset_time*sr > start and onset_time*sr < stop:
        
        ax[1].vlines(onset_time*sr-start, -0.5, elecs_chans-.5, color="k", linewidth=1);
    
    
    fig.tight_layout()

    return fig, ax


def plot_hippocampus_channels_embedded(data_elecs_band_embedded, elecs_chans, elecs_label_names, start, stop, segment, onset_time, freq_band):

    rows_seg = stop - start

###################################################
    fig, ax = subplots(ncols=2, figsize=(8, 4))
                
    ax[0].plot(data_elecs_band_embedded[start:stop, :, plot_dim] + offset*arange(elecs_chans-1,-1,-1), 
            linewidth=1, color='b');
    
    ax[0].margins(x=0)
    ax[0].set_yticks(offset*arange(len(elecs_label_names)));
    ax[0].set_yticklabels(flip(elecs_label_names), fontsize=6);
    ax[0].set_xticks(linspace(0, rows_seg-1, 3))
    elec_labl = around(linspace(start/sr, stop/sr, 3), 2)
    ax[0].set_xticklabels(elec_labl, fontsize=10)
    
    
    if onset_time*sr > start and onset_time*sr < stop:
        
        ax[0].vlines(onset_time*sr-start, -0.5*offset, offset*elecs_chans-.5, color="k", linewidth=1);
    
    
    ax[1].imshow(data_elecs_band_embedded[start:stop, :, plot_dim].T, aspect='auto', 
              cmap='bwr', vmin=vmin, vmax=vmax);
    ax[1].set_xticks(linspace(0, rows_seg-1, 3))
    elec_labl = around(linspace(start/sr, stop/sr, 3), 2)
    ax[1].set_xticklabels(elec_labl, fontsize=10)
    ax[1].set_yticks(arange(len(elecs_label_names)));
    ax[1].set_yticklabels(elecs_label_names[:], fontsize=6);
    ax[1].set_xlabel('Time (seconds)', fontsize=10);

    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    
    textstr = 'Sz ' + str(segment+1) + ', ' + str(freq_band[0]) + ' - ' + str(freq_band[1]) + ' Hz'
    
    # ax[1].text(-.16, 0.95, textstr, transform=ax[1].transAxes, fontsize=6,
    # verticalalignment='top', bbox=props)
    ax[0].set_title(textstr)
    
    if onset_time*sr > start and onset_time*sr < stop:
        
        ax[1].vlines(onset_time*sr-start, -0.5, elecs_chans-.5, color="k", linewidth=1);
    
    
    fig.tight_layout()

    return fig, ax

def downsamp_interpol_data(data, rows_downs, chans_inter):

    from scipy.signal import decimate
    from scipy.interpolate import CubicSpline

    data_dec = decimate(data, rows_downs, axis=0)

    rows_dec, chans = data_dec.shape

    x  = arange(chans)
    
    xs = linspace(0, chans, chans_inter, endpoint=False)
    
    data_interp = zeros((rows_dec, chans_inter))
    
    for index, y in enumerate(data_dec):
    
        cs = CubicSpline(x, y)
    
        data_interp[index, :] = cs(xs) 
    
    return data_interp



def get_probabilities(data, bins):
    """
    Function to get M probability distributions from an N x M data set
    N: samples
    M: features 
    """

    from numpy import histogram

    cnts = zeros(shape=(data.shape[1], bins))

    for index, dat in enumerate(data.T):

        cnt = histogram(dat, bins=bins)[0]

        cnts[index, :] = cnt

    probs = array(cnts)/data.shape[0]
   
    return probs


def jensen_shannon_distance(p, q):
    """
    Function to compute the Jensen-Shannon Distance 
    between two probability distributions p and q
    """

    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = (entropy(p, m, base=2) + entropy(q, m, base=2)) / 2

    # compute the Jensen Shannon Distance
    distance = sqrt(divergence)

    return distance
