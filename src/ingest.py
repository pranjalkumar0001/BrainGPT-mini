import mne
from mne.io import read_raw_edf, concatenate_raws
import random
import numpy as np
import os

"""
making data directory if it doesn't exist, and setting DATA_DIR to the relative path "data" for saving files
"""
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

os.makedirs(DATA_DIR, exist_ok=True)
DATA_DIR = "data"


def get_filtered(subjects, runs, event_dict, tmin, tmax, data_type, tag):
    """
    subjects   : list of subject IDs to load
    runs       : list of run numbers (e.g. [4, 8, 12])
    event_dict : dict mapping annotation label -> semantic name,
                 e.g. {'T0': 'rest', 'T1': 'left_fist', 'T2': 'right_fist'}
    tmin, tmax : epoch window in seconds relative to event onset
    data_type  : 'train' or 'test' (which subject pool this is)
    tag        : short string identifying which run-group this is,
                 e.g. 'lr' (left/right) or 'ff' (fists/feet) -- keeps
                 output files distinct so two different tasks are
                 never accidentally concatenated together.
    """
    X_list, y_list = [], []

    for subject in subjects:
        file_paths = mne.datasets.eegbci.load_data(subject, runs)
        raw_objects = [read_raw_edf(fp, preload=True) for fp in file_paths]
        raw = concatenate_raws(raw_objects)

        # Guard against silently mixing subjects recorded at a different
        # sampling rate (the known 88/89/92/100 issue, checked here in
        # case the excluded list ever needs revisiting).
        assert raw.info['sfreq'] == 160, (
            f"Subject {subject} has sfreq={raw.info['sfreq']}, expected 160"
        )

        raw.filter(l_freq=8, h_freq=30)
        raw.notch_filter(freqs=[60])

        events, event_id_dict = mne.events_from_annotations(raw)

        # Build the mapping from event_id_dict's ACTUAL codes, not
        # hardcoded integers -- concatenate_raws can insert boundary
        # annotations that shift what code T0/T1/T2 end up with.
        mapping = {}
        for annot_key, semantic_name in event_dict.items():
            if annot_key not in event_id_dict:
                raise KeyError(
                    f"Subject {subject}: expected annotation '{annot_key}' "
                    f"not found. Found: {list(event_id_dict.keys())}"
                )
            mapping[semantic_name] = event_id_dict[annot_key]

        epochs = mne.Epochs(
            raw, events, event_id=mapping,
            tmin=tmin, tmax=tmax,
            baseline=None, preload=True
        )

        X = epochs.get_data()
        y = epochs.events[:, -1]

        # Mask out rest, keep only the two task classes present in `mapping`
        task_codes = [v for k, v in mapping.items() if k != 'rest']
        mask = np.isin(y, task_codes)
        X_list.append(X[mask])
        y_list.append(y[mask])

    X_final = np.concatenate(X_list, axis=0)
    y_final = np.concatenate(y_list, axis=0)

    np.save(os.path.join(DATA_DIR, f"X_{data_type}_{tag}.npy"), X_final)
    np.save(os.path.join(DATA_DIR, f"y_{data_type}_{tag}.npy"), y_final)
    print(f"[{data_type}_{tag}] saved X{X_final.shape}, y{y_final.shape}")


#main
subjects = [i for i in range(1, 110) if i not in [88, 89, 92, 100]]
random.seed(1234)
random.shuffle(subjects) #random shuffle to avoid any bias in train/test split
n = int(0.85 * len(subjects))
train_sub = subjects[:n]   
test_sub  = subjects[n:]  

# ---- run groups: kept as SEPARATE tasks, never merged ----
run_lr = [4, 8, 12]
event_dict_lr = {'T0': 'rest', 'T1': 'left_fist', 'T2': 'right_fist'}

run_ff = [6, 10, 14]
event_dict_ff = {'T0': 'rest', 'T1': 'both_fists', 'T2': 'both_feet'}
TMIN, TMAX = 0.0, 4

# ---- four calls: {train, test} x {left/right, fists/feet} ----
get_filtered(train_sub, run_lr, event_dict_lr, TMIN, TMAX, "train", "lr")
get_filtered(test_sub,  run_lr, event_dict_lr, TMIN, TMAX, "test",  "lr")
get_filtered(train_sub, run_ff, event_dict_ff, TMIN, TMAX, "train", "ff")
get_filtered(test_sub,  run_ff, event_dict_ff, TMIN, TMAX, "test",  "ff")