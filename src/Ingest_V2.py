import mne
from mne.io import read_raw_edf, concatenate_raws
import numpy as np


def get_filtered_data(subjects, runs, event_dict, tmin, tmax, data_type, tag):
    """
    subjects   : list of subject IDs to load
    runs       : list of run numbers (e.g. [4, 8, 12])
    event_dict : dict mapping annotation label -> semantic name,
                 e.g. {'T0': 'rest', 'T1': 'left_fist', 'T2': 'right_fist'}
    tmin, tmax : epoch window in seconds relative to event onset
    data_type  : 'train' or 'test' (which subject pool this is) -- stored
                 as metadata alongside each subject's data
    tag        : short string identifying which run-group this is,
                 e.g. 'lr' (left/right) or 'ff' (fists/feet) -- caller is
                 responsible for keeping outputs from different tags
                 separate (e.g. two different variables/files)

    Returns:
        filtered_dict : {
            "s001": {
                "epochs": array (n_epochs_total, n_channels, n_times) -- all
                          runs for this subject, concatenated,
                "labels": array (n_epochs_total,),
                "run_id": array (n_epochs_total,) -- which run each epoch
                          came from, parallel to epochs/labels,
                "data_type": data_type,
                "tag": tag,
            },
            ...
        }
        skipped : list of (subject, reason) for subjects that failed and
                  were excluded rather than crashing the whole batch
    """
    filtered_dict = {}
    skipped = []

    for sub in subjects:
        sub_epochs, sub_labels, sub_run_ids = [], [], []

        try:
            for run in runs:
                filepaths = mne.datasets.eegbci.load_data(sub, [run])
                raw_objects = [read_raw_edf(fp, preload=True) for fp in filepaths]
                raw = concatenate_raws(raw_objects)

                # Guard against silently mixing in subjects recorded at a different sampling rate (the known 88/89/92/100 issue).
                assert raw.info["sfreq"] == 160, (f"Subject {sub} run {run} has sfreq={raw.info['sfreq']}, expected 160")

                raw.filter(l_freq=8, h_freq=30)
                raw.notch_filter(freqs=[60])

                event, event_id_dict = mne.events_from_annotations(raw)

                mapping = {}
                for annot_key, semantic_name in event_dict.items():
                    if annot_key not in event_id_dict:
                        raise KeyError(
                            f"Subject {sub} run {run}: expected annotation "
                            f"'{annot_key}' not found. Found: {list(event_id_dict.keys())}"
                        )
                    mapping[semantic_name] = event_id_dict[annot_key]

                epochs = mne.Epochs(raw, event, mapping, tmin, tmax, baseline=None, preload=True)
                X = epochs.get_data()
                y = epochs.events[:, -1]

                sub_epochs.append(X)
                sub_labels.append(y)
                sub_run_ids.append(np.full(len(y), run, dtype=int))

        except Exception as e:
            skipped.append((sub, str(e)))
            print(f"[skip] subject {sub}: {e}")
            continue

        filtered_dict[f"s{sub:03}"] = {
            "epochs": np.concatenate(sub_epochs, axis=0),
            "labels": np.concatenate(sub_labels, axis=0),
            "run_id": np.concatenate(sub_run_ids, axis=0),
            "data_type": data_type,
            "tag": tag,
        }

    if skipped:
        print(f"\n{len(skipped)} subject(s) skipped: {[s for s, _ in skipped]}")

    return filtered_dict, skipped
