import os
import tempfile
import warnings

import h5py
from tensorflow.keras.models import load_model, save_model

# Keras 3 treats HDF5 as a legacy format and warns on every save/load.
# We use HDF5 deliberately (agent serialization embeds model inside an
# HDF5 group alongside encoder metadata), so silence the noise.
warnings.filterwarnings('ignore', message='.*HDF5.*legacy.*', module='keras')


def save_model_to_hdf5_group(model, f):
    # Use Keras save_model to save the full model (including optimizer
    # state) to a file.
    # Then we can embed the contents of that HDF5 file inside ours.
    tempfd, tempfname = tempfile.mkstemp(prefix='tmp-kerasmodel', suffix='.h5')
    try:
        os.close(tempfd)
        save_model(model, tempfname)
        with h5py.File(tempfname, 'r') as serialized_model:
            root_item = serialized_model.get('/')
            serialized_model.copy(root_item, f, 'kerasmodel')
    finally:
        os.unlink(tempfname)


def load_model_from_hdf5_group(f, custom_objects=None):
    # Extract the model into a temporary file. Then we can use Keras
    # load_model to read it.
    tempfd, tempfname = tempfile.mkstemp(prefix='tmp-kerasmodel', suffix='.h5')
    try:
        os.close(tempfd)
        with h5py.File(tempfname, 'w') as serialized_model:
            root_item = f.get('kerasmodel')
            for attr_name, attr_value in root_item.attrs.items():
                serialized_model.attrs[attr_name] = attr_value
            for k in root_item.keys():
                f.copy(root_item.get(k), serialized_model, k)
        # compile=False: every agent recompiles before training, and
        # inference doesn't need compile state.  Skipping deserialisation
        # of the compiled optimizer/loss config avoids breakage when
        # Keras serializes losses with non-standard names (e.g. the
        # multi-output AC model's ['categorical_crossentropy', 'mse']
        # round-trips as 'keras.metrics.mse', which fails to reload).
        return load_model(tempfname, custom_objects=custom_objects,
                          compile=False)
    finally:
        os.unlink(tempfname)
