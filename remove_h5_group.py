import h5py

GROUP_TO_REMOVE = 'c_shell(r)_real_world_alpha=1e2'

with h5py.File('fluxmaps.h5', 'r+') as f:
    if GROUP_TO_REMOVE in f:
        del f[GROUP_TO_REMOVE]