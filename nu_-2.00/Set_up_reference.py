from m2_helper import *

filename_from = 'Data/beta_2.00_redux/nu_-2.10'
filename_to = 'Data/beta_3.00_redux/nu_-2.10'
beta_to = 3.00

with HDFArchive(filename_from+'.h5', 'r') as f:
    iterations = f['dmft_output']['iterations']
    Sigma_old = f['dmft_output']['Sigma_iw-%i'%iterations]
    dm = f['dmft_output']['dm-%i'%iterations]
    mu = f['dmft_output']['mu-%i'%iterations]

gf_struct = []
for name, sig in Sigma_old:
    gf_struct.append((name, sig.target_shape[0]))

newMesh = MeshImFreq(beta = 4.00, statistic='Fermion', n_iw=2050)
Sigma_new = BlockGf(mesh = newMesh, gf_struct=gf_struct)


def interpolate(Sigma_old, Sigma_new):
    mesh_old = Sigma_old.mesh
    mesh_new = Sigma_new.mesh
    iwn_old = np.array([complex(x) for x in mesh_old])
    iwn_new = np.array([complex(x) for x in mesh_new])
    for name, sig in Sigma_old:
        Sigma_interpolated = np.zeros((len(iwn_new), *sig.target_shape), dtype='complex')
        for i in range(sig.target_shape[0]):
            for j in range(sig.target_shape[0]):
                Sigma_interpolated[:,i, j] = np.interp(iwn_new.imag, iwn_old.imag, sig.data[:,i,j])
        Sigma_new[name] = GfImFreq(mesh=mesh_new, data = Sigma_interpolated)
    return

interpolate(Sigma_old, Sigma_new)

with HDFArchive(filename_to+'.h5', 'w') as f:
    f.create_group('dmft_output')
    iterations=0
    f['dmft_output']['iterations'] = 0
    f['dmft_output']['Sigma_iw-%i'%iterations] = Sigma_new
    f['dmft_output']['dm-%i'%iterations] = dm
    f['dmft_output']['mu-%i'%iterations] = mu

with HDFArchive(filename_to+'.h5', 'r') as f:
    iterations = f['dmft_output']['iterations']
    Sigma_new = f['dmft_output']['Sigma_iw-%i'%iterations]
    dm = f['dmft_output']['dm-%i'%iterations]
    mu = f['dmft_output']['mu-%i'%iterations]
