from m2_helper import *

#Input arguments passed by command line

import argparse
parser=argparse.ArgumentParser()
parser.add_argument("beta", help="inverse temperature", type=float)
parser.add_argument("loops", help="number of DMFT loops", type=int)
parser.add_argument("log_n", help="10^log_n number of QMC cycles", type=int)
parser.add_argument("polarizer", help="field that polarizes to the starting condition", type=float)
parser.add_argument("--mix", help="0.51-1.0 default:0.9", type = float)
parser.add_argument("--symm", help="K-IVC or IVC default:K-IVC", type = str)
parser.add_argument("--restart_from", help="neglect all runs after", type = int)
args=parser.parse_args()
beta = args.beta
log_n = args.log_n
loops = args.loops
polarizer = args.polarizer

###################### Check nu, beta, and pick filename for set of system params############################

# Physical constants

nu = -2.0
broken_symm = 'K-IVC'
Boltzmann=8.617333262e-5*1000
T=1/(Boltzmann*beta)
p={}
p['fit_min_n'] = int(160*T**(-0.9577746167096538))
p["fit_max_n"] = int(209*T**(-0.767031728744396))
p["imag_threshold"] = 1e-10
filename = 'Data/beta_{:.2f}/nu_{:.2f}'.format(beta, nu)
##############################################################################################################

# Not to be touched
# Run parameters
mix = 0.9
if args.mix:
    mix = args.mix
    # filename = filename+"mix_{:.2f}".format(mix)
if args.symm:
    broken_symm=args.symm
    filename = 'Data/'+broken_symm+'/'+filename[5:]

#Create directory if it doesn't exist
directory = os.path.dirname(filename)
os.makedirs(directory, exist_ok=True)

p["n_cycles"] = (10**log_n)//160
constrain=False
sample_len = 3 #FIXME: was 9
BZ_sampling, weights = sample_BZ_direct(sample_len)
n_iw = 1025
prec_mu = 0.001
if polarizer>0:
    prec_mu = prec_mu*10
p["length_cycle"] = 1000
p["n_warmup_cycles"] = 5000
p["perform_tail_fit"] = True
p["fit_max_moment"] = 4

#################### Choose the right symmetry-breaking here:##################################################

#Initialize the density matrix for the various ordered states.
#deg_shells tells the solver how many block there are in each
#sector. Basically spin up has two 2x2 blocks while spin down
#is entirely diagonal. This is important for the solver.
#If you say the orbitals are diagonal, no off-diagonal G and Sigma
#will be calculated. The name of the blocks here are written to
#be consistent with what triqs naturally makes. The deg_shells will
#later be forced on SK.

if broken_symm == 'K-IVC':
    dm_init = {}  # K-IVC
    dm_init['up'] = lin.block_diag(np.diag([1/2,1/2,1/2,1/2]),np.diag([1/2,1/2,1/2,1/2])+0.05*np.kron(sigy,sigy), + np.diag([1/2,1/2,1/2,1/2])-1/2*np.kron(sigy,sigy))
    dm_init['down'] = lin.block_diag(np.diag([1/2,1/2,1/2,1/2]),np.diag([1/2,1/2,1/2,1/2]), + np.diag([0,0,0,0]))
    deg_shells = [[['up_0', 'up_1'],['down_0', 'down_1', 'down_2', 'down_3']]]
elif broken_symm == 'IVC':
    dm_init = {}  # IVC
    dm_init['up'] = lin.block_diag(np.diag([1/2,1/2,1/2,1/2]),np.diag([1/2,1/2,1/2,1/2])+0.05*np.kron(sigx,sigx), + np.diag([1/2,1/2,1/2,1/2])-1/2*np.kron(sigx,sigx))
    dm_init['down'] = lin.block_diag(np.diag([1/2,1/2,1/2,1/2]),np.diag([1/2,1/2,1/2,1/2]), + np.diag([0,0,0,0]))
    deg_shells = [[['up_0', 'up_1'],['down_0', 'down_1', 'down_2', 'down_3']]]
elif broken_symm == 'IVC2':
    dm_init = {}  # IVC2
    dm_init['up'] = lin.block_diag(np.diag([1/2,1/2,1/2,1/2]),np.diag([1/2,1/2,1/2,1/2])-0.05*np.kron(sigy,sigx), + np.diag([1/2,1/2,1/2,1/2])+1/2*np.kron(sigy,sigx))
    dm_init['down'] = lin.block_diag(np.diag([1/2,1/2,1/2,1/2]),np.diag([1/2,1/2,1/2,1/2]), + np.diag([0,0,0,0]))
    deg_shells = [[['up_0', 'up_1'],['down_0', 'down_1', 'down_2', 'down_3']]]
elif broken_symm == 'IVC3':
    dm_init = {}  # IVC3=IVC
    dm_init['up'] = lin.block_diag(np.diag([1/2,1/2,1/2,1/2]),np.diag([1/2,1/2,1/2,1/2])+0.05*np.kron(sigx,sigx), + np.diag([1/2,1/2,1/2,1/2])-1/2*np.kron(sigx,sigx))
    dm_init['down'] = lin.block_diag(np.diag([1/2,1/2,1/2,1/2]),np.diag([1/2,1/2,1/2,1/2]), + np.diag([0,0,0,0]))
    deg_shells = [[['up_0', 'up_1'],['down_0', 'down_1', 'down_2', 'down_3']]]
    
else:
    raise ValueError("Broken symmetry can only by K-IVC or IVC right now.")

############################################################################################################################################ 
## Write a converter for dft_tools. This basically writes a text file with the
#Hamiltonian in it, in a format that dft_tools can read. The header has the following
#shape:
 
def set_converter(H):
    
    #n_k                  # number of k-points
    #density_required     # electron density
    #3                    # number of total atomic shells
    #1 0 2 4              # atom, sorting number, l quantum number, total number of levels per shell (dim)
    #1 1 2 4              # ...
    #1 2 2 4              # ...
    #1                    # number of correlated shells
    #1 2 0 4 0 0          # atom, sorting number of the correlated shell, l quantum number, dim, spin-orbit parameter, interactions? (these last two are not used)
    #2 2 2                # number of ireps, dim of irep
    
    
    n_k = len(BZ_sampling)
    density_required = 12+nu 
    n_shells = 3
    n_valleys = 2

    #just print out the stuff
    if mpi.is_master_node():
        with open(filename, "w") as A:
            A.write(str(n_k)+"\n"+str(density_required)+"\n"+str(n_shells)+"\n")
            # describe all shells
            for ish in range(n_shells):
                A.write("1 "+str(ish)+" 0 "+str(2*n_valleys)+"\n")
            # describe the correlated shells
            A.write("1\n")
            A.write("1 2 0 "+str(2*n_valleys)+" 0 0\n")
            A.write(str(n_valleys)+" 2"*n_valleys+"\n")

        for ik in range(n_k):
            with open(filename, "a") as A:
                for row in H(*BZ_sampling[ik]): #* basically translates a list into a set of numbers
                    A.write("\n")
                    for hopping in row:
                        A.write(str(hopping.real)+" ")
                A.write("\n")
                for row in H(*BZ_sampling[ik]):
                    A.write("\n")
                    for hopping in row:
                        A.write(str(hopping.imag)+" ")
                A.write("\n")
    
    # This saves to the hdf5 files. If no hdf5 filename is provided, it has the same name as the hamiltonian file.
    Converter = HkConverter(filename = filename)
    Converter.convert_dft_input()

    ## Fix the BZ weights for the converter
    if mpi.is_master_node():
        with HDFArchive(filename+'.h5','a') as f:
            f['dft_input']['bz_weights'] = weights           

##################################################################################################################################            
# Just create the noninterating H (see helper file)
H0 = SBhamiltonian()

#just a wrapper
H = lambda kx, ky: H0(kx, ky)

#create the H for the points in the grid
set_converter(H)
###################################################################################################################################

#This initializes the hamiltonian for triqs usage, getting it from the hdf5 file, where
#in turn it was saved by the HkConverter call.
#use_dft_blocks means that we will divide the spin blocks into subblocks depending on the
#density matrix of the correlated shells.
SK = SumkDFT(hdf_file=filename+'.h5',use_dft_blocks=True,beta=beta)

# At this stage, everything is diagonale because there is no symmetry breaking yet. The f-shell
# is just identically zero. So the following command will return this:
# [[['up_0', 'up_1', 'up_2', 'up_3', 'down_0', 'down_1', 'down_2', 'down_3']]]
# Notice this is deg_shells, but for a diagonal problem with 4 orbitals per spin.
print(SK.deg_shells)
#####################################################################################################################################

## Check if previous runs exist
previous_runs = 0
previous_present = False
if mpi.is_master_node():
    if os.path.isfile(filename+'.h5'):
        with HDFArchive(filename+'.h5','a') as f:
            if 'dmft_output' in f:
                ar = f['dmft_output']
                if 'iterations' in ar:
                    previous_present = True
                    previous_runs = ar['iterations']
                    if args.restart_from:
                        previous_runs=args.restart_from
            else:
                f.create_group('dmft_output')
previous_runs = mpi.bcast(previous_runs)
previous_present = mpi.bcast(previous_present)

# Print
if mpi.is_master_node():
    print("Starting from previous run: ",previous_present)
    print("Starting from iteration: ",previous_runs)
#####################################################################################################################################

# Save run parameters
if mpi.is_master_node():
    with HDFArchive(filename+'.h5', 'a') as ar:
        ar['dmft_output']['params_%i-%i'%(previous_runs+1, loops+previous_runs)] = (loops, p['n_cycles'], mix, constrain, polarizer)
######################################################################################################################################

#The DMFT loop
for iteration_number in range(1,loops+1):
    if mpi.is_master_node(): print("Iteration = ", iteration_number)
    ###########################################################################################################################
    #Initialize the density matrix at the first iteration if starting from scratch. Basically every orbital is half-filled and
    #there is a small off-diagonal density according to the chosen symmetry breaking.
    if not previous_present and iteration_number==1:
        dm = dm_init
        if mpi.is_master_node():
            with HDFArchive(filename+'.h5', 'a') as ar:
                ar['dmft_output']['dm-0'] = dm
    
    #Otherwise read the density matrix from the previous run.
    if previous_present and iteration_number==1:
        dm = 0
        if mpi.is_master_node():
            with HDFArchive(filename+'.h5', 'r') as ar:
                dm = ar['dmft_output']['dm-%i'%previous_runs]
    
    #broadcast the density matrix
    dm = mpi.bcast(dm)
    ############################################################################################################################
    
    #here the mean-field terms and the polarization field (kick) are calculated 
    Hmf = {}
    
    Hmf['up'] = mean_field_terms(dm, spin='up')
    Hmf['down'] = mean_field_terms(dm, spin='down')
    
    if constrain: #not sure of the name here
        Hmf['up'] = mean_field_terms(dm_init, spin='up')
        Hmf['down'] = mean_field_terms(dm_init, spin='down')
        
    H_pol = {}
    H_pol['up'] = -polarizer*(dm_init['up'] - lin.block_diag(1/2*np.eye(8), (1/2+nu/8)*np.eye(4) ))
    H_pol['down'] = -polarizer*(dm_init['down'] - lin.block_diag(1/2*np.eye(8), (1/2+nu/8)*np.eye(4) ))

    #Here they are added to the Hamiltonian
    H = lambda kx, ky: H0(kx, ky) + Hmf['up'] + H_pol['up']   # why only up here? probably because there is no down polarization (KIVC + VP)
    
    #This redoes the hamiltonian (nonint + mean-field + polarization)
    set_converter(H)
    
    #####################################################################################################################################
    if mpi.is_master_node():
        # This thins is patching the H_up to create the H_down
        with HDFArchive(filename+'.h5','a') as f:
            f['dft_input']['SP'] = 1 #Set as spin-polarized
            n_orb = f['dft_input']['n_orbitals']
            f['dft_input']['n_orbitals'] = np.concatenate((n_orb, n_orb), axis = 1) #number of orbitals for down is the same as for up
            # proj_mat is a dft_tools thing. Since I did set_converter for the up H, I need the same thing for the down H.
            proj_mat = f['dft_input']['proj_mat']
            
            #see what it is. The dimensions are (N_k, Nspin,N_correlated_blocks,N_orb_correlated,N_bands).These are the rectangular terms.
            print(np.shape(proj_mat))
            for row in proj_mat[0,0,0,:,:]:
                print(" ".join(f"{elem.real: >7.4f}{'+' if elem.imag >= 0 else '-'}{abs(elem.imag):<7.4f}i" for elem in row))
            
            #The hybridization between f and c is identical per spin, because both the noninteracting matrix and the interaction are
            #symmetric. The symmetry breaking is a f-block thing alone. So this proj_mat can be copied by spin without issue.
            f['dft_input']['proj_mat'] = np.concatenate((proj_mat, proj_mat), axis = 1)
            
            #Now build the hamiltonian for down spin
            hopping1 = f['dft_input']['hopping'] #get the hopping for up
            hopping2 = hopping1.copy()           #copy it so the dimension is the same
            for ik in range(len(BZ_sampling)):   #modify the hopping for down
                hopping2[ik,0,:,:] = H0(*BZ_sampling[ik]) + Hmf['down']+ H_pol['down']
                
            #save the up and down hoppings
            f['dft_input']['hopping'] = np.concatenate((hopping1, hopping2), axis = 1)
    
     ##########################################################################################################################################

    
    
    #This diagonalizes the Hlocal. It saves transformation matrices in SK.block_structure for later use.
    #But here a mess happens: since in the block above we basically scotch-taped the down part in the hdf5,
    #now the correlated orbitals are messed up. After SumkDFT is called, it returns [[['down_0', 'down_1', 'down_2', 'down_3']]].
    #Up is gone. It will need to be forced back and it will be done at line 369
    print(SK.deg_shells)
    SK = SumkDFT(hdf_file=filename+'.h5',use_dft_blocks=True, beta=beta)
    print(SK.deg_shells)
    SK.calculate_diagonalization_matrix(write_to_blockstructure=True)
    print(SK.deg_shells)
    
    
    #This is 4 orbitals
    n_orb = SK.corr_shells[0]['dim']
    spin_names = ["up","down"]
    orb_names = [i for i in range(n_orb)]
    
    #This is basically deg_shells. [0] because we have only one correlated shell.
    #It will print out [('down_0', 1), ('down_1', 1), ('down_2', 1), ('down_3', 1), ('up_0', 2), ('up_1', 2)]
    #Later, we will need to force this format again, in case it loses symmetries
    gf_struct = SK.gf_struct_solver_list[0]


    
    #Create the interacting Hamiltonian. This is just a Kanamori Hamiltonian without Hund, for 4 orbitals
    Umat, Upmat = U_matrix_kanamori(n_orb=n_orb, U_int=U, J_hund=0)
    
    #This writes the big interacting Hamiltonian in triqs format, i.e. with the operators.
    #SK.sumk_to_solver[0] (0 because we only have one correlated shell) is the crucial quantity. It contains
    #the orbitals with the degeneracy block they belong to. In our case:
    #{('up', 0): ('up_0', 0), ('up', 3): ('up_0', 1), ('up', 1): ('up_1', 0), ('up', 2): ('up_1', 1), 
    #('down', 0): ('down_0', 0), ('down', 1): ('down_1', 0), ('down', 2): ('down_2', 0), ('down', 3): ('down_3', 0)}
    h_int = h_int_density(spin_names, n_orb, map_operator_structure=SK.sumk_to_solver[0], U=Umat, Uprime=Upmat)
    
    #This initializes the solver
    S = Solver(beta=beta, gf_struct=gf_struct)

    
    #Stuff for the first iteration starting from scratch
    ####################################################################################################
    if not previous_present and iteration_number==1:
        #this is the rotation matrix between the original orbitals and the diagonalized
        #basis that is used in the solver. It is only used to initialize a decent Sigma.        
        tt = SK.block_structure.effective_transformation_solver[0]
        for block_name, mat in tt.items():
            print(f"Block: {block_name}")
            print("Shape:", mat.shape)
            print(mat)
            print("-"*40)        
        #We need to start from some self-energy, this initializes it to a decent guess.
        for name, Sig in S.Sigma_iw:
            if 'up' in name:
                temp = -tt[name]@Hmf['up'][-4:,-4:]@(tt[name].conjugate().T)
                Sig << np.diag([np.average(np.diag(temp))]*len(temp)) + (nu*(W1+W3)/2)
            elif 'down' in name:
                temp = -tt[name]@Hmf['down'][-4:,-4:]@(tt[name].conjugate().T)
                Sig << np.diag([np.average(np.diag(temp))]*len(temp)) + (nu*(W1+W3)/2)
            else:
                raise("problem with self-energy initialization")
                
        #also the chemical potential needs to be initialized decently
        chemical_potential=(nu*(W1+W3)/2)
        if mpi.is_master_node():
            with HDFArchive(filename+'.h5', 'a') as f:
                f['dmft_output']['mu-0'] = chemical_potential

        if mpi.is_master_node():
            with HDFArchive(filename+'.h5', 'a') as ar:
                ar['dmft_output']['Sigma_iw-0'] = S.Sigma_iw
    #################################################################################################### 
     
 
    #Otherwise start from an older sigma
    if mpi.is_master_node():
        with HDFArchive(filename+'.h5', 'r') as ar:
            S.Sigma_iw = ar['dmft_output']['Sigma_iw-%i'%(iteration_number+previous_runs-1)]
    S.Sigma_iw << mpi.bcast(S.Sigma_iw)
    
  
    #This is a tricky point. Basically we have reconstructed a SK, but it doesn't know
    #which blocks are diagonal and which not. So here the structure set up together with
    #dm_init is forced back. Before it prints
    #[[['down_0', 'down_1', 'down_2', 'down_3']]]
    #and after it prints
    #[[['up_0', 'up_1'], ['down_0', 'down_1', 'down_2', 'down_3']]]
    
    print(SK.deg_shells)
    SK.deg_shells = deg_shells
    print(SK.deg_shells)
    
   
    if mpi.is_master_node():
        print(("The degenerate shells are ", SK.deg_shells))
     
     
    #Do this to a self-energy that is already present
    if iteration_number>1 or previous_present: 
        #So this does the symmetrization between the elements of the self-energy.
        #Diagonal elements are averaged, off-diagonal elements are symmetrized according
        #to the symmetries of the problem.
        SK.symm_deg_gf(S.Sigma_iw,ish=0)
        #Then we save the self-energy into the SK class
        SK.set_Sigma([ S.Sigma_iw ])                            
    
    
    #This stuff looks for and sets the chemical potential given the desired occupation
    chemical_potential=0
    if mpi.is_master_node():
        with HDFArchive(filename+'.h5', 'r') as f:
            chemical_potential = f['dmft_output']['mu-%i'%(previous_runs+iteration_number-1)]
    chemical_potential = mpi.bcast(chemical_potential)
    SK.set_mu(chemical_potential)
    chemical_potential = SK.calc_mu(precision = prec_mu , delta = 10)  # find the chemical potential for given density
    if chemical_potential==None:
        if mpi.is_master_node():
            with HDFArchive(filename+'.h5', 'r') as f:
                chemical_potential = f['dmft_output']['mu-%i'%(previous_runs+iteration_number-1)]
        chemical_potential = mpi.bcast(chemical_potential)
        SK.set_mu(chemical_potential)
        chemical_potential = SK.calc_mu(precision = prec_mu*10 , delta = 10)
    
    #This saves the G_loc of the model with all the mess we made as the G_iw in the solver
    S.G_iw << SK.extract_G_loc()[0]                         # calc the local Green function
                                                                
    if mpi.is_master_node():
        mpi.report("Total charge of Gloc : %.6f"%S.G_iw.total_density())
        
    # Calculate new G0_iw to input into the solver:
    S.G0_iw << S.Sigma_iw + inverse(S.G_iw)
    S.G0_iw << inverse(S.G0_iw)
    
        
    # Solve the impurity problem:
    S.solve(h_int=h_int, **p)

    if mpi.is_master_node():
        mpi.report("Total charge of impurity problem : %.6f"%S.G_iw.total_density())
    
    ##########################################################################
    #We got a self-energy, copy it to another object to work on it
    Sigma_symm = S.Sigma_iw.copy()
    
    ##################
    #   TEST PRINT   #
    ##########################################################################
    for name, Sig in S.Sigma_iw:
        n_orb = Sig.data.shape[1]   # number of orbitals in this block
        print(f"Block {name} has {n_orb} orbitals")
    ##########################################################################
    
    #This symmetrizes the self-energy again
    SK.symm_deg_gf(Sigma_symm,ish=0)
    SK.set_Sigma([Sigma_symm])
    
    #Density matrix
    dm = {}
    dm['up'] = 1j*np.zeros((12,12))
    dm['down'] = 1j*np.zeros((12,12))
    
    #This thing is weighted on the k-points.
    #It is the only point where weights are taken into account.
    for ik in range(len(BZ_sampling)):
        w = weights[ik]
        G = SK.lattice_gf(ik)
        for name, g in G:
            dm[name] += w*g.density()
    for name, dens in dm.items():
        if mpi.is_master_node():
            mpi.report("Symmetrizing density. The largest imaginary component in the diagonal of the density matrix is {}".format(np.max(np.abs(np.diag(dm[name].imag)))))
        dm[name] = 1/2*(dm[name] + dm[name].conjugate().T)
        if broken_symm=='IVC3':
            ccf = np.trace(dm['up'][-4:,-4:])/4
            cc1 = np.trace(dm['up'][:4,:4])/4
            cc2 = np.trace(dm['up'][4:-4,4:-4])/4
            for i in range(4):
                dm['up'][-4+i,-4+i]=ccf
                dm['up'][i,i]=cc1
                dm['up'][4+i,4+i]=cc2
    
    #This only saves stuff
    if mpi.is_master_node():
        with HDFArchive(filename+'.h5','r') as ar:
            mpi.report("Mixing Sigma and G with factor %s"%mix)
            S.Sigma_iw << mix * S.Sigma_iw + (1.0-mix) * ar['dmft_output']['Sigma_iw-%i'%(iteration_number+previous_runs-1)]
            if previous_present or iteration_number>1:
                S.G_iw << mix * S.G_iw + (1.0-mix) * ar['dmft_output']['G_iw-%i'%(iteration_number+previous_runs-1)]
            for name, mat in dm.items():
                dm[name]=mix*mat + (1.0-mix)*ar['dmft_output']['dm-%i'%(iteration_number+previous_runs-1)][name]
    S.G_iw << mpi.bcast(S.G_iw)
    S.Sigma_iw << mpi.bcast(S.Sigma_iw)
    dm = mpi.bcast(dm)
    
    if mpi.is_master_node():
        with HDFArchive(filename+'.h5', 'a') as ar:
            ar['dmft_output']['iterations'] = iteration_number + previous_runs  
            ar['dmft_output']['G_0-%i'%(iteration_number + previous_runs)] = S.G0_iw
            ar['dmft_output']['G_tau-%i'%(iteration_number + previous_runs)] = S.G_tau
            ar['dmft_output']['G_iw-%i'%(iteration_number + previous_runs)] = S.G_iw
            ar['dmft_output']['Sigma_iw-%i'%(iteration_number + previous_runs)] = S.Sigma_iw
            ar['dmft_output']['mu-%i'%(iteration_number + previous_runs)] = chemical_potential
            ar['dmft_output']['dm-%i'%(iteration_number + previous_runs)] = dm
        print('saved sigma, G etc')
    
    SK.save(['chemical_potential','dc_imp','dc_energ'])
