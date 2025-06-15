from src.MOF_builder.vlx_mof_builder_v2 import MofBuilder

mof = MofBuilder()
mof.mof_family = 'UiO-66'
mof.node_metal = 'Zr'

mof.linker_xyz_file = '3bdc.xyz'

mof.set_rotation_optimizer_display(True)

#mof.set_use_saved_optimized_rotations_npy('rota')
#mof.set_use_saved_rotations_as_initial_guess(False)

#mof.use_saved_optimized_rotations_npy('rota')
#save optimized rotations to numpy file for later use
#mof.save_optimized_rotations('rota')
#mof.set_supercell_cleaved_buffer_minus(0.2)
mof.supercell = (1, 1, 1)
mof.build()
mof.write_gromacs_files()
mof.show(res_indices=True, res_names=False)
