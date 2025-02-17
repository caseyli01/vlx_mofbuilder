from src.MOF_builder.vlx_mof_builder_v2 import mof_builder
mof = mof_builder()
mof.preparation.select_mof_family('MOF-525')
mof.preparation.select_node_metal('Zr')
#mof.preparation.use_dummy_node(True)
mof.preparation.fetch_node()
mof.preparation.fetch_linker('noCoTCPP.xyz')
mof.preparation_check()

#mof.use_saved_optimized_rotations_npy('rotb')
#save optimized rotations to numpy file for later use
mof.save_optimized_rotations('rotb')
mof.set_supercell([1,1,1])
mof.build()
mof.write_gro()