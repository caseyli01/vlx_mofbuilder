from ditopic import MOF_ditopic
from tritopic import MOF_tri
from tetratopic import MOF_tetra

def multitopic(templates_dir,nodes_dir,edges_dir,template,node_connection,linker_topic):
    if linker_topic==4:
        new_mof = MOF_tetra(templates_dir,nodes_dir,edges_dir,template,node_connection)
    elif linker_topic ==3:
        new_mof = MOF_tri(templates_dir,nodes_dir,edges_dir,template,node_connection)
    elif linker_topic ==2:
         new_mof = MOF_ditopic(templates_dir,nodes_dir,edges_dir,template,node_connection)
    return new_mof