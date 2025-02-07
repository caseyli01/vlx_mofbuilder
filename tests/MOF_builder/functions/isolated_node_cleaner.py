import numpy as np
import networkx as nx
import re




def reindex_frag_array(all_array,fragmentname):
    # according to atoms number in one residue(like a node has 66 atoms) to reassign res_number to residue for next call
    frag_array = all_array[all_array[:,4]== fragmentname] # fragmentname can be "EDGE" or "NODE"
    first_frag=frag_array[0,5]
    stop = 0
    while frag_array[stop,5] == first_frag:
        stop+=1
    # stop is the atoms number in one residue
    frag_array[:,5] = np.array([i//stop+1 for i in range(len(frag_array))])
    #change the index (res_number) to all residues sharing the same name "EDGE" or "NODE"

    return frag_array


def find_single_frag_center_fc(refrag_fcarr,index):
    #extract a residue by res_number(index) and find "X" atoms and then calculate the average of all X to be the center of this residue 
    single_frag = refrag_fcarr[refrag_fcarr[:,5]==index]
    indices=[i for i in range(len(single_frag)) if re.sub('[0-9]','',single_frag[i,2])=='X']
    frag_center = np.mean(single_frag[indices][:,6]),np.mean(single_frag[indices][:,7]),np.mean(single_frag[indices][:,8])
    frag_center = np.round(frag_center,3)
    return frag_center

def get_frag_centers_fc(refrag_fcarr):
    frag_centers = []
    for i in range(refrag_fcarr[-1,5]):
        res_index = i+1
        single_frag_center=find_single_frag_center_fc(refrag_fcarr,res_index)
        Odist = np.linalg.norm(single_frag_center) #Odist is the distance from the beginning point(0,0,0) to the center of this residue
        frag_centers.append((res_index,single_frag_center,Odist))
    return frag_centers

def calculate_eG_net(edgefc_centers,nodefc_centers,linker_topics,sc_unit_cell):
    #calculate and add all edge_center as node to eG, then search for node_center in a range(around closest node distance)
    #this eG is searching neighbor nodes from edge, so the absolute isolated nodes (just single node) cannot be counted because it cannot be found from an edge 
    eG=nx.Graph()
    for i in edgefc_centers:
        eG.add_nodes_from([('E'+str(i[0]), {'fc': i[1],'Odist': i[2]})])

    for e_eG_index in range(len(edgefc_centers)):
        e_eG_node = list(eG.nodes)[e_eG_index]
        e_fc = eG.nodes[e_eG_node]['fc']
        e_dist = eG.nodes[e_eG_node]['Odist']

        n_candidates = [] #seach neighbor node candidates in a +-0.5 shell range of this edge center 
        for j in nodefc_centers:
            if j[2]>e_dist-1 and j[2]< e_dist+1:
                    le_fc=np.linalg.norm(j[1] - e_fc)
                    if le_fc < 1:
                        le_cc = np.linalg.norm(np.dot(sc_unit_cell,(j[1] - e_fc)))
                        n_candidates.append((le_cc,j,e_fc))

        le_list = [i[0] for i in n_candidates]
        le_list.sort()

        if re.sub('[0-9]','',e_eG_node): #if this node is a edge(_center) node in eG graph
            for i in n_candidates:
                if i[0] == le_list[0]:
                    eG.add_nodes_from([(i[1][0],{'fc':i[1][1],'Odist':i[1][2]})])
                    eG.add_edge(e_eG_node,i[1][0])
                if len(le_list)>(linker_topics-1):
                    if (i[0] <= le_list[linker_topics-1]) & (i[0]< 1.3*le_list[0]) :
                        eG.add_nodes_from([(i[1][0],{'fc':i[1][1],'Odist':i[1][2]})])
                        eG.add_edge(e_eG_node,i[1][0])
                else:
                    if (i[0]< 1.3*le_list[0]) :
                        eG.add_nodes_from([(i[1][0],{'fc':i[1][1],'Odist':i[1][2]})])
                        eG.add_edge(e_eG_node,i[1][0])

    return eG








def calculate_eG_net_ditopic(edgefc_centers,nodefc_centers,linker_topics):
    #calculate and add all edge_center as node to eG, then search for node_center in a range(around closest node distance)
    #this eG is searching neighbor nodes from edge, so the absolute isolated nodes (just single node) cannot be counted because it cannot be found from an edge 
    eG=nx.Graph()
    for i in edgefc_centers:
        eG.add_nodes_from([('E'+str(i[0]), {'fc': i[1],'Odist': i[2]})])

    for e_eG_index in range(len(edgefc_centers)):
        e_eG_node = list(eG.nodes)[e_eG_index]
        e_fc = eG.nodes[e_eG_node]['fc']
        e_dist = eG.nodes[e_eG_node]['Odist']

        n_candidates = [] #seach neighbor node candidates in a +-0.5 shell range of this edge center 
        for j in nodefc_centers:
            if j[2]>e_dist-0.5 and j[2]< e_dist+0.5:
                    le=np.linalg.norm(j[1] - e_fc)
                    if le < 0.5: #for ditopic edge center to neighbor node
                        n_candidates.append((le,j,e_fc))

        le_list = [i[0] for i in n_candidates]
        le_list.sort()

        if re.sub('[0-9]','',e_eG_node): #if this node is not an edge(_center) node in eG graph
            for i in n_candidates:
                if i[0] == le_list[0]:
                    eG.add_nodes_from([(i[1][0],{'fc':i[1][1],'Odist':i[1][2]})])
                    eG.add_edge(e_eG_node,i[1][0],type='real')
                elif (i[0] <= le_list[linker_topics-1]) & (i[0]< 1.3*le_list[0]) :

                    eG.add_nodes_from([(i[1][0],{'fc':i[1][1],'Odist':i[1][2]})])
                    eG.add_edge(e_eG_node,i[1][0],type='real')
    return eG



def locate_min_idx(matrix):
    min_idx = np.unravel_index(matrix.argmin(), matrix.shape)
    return min_idx[0],min_idx[1]


def calculate_eG_net_ditopic_PILLAR(edgefc_centers,nodefc_centers,linker_topics,bridge_node_distance):
    #calculate and add all edge_center as node to eG, then search for node_center in a range(around closest node distance)
    #this eG is searching neighbor nodes from edge, so the absolute isolated nodes (just single node) cannot be counted because it cannot be found from an edge 
    eG=nx.Graph()
    for i in edgefc_centers:
        eG.add_nodes_from([('E'+str(i[0]), {'fc': i[1],'Odist': i[2]})])
        print('E'+str(i[0]),i[1],i[2])
    #2 rounds: round1, search for closest node, round2, search for 2nd closest node
    e_n_distance_matrix = np.zeros((len(edgefc_centers),len(nodefc_centers)))
    for i in range(len(edgefc_centers)):
        for j in range(len(nodefc_centers)):
            e_n_distance_matrix[i,j] = np.linalg.norm(edgefc_centers[i][1] - nodefc_centers[j][1])

    #nodes can be shared, so hungrain algorithm is not suitable
    for m in range(len(edgefc_centers)):
        e_n_min_distance = np.min(e_n_distance_matrix[m:m+1,:])
        if e_n_min_distance < 0.6:
            _,e_n = locate_min_idx(e_n_distance_matrix[m:m+1,:])
            
            eG.add_nodes_from([(nodefc_centers[e_n][0],{'fc':nodefc_centers[e_n][1],'Odist':nodefc_centers[e_n][2]})])
            eG.add_edge('E'+str(edgefc_centers[m][0]),nodefc_centers[e_n][0],type='real')
            e_n_distance_matrix[m,e_n] = 1000
        _, sec_round_n = locate_min_idx(e_n_distance_matrix[m:m+1,:])
        if e_n_distance_matrix[m,sec_round_n] < 0.6:
            _,e_k = locate_min_idx(e_n_distance_matrix[m:m+1,:])
            eG.add_nodes_from([(nodefc_centers[e_k][0],{'fc':nodefc_centers[e_k][1],'Odist':nodefc_centers[e_k][2]})])
            eG.add_edge('E'+str(edgefc_centers[m][0]),nodefc_centers[e_k][0],type='real')

       

    #add pillar nodes virtual edges
    n_n_distance_matrix = np.zeros((len(nodefc_centers),len(nodefc_centers)))
    for i in range(len(nodefc_centers)):
        for j in range(len(nodefc_centers)):
            n_n_distance_matrix[i,j] = np.linalg.norm(nodefc_centers[i][1] - nodefc_centers[j][1])
        n_n_distance_matrix[i,i] = 1000
    #use hungrain algorithm to find the shortest path between all nodes
    

    for i in range(len(nodefc_centers)):
        n_n_min_distance = np.min(n_n_distance_matrix[i:i+1,:])
        if n_n_min_distance < bridge_node_distance:
            n_i,n_j = locate_min_idx(n_n_distance_matrix[i:i+1,:])
            eG.add_nodes_from([(nodefc_centers[i][0],{'fc':nodefc_centers[i][1],'Odist':nodefc_centers[i][2]})])
            eG.add_nodes_from([(nodefc_centers[n_j][0],{'fc':nodefc_centers[n_j][1],'Odist':nodefc_centers[n_j][2]})])
            eG.add_edge(nodefc_centers[i][0],nodefc_centers[n_j][0],type='virtual')
    return eG



def calculate_eG_net_multitopic(edgefc_centers,nodefc_centers,linker_topics,bridge_node_distance):
    #calculate and add all edge_center as node to eG, then search for node_center in a range(around closest node distance)
    #this eG is searching neighbor nodes from edge, so the absolute isolated nodes (just single node) cannot be counted because it cannot be found from an edge 
    #eG nodes are all nodes with note CV in TG
    eG=nx.Graph()  
    
    for i in edgefc_centers:
        eG.add_nodes_from([('E'+str(i[0]), {'fc': i[1],'Odist': i[2]})])
        print('E'+str(i[0]),i[1],i[2])
    #2 rounds: round1, search for closest node, round2, search for 2nd closest node
    e_n_distance_matrix = np.zeros((len(edgefc_centers),len(nodefc_centers)))
    for i in range(len(edgefc_centers)):
        for j in range(len(nodefc_centers)):
            e_n_distance_matrix[i,j] = np.linalg.norm(edgefc_centers[i][1] - nodefc_centers[j][1])

    #nodes can be shared, so hungrain algorithm is not suitable
    for m in range(len(edgefc_centers)):
        e_n_min_distance = np.min(e_n_distance_matrix[m:m+1,:])
        if e_n_min_distance < 0.6:
            _,e_n = locate_min_idx(e_n_distance_matrix[m:m+1,:])
            
            eG.add_nodes_from([(nodefc_centers[e_n][0],{'fc':nodefc_centers[e_n][1],'Odist':nodefc_centers[e_n][2]})])
            eG.add_edge('E'+str(edgefc_centers[m][0]),nodefc_centers[e_n][0],type='real')
            e_n_distance_matrix[m,e_n] = 1000
        _, sec_round_n = locate_min_idx(e_n_distance_matrix[m:m+1,:])
        if e_n_distance_matrix[m,sec_round_n] < 0.6:
            _,e_k = locate_min_idx(e_n_distance_matrix[m:m+1,:])
            eG.add_nodes_from([(nodefc_centers[e_k][0],{'fc':nodefc_centers[e_k][1],'Odist':nodefc_centers[e_k][2]})])
            eG.add_edge('E'+str(edgefc_centers[m][0]),nodefc_centers[e_k][0],type='real')

       

    #add pillar nodes virtual edges
    n_n_distance_matrix = np.zeros((len(nodefc_centers),len(nodefc_centers)))
    for i in range(len(nodefc_centers)):
        for j in range(len(nodefc_centers)):
            n_n_distance_matrix[i,j] = np.linalg.norm(nodefc_centers[i][1] - nodefc_centers[j][1])
        n_n_distance_matrix[i,i] = 1000
    #use hungrain algorithm to find the shortest path between all nodes
    

    for i in range(len(nodefc_centers)):
        n_n_min_distance = np.min(n_n_distance_matrix[i:i+1,:])
        if n_n_min_distance < bridge_node_distance:
            n_i,n_j = locate_min_idx(n_n_distance_matrix[i:i+1,:])
            eG.add_nodes_from([(nodefc_centers[i][0],{'fc':nodefc_centers[i][1],'Odist':nodefc_centers[i][2]})])
            eG.add_nodes_from([(nodefc_centers[n_j][0],{'fc':nodefc_centers[n_j][1],'Odist':nodefc_centers[n_j][2]})])
            eG.add_edge(nodefc_centers[i][0],nodefc_centers[n_j][0],type='virtual')
    return eG






def filter_connected_node_loose(bare_nodeedge_fc_loose,boundary_node_res_loose,linker_topics,sc_unit_cell):
    renode_fcarr=reindex_frag_array(bare_nodeedge_fc_loose,'NODE')
    reedge_fcarr=reindex_frag_array(bare_nodeedge_fc_loose,'EDGE')

    edgefc_centers = get_frag_centers_fc(reedge_fcarr)
    nodefc_centers = get_frag_centers_fc(renode_fcarr)

    eG = calculate_eG_net(edgefc_centers,nodefc_centers,linker_topics,sc_unit_cell)
    #nodes_indices = [i+1 for i in range(nodefc_centers[-1][0])]
    connected_nodes_indices=[i for i in list(eG.nodes) if isinstance(i,int)] # the name of nodes in eG is int, edges' are Exx (str)
    #isolated_nodes_indices = [i for i in nodes_indices if i not in connected_nodes_indices]

    mask = np.isin(renode_fcarr[:, 5], connected_nodes_indices)
    connected_nodes_fcarr = renode_fcarr[mask]
    connected_nodeedge_fc_loose = np.vstack((connected_nodes_fcarr,reedge_fcarr)) ##

    nodes_fc=bare_nodeedge_fc_loose[bare_nodeedge_fc_loose[:,4]=='NODE']
    mask_boundary_nodefc=np.isin(nodes_fc[:,5],boundary_node_res_loose)

    boundary_connected_nodes_res=np.unique(renode_fcarr[mask_boundary_nodefc & mask][:,5]) ##

    return connected_nodeedge_fc_loose, boundary_connected_nodes_res,eG