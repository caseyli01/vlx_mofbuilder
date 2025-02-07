# Flag to ignore all errors during execution
IGNORE_ALL_ERRORS = False

# Flag to enable or disable printing of debug information
PRINT = False

# Default bond length for connection sites
CONNECTION_SITE_BOND_LENGTH = 1.54

# Flag to enable or disable writing of check files
WRITE_CHECK_FILES = False

# Flag to enable or disable writing of CIF files
WRITE_CIF = True

# Flag to enable or disable all node combinations
ALL_NODE_COMBINATIONS = False

# Flag to enable or disable user-specified node assignment
USER_SPECIFIED_NODE_ASSIGNMENT = False

# Flag to enable or disable combinatorial edge assignment
COMBINATORIAL_EDGE_ASSIGNMENT = True

# Flag to enable or disable charges
CHARGES = True

# Symmetry tolerance values for different coordination numbers
SYMMETRY_TOL = {2:0.10, 3:0.19, 4:0.35, 5:0.25, 6:0.46, 7:0.35, 8:0.41, 9:0.60, 10:0.60, 12:0.60}

# Bond tolerance value
BOND_TOL = 5.0

# Flag to enable or disable orientation-dependent nodes
ORIENTATION_DEPENDENT_NODES = False

# Flag to enable or disable placing edges between connection points
PLACE_EDGES_BETWEEN_CONNECTION_POINTS = True

# Flag to enable or disable recording of callback results
RECORD_CALLBACK = False

# Flag to enable or disable output of scaling data
OUTPUT_SCALING_DATA = True

# Fixed unit cell parameters (0 means not fixed, 1 means fixed)
FIX_UC = (0,0,0,0,0,0)

# Minimum cell length
MIN_CELL_LENGTH = 5.0

# Optimization method to use ('L-BFGS-B' or 'differential_evolution')
OPT_METHOD = 'L-BFGS-B'

# Pre-scaling factor
PRE_SCALE = 1.00

# Number of scaling iterations
SCALING_ITERATIONS = 1

# Flag to enable or disable single metal MOFs only
SINGLE_METAL_MOFS_ONLY = True

# Flag to enable or disable MOFs only
MOFS_ONLY = True

# Flag to enable or disable merging of catenated nets
MERGE_CATENATED_NETS = True

# Flag to enable or disable parallel execution
RUN_PARALLEL = False

# Flag to enable or disable removal of dummy atoms
REMOVE_DUMMY_ATOMS = True
