import numpy as np
from scipy.optimize import minimize
from _place_node_edge import unit_cell_to_cartesian_matrix, cartesian_to_fractional


#scale optimizer for the cif parameters update 
def scale_objective_function(params, old_cell_params, old_cartesian_coords, new_cartesian_coords):
    a_new, b_new, c_new, _,_,_ = params
    a_old, b_old, c_old, alpha_old, beta_old, gamma_old = old_cell_params

    # Compute transformation matrix for the old unit cell, T is the unit cell matrix
    T_old = unit_cell_to_cartesian_matrix(a_old, b_old, c_old, alpha_old, beta_old, gamma_old)
    T_old_inv= np.linalg.inv(T_old)
    old_fractional_coords = cartesian_to_fractional(old_cartesian_coords,T_old_inv)

    #backup
    #old_fractional_coords = cartesian_to_fractional(old_cartesian_coords,T_old_inv)

    # Compute transformation matrix for the new unit cell
    T_new = unit_cell_to_cartesian_matrix(a_new, b_new, c_new, alpha_old, beta_old, gamma_old)
    T_new_inv = np.linalg.inv(T_new)

    # Convert the new Cartesian coordinates to fractional coordinate using the old unit cell
    

    # Recalculate fractional coordinates from updated Cartesian coordinates
    new_fractional_coords = cartesian_to_fractional(new_cartesian_coords,T_new_inv)
    
    # Compute difference from original fractional coordinates
    diff = new_fractional_coords - old_fractional_coords
    return np.sum(diff**2)  # Sum of squared differences
    

# Example usage
def optimize_cell_parameters(cell_info,original_ccoords,updated_ccoords):
    # Old cell parameters (example values)
    old_cell_params = cell_info  # [a, b, c, alpha, beta, gamma]

    # Old Cartesian coordinates of points (example values)
    old_cartesian_coords = np.vstack(list(original_ccoords.values()))  # original_ccoords

    # New Cartesian coordinates of the same points (example values)
    new_cartesian_coords = np.vstack(list(updated_ccoords.values()))  # updated_ccoords
    # Initial guess for new unit cell parameters (e.g., slightly modified cell)
    initial_params = cell_info

    # Bounds: a, b, c > 3; angles [0, 180]
    bounds = [(3, None),(3, None),(3, None)] + [(20, 180)] * 3

    # Optimize using L-BFGS-B to minimize the objective function
    result = minimize(
        scale_objective_function,
        x0=initial_params,
        args=(old_cell_params, old_cartesian_coords, new_cartesian_coords),
        method='L-BFGS-B',
        bounds=bounds
    )

    # Extract optimized parameters
    optimized_params = np.round(result.x,5)
    print("Optimized New Cell Parameters:", optimized_params,"\nTemplate Cell Parameters:",cell_info)

    return optimized_params


