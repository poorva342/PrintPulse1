import os
from reference_data import gcoderead
from readfile import readfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

def icp_translation_only(source_points, target_points, max_iterations=50, tolerance=0.001):
    """
    Perform Iterative Closest Point algorithm to align source_points to target_points (translation only).

    Args:
    - source_points (np.array): Array of shape (N, 2) representing 2D points in the source point cloud.
    - target_points (np.array): Array of shape (N, 2) representing 2D points in the target point cloud.
    - max_iterations (int): Maximum number of iterations to run the algorithm.
    - tolerance (float): Convergence threshold.

    Returns:
    - t (np.array): Translation vector.
    """

    # Initialize translation vector
    t = np.zeros(2)  # Zero translation vector

    for iteration in range(max_iterations):
        # Find closest points in target cloud for each point in the source cloud
        tree = KDTree(target_points)
        closest_indices = tree.query(source_points)[1]
        closest_points = target_points[closest_indices]

        # Compute centroids of both point clouds
        centroid_source = np.mean(source_points, axis=0)
        centroid_target = np.mean(closest_points, axis=0)

        # Compute translation vector
        t_iteration = centroid_target - centroid_source

        # Update translation
        t += t_iteration
        
        if np.linalg.norm(t_iteration) < tolerance:
            break

    return t


def rmse(source_points, target_points):
    """
    Compute the Root Mean Square Error (RMSE) between corresponding points in source and target point clouds.

    Args:
    - source_points (np.array): Array of shape (N, 2) representing 2D points in the source point cloud.
    - target_points (np.array): Array of shape (N, 2) representing 2D points in the target point cloud.

    Returns:
    - rmse (float): Root Mean Square Error.
    """
    # Compute KDTree for target points
    tree = KDTree(target_points)
    
    # Query nearest neighbors for each point in source points
    _, indices = tree.query(source_points)
    
    # Compute RMSE based on nearest neighbors
    rmse = np.sqrt(np.mean(np.sum((target_points[indices] - source_points)**2, axis=1)))
    
    return rmse


def find_best_alignment(source_points, target_points, max_iterations_range):
    """
    Find the combination of max_iterations that gives the best alignment between source and target point clouds.

    Args:
    - source_points (np.array): Array of shape (N, 2) representing 2D points in the source point cloud.
    - target_points (np.array): Array of shape (N, 2) representing 2D points in the target point cloud.
    - max_iterations_range (list): List of integers representing the range of max_iterations values to try.

    Returns:
    - best_alignment (dict): Dictionary containing the best alignment parameters and corresponding RMSE.
    - rmse_values (list): List of RMSE values for each combination of max_iterations.
    """
    best_rmse = float('inf')
    best_alignment = {'max_iterations': None, 'tolerance': None, 'translation': None}
    rmse_values = []

    for max_iter in max_iterations_range:
        translation = icp_translation_only(source_points, target_points, max_iterations=max_iter)
        transformed_data = source_points + translation
        alignment_rmse = rmse(transformed_data, target_points)
        rmse_values.append(alignment_rmse)
        if alignment_rmse < best_rmse:
            best_rmse = alignment_rmse
            best_alignment['max_iterations'] = max_iter
            best_alignment['translation'] = translation

    return best_alignment, rmse_values


def icp(reference_data, measured_data, max_iterations_range = range(1, 50)):
    # all_float = np.all([isinstance(element, float) for element in reference_data.flatten()])
    # print(all_float)

    reference_data = reference_data.astype(float)

    # all_float = np.all([isinstance(element, float) for element in measured_data.flatten()])
    # print(all_float)

    # Find the best alignment parameters
    best_alignment, rmse_values = find_best_alignment(measured_data, reference_data, max_iterations_range)
    # print("Best alignment parameters:")
    # print("Max Iterations:", best_alignment['max_iterations'])
    # print("RMSE:", rmse(measured_data + best_alignment['translation'], reference_data))

    # Apply translation using the best alignment parameters
    transformed_data = measured_data + best_alignment['translation']

    return transformed_data, rmse_values, max_iterations_range


def main():
    dir_path = 'C:\\Users\\Dell PC\\Desktop\\Realtime Monitoring\\Final'
    filename = input("Enter the file name: ")
    ref_folder_name = 'gcode'
    gcode_file = os.path.join(dir_path, ref_folder_name, filename + '.gcode')
    ref_data = gcoderead(gcode_file)
    Z_ref_distinct = ref_data['Z_ref'].astype(float).unique().tolist()
    print(Z_ref_distinct)

    available_files = os.listdir(os.path.join(dir_path,filename))
    for file in available_files:
        if file.endswith('.csv'):
            available_files = [file.replace('.csv', '') for file in available_files]
    Z_mea_distinct = [float(elements) for elements in available_files]
    print(Z_mea_distinct)

    Z_ref = input("Enter the reference layer: ")
    Z_mea = input("Enter the measured layer: ")

    reference_data = ref_data[ref_data['Z_ref'] == float(Z_ref)][['X_ref', 'Y_ref']].to_numpy()
    measured_data = readfile(Z_mea, os.path.join(dir_path,filename))[['X_mea', 'Y_mea']].to_numpy()
    transformed_data, rmse_values, max_iterations_range = icp(reference_data, measured_data)

    # Plot reference_data and transformed_data
    plt.plot(reference_data[:, 0], reference_data[:, 1], label='Reference Data', marker='o', linestyle='-', color='blue')
    plt.plot(transformed_data[:, 0], transformed_data[:, 1], label='Transformed Data', marker='x', linestyle='-', color='red')
    # plt.plot(measured_data[:, 0], measured_data[:, 1], label='Measured Data', marker='x', linestyle='-', color='green')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.axis('equal')
    plt.show()

    # Plot RMSE vs Max Iterations
    plt.plot(max_iterations_range, rmse_values, marker='o', linestyle='-')
    plt.title('RMSE vs Max Iterations')
    plt.xlabel('Max Iterations')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.show()
    min_rmse = min(rmse_values)
    min_rmse_index = rmse_values.index(min_rmse)
    print("Minimum RMSE:", min_rmse)
    print("Iterations:", min_rmse_index + 1)
if __name__ == '__main__':
    main()