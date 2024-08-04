import os
import matplotlib.pyplot as plt
from reference_data import gcoderead
from readfile import readfile
from scipy.spatial.distance import cdist
import numpy as np
from icp import icp
from scipy.spatial import KDTree


# Current method to caculate the closest points
def find_closest_points(reference_data, measured_data):
    # Build a KD tree for the measured data
    tree = KDTree(measured_data)

    # Query the KD tree to find the nearest neighbor for each point in the reference data
    _, closest_indices = tree.query(reference_data)
    
    # Create a new array with the closest points from measured data
    closest_points = measured_data[closest_indices]

    # Calculate errors for each point
    distances = np.linalg.norm(reference_data - closest_points, axis=1)
    
    # Calculate average and standard deviation of errors
    average_error = np.mean(distances)
    std_dev_error = np.std(distances)

    # print(f'Average Error: {average_error}')
    # print(f'Standard Deviation of Errors: {std_dev_error}')

    return closest_points

# Function to calculate the error between the reference data and the measured data
def analysis(reference_data, measured_data):
    distances = cdist(reference_data, measured_data)
    euclidean_distances = np.diagonal(distances)
    average_error = np.mean(euclidean_distances)
    std_dev_error = np.std(euclidean_distances)
    return euclidean_distances, average_error, std_dev_error

def layer_analysis(reference_data, measured_data):
    reference_centroid = np.mean(reference_data, axis=0)
    measured_centroid = np.mean(measured_data, axis=0)
    translated_measured_data = measured_data - measured_centroid + reference_centroid
    transformed_data, rmse_values, max_iterations_range = icp(reference_data, translated_measured_data)
    measured = find_closest_points(reference_data, transformed_data)
    error, average_error, std_dev_error = analysis(reference_data, measured)

    # plt.plot(range(len(reference_data)), error, label='Error', marker='o', linestyle='-', color='purple')
    # plt.xlabel('Reference Data Index')
    # plt.ylabel('Error')
    # plt.legend()
    # plt.show()

    return average_error, std_dev_error


def main():
    filename = input("Enter the file name: ")
    ref_folder_name = 'gcode'
    gcode_file = os.path.join(ref_folder_name, filename + '.gcode')
    ref_data = gcoderead(gcode_file)
    Z_ref_distinct = ref_data['Z_ref'].astype(float).unique().tolist()
    print(Z_ref_distinct)
    
    available_files = os.listdir(filename)
    for file in available_files:
        if file.endswith('.csv'):
            available_files = [file.replace('.csv', '') for file in available_files]
    Z_mea_distinct = [float(elements) for elements in available_files]
    print(Z_mea_distinct)
    
    Z_ref = float(input("Enter the reference layer: "))
    Z_mea = input("Enter the measured layer: ")
    
    measured_data = readfile(f'{Z_mea}', filename)[['X_mea', 'Y_mea']].to_numpy()
    reference_data = ref_data[ref_data['Z_ref'] == Z_ref][['X_ref', 'Y_ref']].to_numpy()
    average_error, std_dev_error = layer_analysis(reference_data, measured_data)
    print(f'Average Error: {average_error}')
    print(f'Standard Deviation Error: {std_dev_error}')

if __name__ == '__main__':
    main()