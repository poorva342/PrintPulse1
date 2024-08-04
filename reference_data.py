import pandas as pd
import numpy as np
from gcodeparser import GcodeParser
import os
import matplotlib.pyplot as plt

# Function to interpolate points based on Euclidean distance
def interpolate_points(x, y, z, command, distance_threshold=2):
    try:
        # Convert x, y, z to numpy arrays to ensure they are treated as floats
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)

        # Combine x and y into points array
        points = np.column_stack((x, y))

        # Calculate differences between consecutive points
        differences = np.diff(points, axis=0)

        # Calculate distances between consecutive points using Euclidean norm
        distances = np.linalg.norm(differences, axis=1)

        # Initialize interpolated data dictionary
        interpolated_data = {'X_ref': [x[0]], 'Y_ref': [y[0]], 'Z_ref': [z], 'command': [command[0]]}

        # Iterate over each point
        for i in range(1, len(x)):
            # num_interpolated_points = int(distances[i - 1])  # Number of points to interpolate based on distance
            distance = distances[i - 1]
            num_interpolated_points = int(distance / distance_threshold)  # Number of points to interpolate based on distance
            if num_interpolated_points > 0:
                # Generate interpolated x and y coordinates
                x_interp = np.linspace(x[i - 1], x[i], num=num_interpolated_points + 2)[1:-1]
                y_interp = np.linspace(y[i - 1], y[i], num=num_interpolated_points + 2)[1:-1]

                # Extend interpolated data with new points
                interpolated_data['X_ref'].extend(x_interp)
                interpolated_data['Y_ref'].extend(y_interp)
                interpolated_data['Z_ref'].extend([z] * num_interpolated_points)
                interpolated_data['command'].extend([command[i - 1]] * num_interpolated_points)

            # Append original point to interpolated data
            interpolated_data['X_ref'].append(x[i])
            interpolated_data['Y_ref'].append(y[i])
            interpolated_data['Z_ref'].append(z)
            interpolated_data['command'].append(command[i])

        # Create DataFrame from interpolated data dictionary
        return pd.DataFrame(interpolated_data)

    except Exception as e:
        print(f"Error in interpolate_points: {e}")
        print(f"x: {x}, y: {y}, z: {z}, command: {command}")
        return pd.DataFrame()  # Return an empty DataFrame if an error occurs


def gcoderead(gcodefile):
    with open(gcodefile, 'r') as f:
        gcode = f.read()
    code = GcodeParser(gcode).lines
    data = pd.DataFrame(code)

    # To convert the ('G', 0) to G0
    data['command'] = data['command'].apply(lambda x: f"{x[0]}{x[1]}")

    # Extract key-value pairs from 'params' column and save them into separate columns
    data_params = pd.json_normalize(data['params'])
    data = pd.concat([data, data_params], axis=1)

    # To only select the X Y Z and command columns form the table
    f_data = data[['command', 'X', 'Y', 'Z']]

    # Set default values as 0 and fill them to the table
    default_values = {'X': 0, 'Y': 0, 'Z': 0}
    f_data.iloc[0] = f_data.iloc[0].fillna(default_values)

    # Fill the missing values
    f_data = f_data.ffill()

    # Check for the command to be G0 or G1
    f_data = f_data[f_data['command'].isin(["G0", "G1"])]

    f_data = f_data.iloc[8:-4]

    f_data = f_data[f_data['Z'].map(f_data['Z'].value_counts()) > 4]

    # Group DataFrame by Z values
    grouped = f_data.groupby('Z')

    # Interpolate points for each group
    interpolated_dfs = []
    for name, group in grouped:
        x = group['X'].values
        y = group['Y'].values
        z = group['Z'].values[0]  # Assuming Z values are the same for each group
        command = group['command'].values
        interpolated_df = interpolate_points(x, y, z, command)
        interpolated_dfs.append(interpolated_df)

    # Concatenate the interpolated DataFrames (Reference Data with interpolation)
    reference_data = pd.concat(interpolated_dfs, ignore_index=True)
    reference_data = reference_data.rename(columns={'command': 'command', 'X': 'X_ref', 'Y': 'Y_ref', 'Z': 'Z_ref'})
    
    # For getting data without interpolation
    # reference_data = f_data.rename(columns={'X': 'X_ref', 'Y': 'Y_ref', 'Z': 'Z_ref'})
    return reference_data


def main():
    dir_path = 'C:\\Users\\Dell PC\\Desktop\\Realtime Monitoring\\Final'
    folder_name = 'gcode'
    filename = input("Enter G-code file name: ")
    gcodefile = os.path.join(dir_path, folder_name, filename+'.gcode')
    reference_data = gcoderead(gcodefile)

    # Ensure 'Z_ref' column contains scalar values (e.g., float or integer)
    reference_data['Z_ref'] = reference_data['Z_ref'].astype(float)  # Convert to float if necessary

    # Print a sample of reference_data to inspect the 'Z_ref' column
    print(reference_data)

    # Plot x_ref, y_ref for every group of z_ref
    for name, group in reference_data.groupby('Z_ref'):
        plt.figure()
        plt.plot(group['X_ref'], group['Y_ref'], label=f'Z_ref = {name}', marker='o', linestyle='-', color='blue')
        plt.xlabel('X_ref')
        plt.ylabel('Y_ref')
        plt.legend()
        plt.axis('equal')
        plt.show()

    plt.figure(figsize=(10, 8))

    ax = plt.axes(projection='3d')

    # Data for a three-dimensional line
    zdata = reference_data['Z_ref']
    xdata = reference_data['X_ref']
    ydata = reference_data['Y_ref']

    ax.plot3D(xdata, ydata, zdata, 'gray')
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    main()