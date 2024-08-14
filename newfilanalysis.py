import os
import time
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import streamlit as st
from scipy.spatial import KDTree
import pandas as pd
from scipy.spatial.distance import cdist
from pymongo import MongoClient
from icp import icp  # Assumes this is a custom module
from reference_data import gcoderead  # Assumes this is a custom module

# MongoDB connection setup
connection_string = "mongodb+srv://poorvapatil426:PdNL6ytcsbH4dNuk@cluster0.wtyebsm.mongodb.net/"
client = MongoClient(connection_string)
db = client['PrintPulse']

# Function to fetch JSON data from MongoDB
def fetch_json_data(gcode_file, feed_factor, batch_size=100):
    collection_name = f"{gcode_file}_{feed_factor}"
    collection = db[collection_name]
    
    X_mea = []
    Y_mea = []
    Z_mea = []
    
    cursor = collection.find().batch_size(batch_size)
    for document in cursor:
        try:
            X_mea.append(document["X_mea"])
            Y_mea.append(document["Y_mea"])
            Z_mea.append(document["Z_mea"])
        except KeyError as e:
            print(f"Missing key in document: {e}")
    
    X_mea = np.array(X_mea)
    Y_mea = np.array(Y_mea)
    Z_mea = np.array(Z_mea)

    if X_mea.size == 0 or Y_mea.size == 0 or Z_mea.size == 0:
        print("Some data points are missing. Please check the collection.")
    
    return {"X_mea": X_mea, "Y_mea": Y_mea, "Z_mea": Z_mea}


# Function to find the closest points
def find_closest_points(reference_data, measured_data):
    tree = KDTree(measured_data)
    _, closest_indices = tree.query(reference_data)
    closest_points = measured_data[closest_indices]
    return closest_points

# Function to calculate the error between the reference data and the measured data
def calculate_errors(reference_data, measured_data):
    distances = cdist(reference_data, measured_data)
    euclidean_distances = np.min(distances, axis=1)
    average_error = np.mean(euclidean_distances)
    std_dev_error = np.std(euclidean_distances)
    return euclidean_distances, average_error, std_dev_error

# Main function
def main():
    st.title("PrintPulse")

    # File upload for G-code
    uploaded_file = st.file_uploader("Upload the G-code file", type=["gcode"])
    filename = st.text_input("Enter a name for the G-code file (without extension):")
    feedfactor = st.text_input("Enter the feed factor:")

    if uploaded_file and filename and feedfactor:
        # Save the uploaded G-code file with the user-provided filename
        gcode_file = f"{filename}.gcode"
        with open(gcode_file, "wb") as f:
            f.write(uploaded_file.read())
        
        try:
            json_data = fetch_json_data(filename, feedfactor)  # Use the provided filename

            if json_data["X_mea"].size > 0:
                # Read reference data from the saved G-code file
                ref_data = gcoderead(gcode_file)
                Z_ref_distinct = np.unique(ref_data['Z_ref'].astype(float))

                Z_mea_distinct = np.unique(json_data["Z_mea"])

                std_dev_errors = []
                average_errors = []
                ucl_values = []
                lcl_values = []

                # Placeholders for the graphs
                graph_placeholder = st.empty()
                avg_error_chart_placeholder = st.empty()
                std_dev_chart_placeholder = st.empty()

                for Z_ref, Z_mea in zip(Z_ref_distinct, Z_mea_distinct):
                    if 0.00 <= Z_mea <= 0.02:
                       continue

                    measured_data = np.vstack((json_data["X_mea"], json_data["Y_mea"])).T
                    reference_data = np.vstack((ref_data[ref_data['Z_ref'] == Z_ref]['X_ref'].values,
                                                 ref_data[ref_data['Z_ref'] == Z_ref]['Y_ref'].values)).T

                    reference_centroid = np.mean(reference_data, axis=0)
                    measured_centroid = np.mean(measured_data, axis=0)
                    translated_measured_data = measured_data - measured_centroid + reference_centroid
                    transformed_data, _, _ = icp(reference_data, translated_measured_data)
                    measured = find_closest_points(reference_data, transformed_data)

                    # Plot Reference vs. Measured Layer
                    plt.figure()
                    plt.plot(reference_data[:, 0], reference_data[:, 1], marker='o', linestyle='-', color='blue', label=f'Reference Z={Z_ref}')
                    plt.plot(measured[:, 0], measured[:, 1], marker='.', linestyle='-', color='green', label=f'Measured Z={Z_ref}')
                    plt.xlabel('X')
                    plt.ylabel('Y')
                    plt.legend()
                    plt.title(f'Reference vs. Measured Layer for Z={Z_ref}')
                    plt.axis('equal')

                    graph_placeholder.pyplot(plt)
                    plt.close()  # Close the figure after rendering

                    # Calculate Errors
                    error, average_error, std_dev_error = calculate_errors(reference_data, measured)
                    average_errors.append(average_error)
                    std_dev_errors.append(std_dev_error)

                    # Calculate UCL and LCL based on the standard deviations
                    if len(average_errors) > 1:
                        mean_error = np.mean(average_errors)
                        std_dev_of_errors = np.std(average_errors)
                        ucl = mean_error + 3 * std_dev_of_errors
                        lcl = mean_error - 3 * std_dev_of_errors

                        ucl_values.append(ucl)
                        lcl_values.append(lcl)
                    else:
                        ucl_values.append(np.nan)
                        lcl_values.append(np.nan)

                    # Save transformed data
                    transformed_folder_name = 'transformed'
                    os.makedirs(transformed_folder_name, exist_ok=True)
                    transformed_file = os.path.join(transformed_folder_name, f'{Z_ref}.csv')
                    np.savetxt(transformed_file, np.column_stack((measured, np.full((len(measured), 1), Z_ref))), delimiter=',')

                    # Add a small delay to increase frame rate
                    time.sleep(0.1)

                    # Update charts in real-time
                    error_data = {
                        'Z': Z_ref_distinct[:len(average_errors)],
                        'Average Error': average_errors,
                        'Standard Deviation Error': std_dev_errors,
                        'UCL': ucl_values,
                        'LCL': lcl_values
                    }

                    avg_error_chart = alt.Chart(pd.DataFrame(error_data)).mark_line().encode(
                        x='Z',
                        y='Average Error'
                    ).properties(
                        title='Average Errors Over Z'
                    )

                    ucl_chart = alt.Chart(pd.DataFrame(error_data)).mark_line(color='red', strokeDash=[5, 5]).encode(
                        x='Z',
                        y='UCL'
                    ).properties(
                        title='UCL'
                    )

                    lcl_chart = alt.Chart(pd.DataFrame(error_data)).mark_line(color='blue', strokeDash=[5, 5]).encode(
                        x='Z',
                        y='LCL'
                    ).properties(
                        title='LCL'
                    )

                    avg_error_chart_placeholder.altair_chart(avg_error_chart + ucl_chart + lcl_chart, use_container_width=True)

                    std_dev_chart = alt.Chart(pd.DataFrame(error_data)).mark_line().encode(
                        x='Z',
                        y='Standard Deviation Error'
                    ).properties(
                        title='Standard Deviation of Errors Over Z'
                    )

                    std_dev_chart_placeholder.altair_chart(std_dev_chart, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
