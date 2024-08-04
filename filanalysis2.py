import pickle
from pathlib import Path
# import streamlit_authenticator as stauth
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from icp import icp  # Assumes this is a custom module
from reference_data import gcoderead  # Assumes this is a custom module
from readfile import readfile  # Assumes this is a custom module

#-------user authentiaction----------
# names=["peter parker","rebecca miller"]
# usernames=["pparker","rmiller"]

# file_path=Path(__file__).parent / "hashed_pw.pkl"
# with file_path.open("wb") as file:
#     hashed_passwords=pickle.load(file)

# authenticator=stauth.Authenticate(names,usernames,hashed_passwords,"sales_dashboard","abcdef",cookie_expiry_days=30)    

# name,authentication_status,username=authenticator.login("Login","main")

# if authentication_status==False:
#     st.error("Username/password ids incorrect")

# if authentication_status==None:
#     st.warning("Please enter your username and password")

# if authentication_status:

API_BASE_URL = 'http://10.202.1.194:6500'
API_KEY = '8C90C88D457C46FE95F35326E2AA335A'

# Function to find the closest points
def find_closest_points(reference_data, measured_data):
    tree = KDTree(measured_data)
    _, closest_indices = tree.query(reference_data)
    closest_points = measured_data[closest_indices]
    return closest_points

# Function to calculate the error between the reference data and the measured data
def analysis(reference_data, measured_data):
    distances = cdist(reference_data, measured_data)
    euclidean_distances = np.diagonal(distances)
    average_error = np.mean(euclidean_distances)
    std_dev_error = np.std(euclidean_distances)
    return euclidean_distances, average_error, std_dev_error

# Main function
#authenticator.logout("Logout")
#st.sidebar.title(f"Welcome {name}")
def main():
    st.title("PrintPulse")

    filename = st.text_input("Enter the file name:")
    if os.path.exists('C:/Users/Asus/OneDrive - IIT Indore/Realtime Monitoring/Final/Done_Analysis'):
      folder_path = 'C:/Users/Asus/OneDrive - IIT Indore/Realtime Monitoring/Final/Done_Analysis'
    else:
      folder_path = 'C:/Users/Dell PC/OneDrive - IIT Indore/Ajinkya Kulkarni/Realtime Monitoring/Final/Done_Analysis'

    if filename:
        if filename == "Curve_Aerofoil":
            selected_path = 'C:/Users/Asus/OneDrive - IIT Indore/Realtime Monitoring/Final/Done_Analysis/Curve_Airfoil/50%/Curve_Airfoil'
            feedfactor = "50%"
        elif filename == "Curve_Aerofoil_shell":
            selected_path = 'C:/Users/Asus/OneDrive - IIT Indore/Realtime Monitoring/Final/Done_Analysis/Curve_Aerofoil_shell'
            feedfactor = None
        else:
            possible_feed_factors = ["50%", "70%", "75%", "100%"]
            available_feedfactors = [f for f in possible_feed_factors if os.path.exists(os.path.join(folder_path, filename, f, '1', filename))]
            if not available_feedfactors:
                st.toast(f"No valid feed factor folders found for {filename}.")
                return
            feedfactor = st.selectbox("Select feedfactor:", available_feedfactors)

            selected_path = os.path.join(folder_path, filename, feedfactor, '1', filename)

            if not os.path.exists(selected_path):
                st.toast(f"Folder {selected_path} does not exist. Changing to fallback path.")
                fallback_folder = os.path.join(folder_path, "Propeller", "50%", "1", "Propeller")
                st.toast(f"Fallback path: {fallback_folder}")
                selected_path = fallback_folder

        ref_folder_name = 'gcode'
        gcode_file = os.path.join(ref_folder_name, f'{filename}.gcode')

        if not os.path.exists(gcode_file):
            st.toast(f"File {gcode_file} does not exist.")
            return

        ref_data = gcoderead(gcode_file)
        Z_ref_distinct = ref_data['Z_ref'].astype(float).unique().tolist()

        available_files = [f.replace('.csv', '') for f in os.listdir(selected_path) if f.endswith('.csv')]
        Z_mea_distinct = sorted([float(z) for z in available_files])

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

            measured_data = readfile(f'{Z_mea}', selected_path)[['X_mea', 'Y_mea']].to_numpy()
            reference_data = ref_data[ref_data['Z_ref'] == Z_ref][['X_ref', 'Y_ref']].to_numpy()

            reference_centroid = np.mean(reference_data, axis=0)
            measured_centroid = np.mean(measured_data, axis=0)
            translated_measured_data = measured_data - measured_centroid + reference_centroid
            transformed_data, _, _ = icp(reference_data, translated_measured_data)
            measured = find_closest_points(reference_data, transformed_data)

            # Plot Reference vs. Measured Layer
            plt.figure()
            plt.plot(reference_data[:, 0], reference_data[:, 1], marker='o', linestyle='-', color='blue', label=f'Reference Z={Z_ref}')
            plt.plot(measured[:, 0], measured[:, 1], marker='.', linestyle='-', color='green', label=f'Measured Z={Z_mea}')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()
            plt.title(f'Reference vs. Measured Layer for Z={Z_ref}')
            plt.axis('equal')

            graph_placeholder.pyplot(plt)
            plt.close()  # Close the figure after rendering

            # Calculate Errors
            error, average_error, std_dev_error = analysis(reference_data, measured)
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
            transformed_file = os.path.join(transformed_folder_name, f'{Z_mea}.csv')
            np.savetxt(transformed_file, np.column_stack((measured, np.full((len(measured), 1), Z_ref))), delimiter=',')

            # Add a small delay to increase frame rate
            time.sleep(0.1)

            # Update charts in real-time
            error_data = pd.DataFrame({
                'Z': Z_mea_distinct[:len(average_errors)],
                'Average Error': average_errors,
                'Standard Deviation Error': std_dev_errors,
                'UCL': ucl_values,
                'LCL': lcl_values
            })

            avg_error_chart = alt.Chart(error_data).mark_line().encode(
                x='Z',
                y='Average Error'
            ).properties(
                title='Average Errors Over Z'
            )

            ucl_chart = alt.Chart(error_data).mark_line(color='red', strokeDash=[5, 5]).encode(
                x='Z',
                y='UCL'
            ).properties(
                title='UCL'
            )

            lcl_chart = alt.Chart(error_data).mark_line(color='blue', strokeDash=[5, 5]).encode(
                x='Z',
                y='LCL'
            ).properties(
                title='LCL'
            )

            avg_error_chart_placeholder.altair_chart(avg_error_chart + ucl_chart + lcl_chart, use_container_width=True)

            std_dev_chart = alt.Chart(error_data).mark_line().encode(
                x='Z',
                y='Standard Deviation Error'
            ).properties(
                title='Standard Deviation of Errors Over Z'
            )

            std_dev_chart_placeholder.altair_chart(std_dev_chart, use_container_width=True)

if __name__ == '__main__':
    main()
