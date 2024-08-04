import os
import pandas as pd

def readfile(file_name, measured_folder_name):
    if not file_name.endswith('.csv'):
        file_name += '.csv'
    measured_file = os.path.join(measured_folder_name, file_name)
    measured_data = pd.read_csv(measured_file, usecols=['timestamp', 'X_mea', 'Y_mea', 'Z_mea'])
    return measured_data


def main():
    measured_folder_name = input("Enter measured folder name: ")
    file_name = input("Enter the filename: ")
    measured_data = readfile(file_name, measured_folder_name)
    print(measured_data)

if __name__ == "__main__":
    main() 