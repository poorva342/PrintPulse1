import os
import csv
import pandas as pd
import matplotlib.pyplot as plt

folder_path = 'C:\\Users\\Dell PC\\Desktop\\Realtime Monitoring\\Final'  # Replace with the actual folder path
file_name = input("Enter the filename: ")

csv_files = [file for file in os.listdir(os.path.join(folder_path, file_name)) if file.endswith('.csv')]

X_mea = []
Y_mea = []
Z_mea = []

for csv_file in csv_files:
    file_path = os.path.join(folder_path, file_name, csv_file)
    with open(file_path, 'r') as file:

        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the first line of the CSV file
        for row in csv_reader:
            # Process each row of the CSV file
            # Currently changed for transformed data 0, 1, 2 columns insted of 1, 2, 3 columns for reading gcode as reference data
            x = row[0]  # 2nd column
            y = row[1]  # 3rd column
            z = row[2]  # 4th column
            X_mea.append(x)
            Y_mea.append(y)
            Z_mea.append(z)

            # print(x, y, z)

df = pd.DataFrame({'X_mea': X_mea, 'Y_mea': Y_mea, 'Z_mea': Z_mea})

# Convert each element in each column to float
for column in df.columns:
    df[column] = df[column].astype(float)

# Plot the 3D line plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(df['X_mea'], df['Y_mea'], df['Z_mea'], marker= '.', color='#87CEEB', markerfacecolor='black', markeredgecolor='black', markersize=2, linestyle='--')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Increase the axis length
ax.set_xlim3d(-10, 20)
ax.set_ylim3d(-10, 20)
ax.set_zlim3d(-10, 20)

# Increase the figure size
fig.set_size_inches(5, 6)

plt.axis('equal')
plt.show()



