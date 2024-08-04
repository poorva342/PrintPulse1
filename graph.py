import pandas as pd
from matplotlib.animation import FuncAnimation
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import matplotlib.pyplot as plt

def graph():

    # Global variable to store the latest data
    latest_data = None

    # Function to update plots
    def update_plots(frame):
        nonlocal latest_data

        if latest_data is not None:
            plt.clf()  # Clear the current plot
            for i, column in enumerate(['Average Error', 'Std Dev Error'], 1):

                # Modify the subplot to have 1 column
                plt.subplot(2, 1, i)

                # Plot the data with ".-" style
                plt.plot(latest_data['Z_ref'], latest_data[column], marker='o', markerfacecolor='red', markeredgecolor='red', markersize=5, linestyle='--', color='blue')  
                plt.xlabel('Z_ref')
                plt.ylabel(column)
                plt.title(column)
                plt.tight_layout()

    # Function to handle file changes
    def on_modified(event):
        nonlocal latest_data

        if event.src_path.endswith('error_analysis.csv'):
            latest_data = pd.read_csv(event.src_path)

    # Create a figure
    fig = plt.figure(figsize=(10, 8))

    # Start the animation
    ani = FuncAnimation(fig, update_plots, frames=100, interval=1000)  # Update plot every 1000 milliseconds (1 second)

    # Set up file system event handler
    event_handler = FileSystemEventHandler()
    event_handler.on_modified = on_modified
    observer = Observer()
    observer.schedule(event_handler, '.', recursive=False)
    observer.start()

    plt.show()

def main():
    graph()

if __name__ == "__main__":
    main()
