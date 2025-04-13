from loader import DataLoader
from Models.MBLP import MBLP
from Models.alternative_MBLP import AlternativeMBLP
from Models.Matheuristic import Matheuristic
import os
import time
import psutil


def run():
    while True:
        file_path = input(
            "Enter the path to the data folder (e.g., 'Datasets\\GS1'): "
        ).strip()
        if os.path.exists(file_path):
            break
        print("Invalid input path. Please try again.")

    print("Loading data...")
    dataloader = DataLoader(file_path)
    data = dataloader.data

    visualize_data = input("Do you want to display the processed data? y/n: ").strip().lower()
    if visualize_data == "y":
        for key, value in dataloader.data.items():
            print(f"{key}: {value}")
    elif visualize_data != "n":
        print("Invalid input. Continuing without visualization.")

    valid_models = ['matheuristic', 'mblp', 'alternative mblp']
    while True:
        model_name = input(f"Choose a model ({', '.join(valid_models)}): ").strip().lower()
        if model_name in valid_models:
            break
        else:
            print("Invalid model name. Please choose from: " + ", ".join(valid_models))

    if model_name == 'matheuristic':
        model = Matheuristic(data)
    elif model_name == 'mblp':
        model = MBLP(data)
    elif model_name == 'alternative mblp':
        model = AlternativeMBLP(data)
    else:
        print("Unexpected error: invalid model name.")
        return

    print(f"\nRunning model: {model_name}")
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)
    start_time = time.time()
    model.run()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nModel execution completed in {elapsed_time:.2f} seconds.")
    mem_after = process.memory_info().rss / (1024 * 1024)
    print(f"Memory usage increased by: {mem_after - mem_before:.2f} MB")


if __name__ == "__main__":
    run()