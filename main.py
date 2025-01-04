import os
from loader import DataLoader
from MBLP import MBLP
from alternative_MBLP import AlternativeMBLP
from Matheuristic import Matheuristic


def run():
    # User inputs path to dataset
    file_path = input("Enter the path to the data folder (e.g., 'C:\\Users\\User\\Desktop\\MathOpt\\Data\\GS1'): ")

    # Data loading
    if os.path.exists(file_path):
        print("Loading data...")
        dataloader = DataLoader(file_path)
        data = dataloader.data  # Access loaded data
    else:
        print("Invalid input path.")
        return

    # Prompt the user to select a model to run
    valid_models = ['matheuristic', 'mblp', 'alternative mblp']
    model_name = None
    while model_name not in valid_models:
        model_name = input(f"Choose a model ({', '.join(valid_models)}): ").strip().lower()
        if model_name not in valid_models:
            print("Invalid model name. Please choose from matheuristic, mblp, alternative mblp.")

    # Initialize the selected model
    if model_name == 'matheuristic':
        model = Matheuristic(data)
    elif model_name == 'mblp':
        model = MBLP(data)
    elif model_name == 'alternative mblp':
        model = AlternativeMBLP(data)
    else:
        print("Unexpected error: invalid model name.")
        return

    # Run the selected model
    print(f"\nRunning model: {model_name}")
    model.run()

    # Indicate completion
    print("\nModel execution completed.")

if __name__ == "__main__":
    run()
