from gurobipy import GRB, GurobiError
from loader import DataLoader
from Models.MBLP import MBLP
from Models.Matheuristic import Matheuristic
from Models.alternative_MBLP import AlternativeMBLP


models = {'MBLP': MBLP, 'alternative MBLP': AlternativeMBLP, 'Matheuristic': Matheuristic}

# Run the three models on the toy dataset presented in the article as a toy example
def run_toy_test():
    instance_name = 'toy_dataset'
    instance = 'Datasets\\' + instance_name
    loader = DataLoader(instance)
    data = loader.data

    for model_name, model in models.items():
        if model_name == 'Matheuristic':
            model = model(data, k=2)
            print(f"-------------------------Running model {model_name} on instance {instance_name}------------------------------\n")
            model.run()
        else:
            model = model(data)
            print(f"-------------------------Running model {model_name} on instance {instance_name}------------------------------\n")
            model.run()

# Rune the three models on two small datasets
def run_small_test():
    instances = ['GS1', 'GS3']

    for instance_name in instances:
        path = 'Datasets\\' + instance_name
        loader = DataLoader(path)
        data = loader.data

        for model_name, model in models.items():
            model = model(data)
            print(f"-------------------------Running model {model_name} on instance {instance_name}------------------------------\n")
            model.run()



if __name__=='__main__':
    # toy dataset test total running time ~1 sec
    run_toy_test()

    # small test total running time ~35 sec
    # run_small_test()
