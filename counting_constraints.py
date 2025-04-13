from gurobipy import GRB, GurobiError
from loader import DataLoader
from Models.Matheuristic import Matheuristic
import os
import psutil
import time


instance_base_path = 'Datasets\\'
instances = ['GS8']


def run_heuristic(instance_base_path, instances):
    for instance in instances:
        path = instance_base_path + instance
        loader = DataLoader(path)
        data = loader.data
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)
        model = Matheuristic(data)
        print(f"-----------Running model for instance {instance}----------------------------------")
        start_time = time.time()
        model.run()
        end_time = time.time() - start_time
        mem_after = process.memory_info().rss / (1024 * 1024)
        mem_usage = mem_after - mem_before
        print(f"CPU time: {end_time}")
        print(f"Mem Usage: {mem_usage}")

if __name__=='__main__':
    run_heuristic(instance_base_path, instances)