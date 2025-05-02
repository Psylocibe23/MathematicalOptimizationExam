import os
import time
import psutil
from gurobipy import GRB, GurobiError
from loader import DataLoader
from Models.MBLP import MBLP
from Models.Matheuristic import Matheuristic
from Models.alternative_MBLP import AlternativeMBLP
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


file_name = 'results.csv'
field_names = ['Model', 'Instance', 'OFV', 'MipGap(%)', 'CPU_time(SEC)', 'Memory_usage(MB)', 'Number_constraints', 'Status']
results = []

# -------------------------------------------------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------------------------------------------------

def get_status_string(status):
    if status == GRB.OPTIMAL:
        return "Optimal"
    elif status == GRB.INFEASIBLE:
        return "Infeasible"
    elif status == GRB.UNBOUNDED:
        return "Unbounded"
    elif status == GRB.TIME_LIMIT:
        return "Time limit reached"
    elif status == GRB.INTERRUPTED:
        return "Interrupted"
    else:
        return f"Status {status}"


def update_results_csv(new_row, file_name, field_names):
    """
    After the model results are collected, updates the CSV file.
    new_row: a dictionary with keys that should match the field_names
    """
    dtype_dict = {
        'Model': str,
        'Instance': str,
        'OFV': float,
        'MipGap(%)': float,
        'CPU_time(SEC)': float,
        'Memory_usage(MB)': float,
        'Number_constraints': float,
        'Status': str
    }

    # Ensure new_row has all expected fields
    for field in field_names:
        if field not in new_row:
            new_row[field] = ""

    new_row_df = pd.DataFrame([new_row]).astype(dtype_dict, errors='ignore')

    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
        condition = (df['Model'] == new_row['Model']) & (df['Instance'] == new_row['Instance'])
        if condition.any():
            # If a row for this (Model, Instance) pair already exists,
            # overwrite it by repeating our new_row for each occurrence.
            # Count the number of rows matching the condition
            count = int(condition.sum())
            # Create a DataFrame with that many identical rows from new_row_df
            repeated_df = pd.concat([new_row_df] * count, ignore_index=True)
            # Update matching rows
            df.loc[condition, :] = repeated_df.values
        else:
            df = pd.concat([df, new_row_df], ignore_index=True)
        df.to_csv(file_name, index=False)
    else:
        df = pd.DataFrame([new_row], columns=field_names).astype(dtype_dict, errors='ignore')
        df.to_csv(file_name, index=False)


# -------------------------------------------------------------------------------------------------------------------
# Results Collection Functions
# -------------------------------------------------------------------------------------------------------------------

# Define three methods to run the models on specified instances and save the results to results.csv
# I had to use this approach since running multiple models on multiple instances in a for loop ended up in unexpected
# out of memory errors
def collect_mblp_results(model_class=MBLP, instance='GS1'):
    """
        Run the MBLP or the alternative MBLP model on a specified instance and collect results on csv
    """
    instance_path = "Datasets\\" + instance
    loader = DataLoader(instance_path)
    data = loader.data

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)
    model = model_class(data)
    model.model.setParam('TimeLimit', 1800)

    try:
        start_time = time.time()
        model.run()
        status = model.model.status
    except GurobiError as e:
        if 'out of memory' in str(e).lower():
            print('Out of memory error')
            status = None
        else:
            raise

    end_time = time.time() - start_time
    mem_after = process.memory_info().rss / (1024 * 1024)
    mem_usage = mem_after - mem_before
    obj_value = model.model.ObjVal if hasattr(model.model, 'ObjVal') else None

    result_row = {
        'Model': model_class.__name__,
        'Instance': instance,
        'OFV': round(obj_value, 0) if obj_value is not None else None,
        'MipGap(%)': round(model.model.MIPGap, 2) if hasattr(model.model, 'MIPGap') else None,
        'CPU_time(SEC)': round(end_time, 2),
        'Memory_usage(MB)': round(mem_usage, 2),
        'Number_constraints': model.model.NumConstrs if hasattr(model.model, 'NumConstrs') else None,
        'Status': get_status_string(status)
    }

    update_results_csv(result_row, file_name, field_names)


def collect_heuristic_results(instance='GS1', k=20):
    """
        Run the Matheuristic on a specified instance and collect results in csv
        k is the value for the mini-batch k-means algorithm to further divide groups into subgroups
    """
    instance_path = 'Datasets\\' + instance
    loader = DataLoader(instance_path)
    data = loader.data

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)
    model = Matheuristic(data, k)

    start_time = time.time()
    model.run()
    end_time = time.time() - start_time
    mem_after = process.memory_info().rss / (1024 * 1024)
    mem_usage = mem_after - mem_before
    obj_value = model.iterative_algorithm.compute_objective_value()
    # Use the optimal value of the MBLP to compute MipGap (%)
    mblp_obj = None
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
        mblp_row = df[(df['Model'] == 'MBLP') & (df['Instance'] == instance)]
        if not mblp_row.empty:
            mblp_obj_val = mblp_row.iloc[0]['OFV']
            try:
                mblp_obj = float(str(mblp_obj_val).replace(',', ''))
            except Exception as e:
                print("Error converting MBLP OFV to float:", e)
                mblp_obj = None

    if mblp_obj is not None and obj_value is not None:
        mip_gap = abs(mblp_obj - obj_value) / abs(mblp_obj) * 100
        mip_gap = round(mip_gap, 2)
    else:
        mip_gap = None

    result_row = {
        'Model': 'Matheuristic',
        'Instance': instance,
        'OFV': round(obj_value, 0),
        'MipGap(%)': mip_gap,
        'CPU_time(SEC)': round(end_time, 2),
        'Memory_usage(MB)': abs(round(mem_usage, 2)),
        'Number_constraints': model.lp.model.NumConstrs,
        'Status': f"K={k}"  # Here the status is the value of k (# sungroups) instead of model status
    }

    update_results_csv(result_row, file_name, field_names)


def collect_heuristic_without_modeling(instance='GS1', k=20):
    """
        Collect the Matheuristic without the new modeling technique, to be compared with Heuristic results to see
        the impact of the new modeling technique
    """
    instance_path = 'Datasets\\' + instance
    loader = DataLoader(instance_path)
    data = loader.data

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)
    model = Matheuristic(data, k)

    start_time = time.time()
    model.run_without_new_modeling()
    end_time = time.time() - start_time
    mem_after = process.memory_info().rss / (1024 * 1024)
    mem_usage = mem_after - mem_before
    obj_value = model.iterative_algorithm.compute_objective_value()

    mblp_obj = None
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
        mblp_row = df[(df['Model'] == 'MBLP') & (df['Instance'] == instance)]
        if not mblp_row.empty:
            mblp_obj_val = mblp_row.iloc[0]['OFV']
            try:
                mblp_obj = float(str(mblp_obj_val).replace(',', ''))
            except Exception as e:
                print("Error converting MBLP OFV to float:", e)
                mblp_obj = None

    if mblp_obj is not None and obj_value is not None:
        mip_gap = abs(mblp_obj - obj_value) / abs(mblp_obj) * 100
        mip_gap = round(mip_gap, 2)
    else:
        mip_gap = None

    result_row = {
        'Model': 'MatheuristicWithoutModeling',
        'Instance': instance,
        'OFV': round(obj_value, 0),
        'MipGap(%)': mip_gap,
        'CPU_time(SEC)': round(end_time, 2),
        'Memory_usage(MB)': abs(round(mem_usage, 2)),
        'Number_constraints': model.lp.model.NumConstrs,
        'Status': f"K={k}"  # Here the status is the value of k (# sungroups) instead of model status
    }

    update_results_csv(result_row, file_name, field_names)

# -------------------------------------------------------------------------------------------------------------------
# Plotting Functions
# -------------------------------------------------------------------------------------------------------------------

# save bar charts for the three models wrt to CPU_time, Memory_usage and Number of Constraints
def plot_overlapped_bar_chart(instances, values_dict, title, ylabel, filename, colors):
    plt.figure(figsize=(15, 6))
    width = 0.8
    models = list(values_dict.keys())

    for i, instance in enumerate(instances):
        bars = [(model, values_dict[model][i], colors[model])
                for model in models if values_dict[model][i] is not None]
        if not bars:
            # Skip this instance if no model has valid data
            continue
        bars_sorted = sorted(bars, key=lambda x: x[1], reverse=True)
        for order, (model, value, color) in enumerate(bars_sorted):
            label = model if i == 0 else None
            plt.bar(i, value, width, color=color, zorder=order, label=label)

    valid_indices = [i for i, instance in enumerate(instances)
                     if any(values_dict[model][i] is not None for model in models)]
    valid_instances = [instances[i] for i in valid_indices]

    plt.xticks(valid_indices, valid_instances, rotation=45)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_from_csv(file_name, colors, dataset_type='GS'):
    df = pd.read_csv(file_name)
    # Filter rows for given instances type (Small, Medium)
    df_filtered = df[df['Instance'].str.startswith(dataset_type)]
    instances = sorted(df_filtered['Instance'].unique())

    # Helper function to build a list of metric values for a given metric and model
    def build_metric_list(metric, model):
        df_model = df_filtered[df_filtered['Model'] == model]
        lookup = {}

        for idx, row in df_model.iterrows():
            # If the status for this row is "out of memory", do not report its metric
            if str(row['Status']).strip().lower() == 'out of memory':
                lookup[row['Instance']] = None
            else:
                lookup[row['Instance']] = row[metric]

        return [lookup.get(inst, None) for inst in instances]

    models = ['MBLP', 'Matheuristic', 'AlternativeMBLP']
    memory_usage = {m: build_metric_list('Memory_usage(MB)', m) for m in models}
    cpu_time = {m: build_metric_list('CPU_time(SEC)', m) for m in models}
    num_constraints = {m: build_metric_list('Number_constraints', m) for m in models}

    # Plot overlapped bar charts for each metric
    plot_overlapped_bar_chart(instances, memory_usage,
                              title="Memory Usage per Instance",
                              ylabel="Memory Usage (MB)",
                              filename="Plots/GM_memory_usage.png",
                              colors=colors)

    plot_overlapped_bar_chart(instances, cpu_time,
                              title="CPU Time per Instance",
                              ylabel="CPU Time (sec)",
                              filename="Plots/GM_cpu_time.png",
                              colors=colors)

    plot_overlapped_bar_chart(instances, num_constraints,
                              title="Number of Constraints per Instance",
                              ylabel="Number of Constraints",
                              filename="Plots/GM_number_of_constraints.png",
                              colors=colors)


def plot_mipgap_vs_k(instance='GS2', optimal_objective=125742):
    """
    Computes and plots the MipGap (%) versus the number of clusters k for the Matheuristic model on a given instance
    For each k, it runs the Matheuristic model, retrieves the objective function value using
    the iterative algorithm, and calculates the MipGap (%) with respect to the provided optimal objective

    Parameters:
      instance (str): instance name
      optimal_objective (float): The known optimal objective value for the instance
    """
    instance_path = f'Datasets\\{instance}'
    dataloader = DataLoader(instance_path)
    data = dataloader.data

    # Create a dictionary to store the MipGap (%) for different k values
    heuristic_mip_gap = {k: 0 for k in range(10, 110, 10)}

    for k in range(10, 110, 10):
        # Instantiate and run the Matheuristic model with a given k
        heuristic = Matheuristic(data, k)
        heuristic.run()
        # Compute MipGap
        gap = (abs(optimal_objective - heuristic.iterative_algorithm.compute_objective_value()) /
               abs(optimal_objective)) * 100
        heuristic_mip_gap[k] = round(gap, 2)

    x_values = list(heuristic_mip_gap.keys())
    y_values = list(heuristic_mip_gap.values())

    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='blue')
    plt.xlabel('k (number of clusters)')
    plt.ylabel('MipGap (%)')
    plt.title(f'MipGap (%) vs k for instance {instance}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Plots/mipgap_vs_k.png')


def plot_cpu_time_vs_k(instance='GS2'):
    """
    Computes and plots the CPU time (sec) versus the number of clusters k for the Matheuristic model on a given
    instance for each k
    """
    instance_path = f'Datasets\\{instance}'
    dataloader = DataLoader(instance_path)
    data = dataloader.data

    # Create a dictionary to store the CPU time (sec) for different k values
    heuristic_cpu_time = {k: 0 for k in range(10, 110, 10)}

    for k in range(10, 110, 10):
        # Instantiate and run the Matheuristic model with a given k
        heuristic = Matheuristic(data, k)
        start_time = time.time()

        heuristic.run()

        end_time = time.time() - start_time
        heuristic_cpu_time[k] = round(end_time, 2)

    x_values = list(heuristic_cpu_time.keys())
    y_values = list(heuristic_cpu_time.values())

    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='blue')
    plt.xlabel('k (number of clusters)')
    plt.ylabel('CPU time (sec)')
    plt.title(f'CPU time (sec) vs k')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Plots/cpu_time_vs_k.png')


def plot_num_constrs_vs_k(instance='GS2'):
    """
    Computes and plots the number of constraints versus the number of clusters k for the Matheuristic model on a given
    instance for each k
    """
    instance_path = f'Datasets\\{instance}'
    dataloader = DataLoader(instance_path)
    data = dataloader.data

    # Create a dictionary to store the CPU time (sec) for different k values
    heuristic_num_constrs = {k: 0 for k in range(10, 110, 10)}

    for k in range(10, 110, 10):
        # Instantiate and run the Matheuristic model with a given k
        heuristic = Matheuristic(data, k)
        heuristic.run()
        heuristic_num_constrs[k] = heuristic.lp.model.NumConstrs

    x_values = list(heuristic_num_constrs.keys())
    y_values = list(heuristic_num_constrs.values())

    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='blue')
    plt.xlabel('k (number of clusters)')
    plt.ylabel('Number of Constraints')
    plt.title(f'Number of Constraints vs k')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Plots/num_constrs_vs_k.png')


def plot_conflicts_vs_k(instance='GS2'):
    """
    Computes and plots the number of conflict constraints versus the number of clusters (k) for the Matheuristic model on a given
    instance for each k
    """
    instance_path = f'Datasets\\{instance}'
    dataloader = DataLoader(instance_path)
    data = dataloader.data

    heuristic_conflicts = {k: 0 for k in range(10, 110, 10)}

    for k in range(10, 110, 10):
        heuristic = Matheuristic(data, k)
        heuristic.run()
        heuristic_conflicts[k] = heuristic.lp.count_conflict_constraints()

    x_values = list(heuristic_conflicts.keys())
    y_values = list(heuristic_conflicts.values())

    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='blue')
    plt.xlabel('k (number of clusters)')
    plt.ylabel('Conflict Constraints')
    plt.title(f'Conflict Constraints vs k')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Plots/conflicts_vs_k.png')


def performance_profile_from_times(instances, times, algo_names):
    """
    Plots a performance profile comparing algorithms based on CPU times
    :param instances: list of instances names
    :param times: dictionary mapping algorithm names to lists of CPU times
    """
    # Convert the times lists into numpy arrays
    times_arr = {algo: np.array(times[algo]) for algo in algo_names}

    best_time = np.min(np.vstack([times_arr[algo] for algo in algo_names]), axis=0)
    # Compute performance ratios for each algorithm
    ratios = {algo: times_arr[algo] / best_time for algo in algo_names}

    t_min = 1.0
    t_max = np.max([np.max(ratios[algo]) for algo in algo_names])
    t_values = np.linspace(t_min, t_max, 100)
    # Compute the fraction of instances solved within each factor t
    profile = {algo: [] for algo in algo_names}
    for t in t_values:
        for algo in algo_names:
            fraction = np.mean(ratios[algo] <= t)
            profile[algo].append(fraction)
    print(f"Instances: {instances}")

    for algo in algo_names:
        print(f"\nRatios for {algo}:")
        print(ratios[algo])

    plt.figure(figsize=(8, 6))
    for algo in algo_names:
        plt.plot(t_values, profile[algo], marker='o', label=algo)
    plt.xlabel('Performance factor t')
    plt.ylabel('Fraction of instances solved within factor t')
    plt.title('Performance Profile (CPU Time)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Plots/performance_profile_cpu_time.png')
    plt.show()


def build_metric_list(metric, model, df_filtered):
    """
    Build a list of metric values for each instance for a given model
    """
    df_model = df_filtered[df_filtered['Model'] == model]

    # Build a lookup dictionary mapping each instance to its metric value
    lookup = {}
    for idx, row in df_model.iterrows():
        if metric == 'CPU_time(SEC)' and str(row['Status']).strip().lower() == 'out of memory':
            lookup[row['Instance']] = np.inf
        else:
            lookup[row['Instance']] = row[metric]
    # Determine the default value: for CPU time missing values, default to infinity
    # for other metrics default to None
    default = np.inf if metric == 'CPU_time(SEC)' else None

    return [lookup.get(inst, default) for inst in instances]


def build_cpu_time_dict_from_csv(file_name, algo_names):
    """
    Reads the CSV file, filters rows for the given algorithms, and returns:
      cpu_time: a dictionary mapping each algorithm name to a list of CPU times corresponding to each instance
    """
    df = pd.read_csv(file_name)
    df_filtered = df[df['Model'].isin(algo_names)]
    instances = sorted(df_filtered['Instance'].unique())

    def local_build_metric_list(metric, model):
        """
        Build a list of values for a given metric and model
        Special handling for cpu_time if the Status is "out of memory" and if a value exists it is multiplied by 3
        if no value exists returns np.inf
        """
        lookup = {}
        df_model = df_filtered[df_filtered['Model'] == model]

        for idx, row in df_model.iterrows():
            if metric == 'CPU_time(SEC)' and str(row['Status']).strip().lower() == 'out of memory':
                if pd.isna(row[metric]):
                    lookup[row['Instance']] = np.inf
                else:
                    lookup[row['Instance']] = row[metric] * 3
            else:
                lookup[row['Instance']] = row[metric]

        default = np.inf if metric == 'CPU_time(SEC)' else None
        return [lookup.get(inst, default) for inst in instances]

    cpu_time = {m: local_build_metric_list('CPU_time(SEC)', m) for m in algo_names}

    return instances, cpu_time



if __name__=='__main__':
    file_name = 'results.csv'
    colors = {
        'MBLP': '#aec7e8',  # Light blue
        'Matheuristic': '#98df8a',  # Light green
        'AlternativeMBLP': '#ffbb78'  # Light orange
    }
    algo_names = ['MBLP', 'AlternativeMBLP']
    # run models and save results
    collect_mblp_results(model_class=MBLP, instance='GS1')
    collect_heuristic_results(instance='GS1', k=20)
    collect_heuristic_without_modeling(instance='GS1', k=20)
    # Plots for comparative analysis
    plot_from_csv(file_name, colors, dataset_type='GS')
    plot_mipgap_vs_k(instance='GS2', optimal_objective=125742)
    instances, cpu_time = build_cpu_time_dict_from_csv(file_name, algo_names)
    performance_profile_from_times(instances, cpu_time, algo_names)
    plot_cpu_time_vs_k()
    plot_num_constrs_vs_k()
    plot_conflicts_vs_k()
