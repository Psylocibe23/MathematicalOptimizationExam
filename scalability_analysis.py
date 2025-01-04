import time
import psutil
import pandas as pd
from loader import DataLoader
from MBLP import MBLP

def get_dataset_data(dataset_name):
    """
    Return data for one these datasets:
    -GS1 (10.000 customers, 50 activities)
    -GS5 (20.000 customers, 75 activities)
    -GM1 (100.000 customers, 100 activities)
    -GM5 (200.000 customers, 125 activities)
    """
    if dataset_name == 'GS1':
        base_path = r"C:\Users\sprea\Desktop\MathematicalOptimization\Datasets\GS1"
    elif dataset_name == 'GS5':
        base_path = r"C:\Users\sprea\Desktop\MathematicalOptimization\Datasets\GS5"
    elif dataset_name == 'GM1':
        base_path = r"C:\Users\sprea\Desktop\MathematicalOptimization\Datasets\GM1"
    elif dataset_name == 'GM5':
        base_path = r"C:\Users\sprea\Desktop\MathematicalOptimization\Datasets\GM5"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Initialize dataloader
    dataloader = DataLoader(base_path)
    return dataloader.data


def run_scalability_experiment():
    dataset_names = ['GS1', 'GS5', 'GM1', 'GM5']
    results = []

    for dset in dataset_names:
        print(f"\n--- Running MBLP on dataset: {dset} ---")

        # 1) Load data via DataLoader
        data = get_dataset_data(dset)

        # 2) Create MBLP model
        model_instance = MBLP(data)

        # 3) measure memory usage BEFORE
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024 * 1024)

        # 4) Record start time
        start_time = time.time()

        # 5) Run the model
        model_instance.run()

        # 6) Record end time
        end_time = time.time()
        runtime = end_time - start_time

        # 7) Memory usage AFTER
        mem_after = process.memory_info().rss / (1024 * 1024)
        mem_used = mem_after - mem_before

        # 8) Gather solution info
        if hasattr(model_instance, 'model') and model_instance.model is not None:
            total_constraints = model_instance.model.NumConstrs
            total_variables   = model_instance.model.NumVars

            if model_instance.model.Status == 2:  # GRB.OPTIMAL
                obj_value = model_instance.model.ObjVal
            else:
                obj_value = None
        else:
            total_constraints = 0
            total_variables   = 0
            obj_value         = None

        # 9) Store in results
        results.append({
            'dataset': dset,
            'runtime_sec': runtime,
            'mem_used_MB': mem_used,
            'constraints': total_constraints,
            'variables': total_variables,
            'objective': obj_value
        })

    # 10) Convert to DataFrame and print
    df = pd.DataFrame(results)
    print("\nScalability Results:")
    print(df)
    df.to_csv("scalability_results.csv", index=False)
    print("Results saved to scalability_results.csv")

run_scalability_experiment()


"""
RESULTS:

Scalability Results:
  dataset  runtime_sec   mem_used_MB  constraints  variables     objective
0     GS1     2.943065    638.851562        10075     500004  2.301497e+05
1     GS5     6.475440   1492.851562        20109    1500003  2.196454e+05
2     GM1    70.052554  13271.167969       100756   10000004  5.875897e+06
3     GM5   208.182224   8499.402344       201565   25000004  8.101797e+06



Risultati riportati sull'articolo per i dataset usati per i test:
-GS1: objective function value 150.000 (1.5 su scala 100k)
      number of constraints 34.000
-GS5: objective function value 130.000 (1.3 su scala 100k)
      number of constraints 36.000
-GM1: objective function value 3.600.000 (36 su scala 100k)
      number of constraints 1.154.000 
-GM2: objective function value 47.600.000 (47.6 su scala 100k)
      number of constraints 3.162.000
      
      
--- Running MBLP on dataset: GS1 ---------------------------------------------------------------------------------------
Set parameter Username
Academic license - for non-commercial use only - expires 2025-04-25
Mapping for b_a: {'direct mail': 0, 'email': 1}
Mapping for b_b: {'call center': 0, ('direct mail', 'email', 'text message'): 1}
Mapping for b_a_bar: {'call center': 0, 'text message': 1}
Mapping for b_s: {'ALL': 0}
Mapping for b_s_bar: {'ALL': 0}
Mapping for b_m_bar: {'ALL': 0}
Gurobi Optimizer version 11.0.2 build v11.0.2rc0 (win64 - Windows 11+.0 (26100.2))

CPU model: 12th Gen Intel(R) Core(TM) i7-12700KF, instruction set [SSE2|AVX|AVX2]
Thread count: 12 physical cores, 20 logical processors, using up to 20 threads

Optimize a model with 10075 rows, 500004 columns and 69152 nonzeros
Model fingerprint: 0x65d5d1d9
Variable types: 4 continuous, 500000 integer (500000 binary)
Coefficient statistics:
  Matrix range     [8e-03, 9e+00]
  Objective range  [2e-03, 6e+02]
  Bounds range     [1e+00, 2e+03]
  RHS range        [1e+00, 9e+03]
Found heuristic solution: objective 61682.238483
Presolve removed 7689 rows and 479903 columns
Presolve time: 0.04s
Presolved: 2386 rows, 20101 columns, 40860 nonzeros
Variable types: 1 continuous, 20100 integer (20099 binary)
Found heuristic solution: objective 94828.864689

Root relaxation: objective 2.301558e+05, 1263 iterations, 0.04 seconds (0.07 work units)

    Nodes    |    Current Node    |     Objective Bounds      |     Work
 Expl Unexpl |  Obj  Depth IntInf | Incumbent    BestBd   Gap | It/Node Time

     0     0 230155.817    0    2 94828.8647 230155.817   143%     -    0s
H    0     0                    230149.72052 230155.817  0.00%     -    0s

Explored 1 nodes (1263 simplex iterations) in 0.28 seconds (0.26 work units)
Thread count was 20 (of 20 available processors)

Solution count 4: 230150 94828.9 62199.3 61682.2 

Optimal solution found (tolerance 1.00e-04)
Best objective 2.301497205224e+05, best bound 2.301558168085e+05, gap 0.0026%
Slack Variables for Min assignment:
z_a[direct mail] = 0.0
z_a[email] = 0.0

Slack Variables for Min sales:
z_s[ALL] = 0.017888404028347914

Slack Variables for Max sales:
z_s_bar[ALL] = 0.017888404028347914

Constraints:
Total number of constraints: 10075
Total number of variables: 500004

Optimization complete for Mixed Binary Linear Model
Maximum potential profit (in 100k dollars): 2,302
Penalty for minimum assignment violations (in 100k dollars): 0,000
Penalty for minimum sales violations (in 100k dollars): 0,000
Penalty for maximum sales violations (in 100k dollars): 0,000
Alpha, Beta, Gamma values: 584.1478039597198, 584.1478039597198, 584.1478039597198
Objective value: 230,150





--- Running MBLP on dataset: GS5 ---------------------------------------------------------------------------------------
Mapping for b_a: {'text message': 0}
Mapping for b_b: {'call center': 0, ('direct mail', 'email', 'text message'): 1}
Mapping for b_a_bar: {'direct mail': 0, 'email': 1}
Mapping for b_s: {'ALL': 0}
Mapping for b_s_bar: {'ALL': 0}
Mapping for b_m_bar: {'ALL': 0}
Gurobi Optimizer version 11.0.2 build v11.0.2rc0 (win64 - Windows 11+.0 (26100.2))

CPU model: 12th Gen Intel(R) Core(TM) i7-12700KF, instruction set [SSE2|AVX|AVX2]
Thread count: 12 physical cores, 20 logical processors, using up to 20 threads

Optimize a model with 20109 rows, 1500003 columns and 99019 nonzeros
Model fingerprint: 0xd13d9e5c
Variable types: 3 continuous, 1500000 integer (1500000 binary)
Coefficient statistics:
  Matrix range     [7e-03, 7e+00]
  Objective range  [1e-04, 4e+02]
  Bounds range     [1e+00, 2e+03]
  RHS range        [1e+00, 4e+03]
Found heuristic solution: objective 155694.83793
Found heuristic solution: objective 155694.83793
Presolve removed 17882 rows and 1473900 columns
Presolve time: 0.09s
Presolved: 2227 rows, 26103 columns, 39273 nonzeros
Variable types: 1 continuous, 26102 integer (26101 binary)

Root relaxation: objective 2.196636e+05, 1196 iterations, 0.03 seconds (0.06 work units)

    Nodes    |    Current Node    |     Objective Bounds      |     Work
 Expl Unexpl |  Obj  Depth IntInf | Incumbent    BestBd   Gap | It/Node Time

     0     0 219663.593    0    2 155694.838 219663.593  41.1%     -    0s
H    0     0                    219645.40439 219663.593  0.01%     -    0s

Explored 1 nodes (1196 simplex iterations) in 0.42 seconds (0.30 work units)
Thread count was 20 (of 20 available processors)

Solution count 3: 219645 155695 154731 

Optimal solution found (tolerance 1.00e-04)
Best objective 2.196454043929e+05, best bound 2.196635926900e+05, gap 0.0083%
Slack Variables for Min assignment:
z_a[text message] = -0.0

Slack Variables for Min sales:
z_s[ALL] = 0.04042460512322307

Slack Variables for Max sales:
z_s_bar[ALL] = 0.04042460512322307

Constraints:
Total number of constraints: 20109
Total number of variables: 1500003

Optimization complete for Mixed Binary Linear Model
Maximum potential profit (in 100k dollars): 2,197
Penalty for minimum assignment violations (in 100k dollars): 0,000
Penalty for minimum sales violations (in 100k dollars): 0,000
Penalty for maximum sales violations (in 100k dollars): 0,000
Alpha, Beta, Gamma values: 417.5877532661888, 417.5877532661888, 417.5877532661888
Objective value: 219,645





--- Running MBLP on dataset: GM1 ---------------------------------------------------------------------------------------
Mapping for b_a: {'text message': 0, 'email': 1}
Mapping for b_b: {'call center': 0, ('direct mail', 'email', 'text message'): 1}
Mapping for b_a_bar: {'call center': 0, 'direct mail': 1}
Mapping for b_s: {'ALL': 0}
Mapping for b_s_bar: {'ALL': 0}
Mapping for b_m_bar: {'ALL': 0}
Gurobi Optimizer version 11.0.2 build v11.0.2rc0 (win64 - Windows 11+.0 (26100.2))

CPU model: 12th Gen Intel(R) Core(TM) i7-12700KF, instruction set [SSE2|AVX|AVX2]
Thread count: 12 physical cores, 20 logical processors, using up to 20 threads

Optimize a model with 100756 rows, 10000004 columns and 1345526 nonzeros
Model fingerprint: 0xf6794324
Variable types: 4 continuous, 10000000 integer (10000000 binary)
Coefficient statistics:
  Matrix range     [4e-03, 2e+01]
  Objective range  [5e-05, 9e+02]
  Bounds range     [1e+00, 2e+04]
  RHS range        [1e+00, 2e+05]
Found heuristic solution: objective 3861012.8098
Presolve removed 26614 rows and 9499832 columns
Presolve time: 0.80s
Presolved: 74142 rows, 500172 columns, 1056819 nonzeros
Found heuristic solution: objective 4627875.6837
Variable types: 1 continuous, 500171 integer (500169 binary)
Found heuristic solution: objective 4741308.2086
Iteration    Objective       Primal Inf.    Dual Inf.      Time
   29583    5.8759027e+06   0.000000e+00   0.000000e+00     11s
Concurrent spin time: 0.00s

Solved with barrier
   29583    5.8759027e+06   0.000000e+00   0.000000e+00     11s

Root relaxation: objective 5.875903e+06, 29583 iterations, 4.50 seconds (3.13 work units)

    Nodes    |    Current Node    |     Objective Bounds      |     Work
 Expl Unexpl |  Obj  Depth IntInf | Incumbent    BestBd   Gap | It/Node Time

     0     0 5875902.75    0    1 4741308.21 5875902.75  23.9%     -   10s
H    0     0                    5875896.5669 5875902.75  0.00%     -   11s

Explored 1 nodes (29583 simplex iterations) in 11.63 seconds (10.12 work units)
Thread count was 20 (of 20 available processors)

Solution count 5: 5.8759e+06 4.74131e+06 4.69515e+06 ... 3.86101e+06

Optimal solution found (tolerance 1.00e-04)
Best objective 5.875896566880e+06, best bound 5.875902746847e+06, gap 0.0001%
Slack Variables for Min assignment:
z_a[text message] = -0.0
z_a[email] = -0.0

Slack Variables for Min sales:
z_s[ALL] = 0.013085551550043428

Slack Variables for Max sales:
z_s_bar[ALL] = 0.013085551550043428

Constraints:
Total number of constraints: 100756
Total number of variables: 10000004

Optimization complete for Mixed Binary Linear Model
Maximum potential profit (in 100k dollars): 58,759
Penalty for minimum assignment violations (in 100k dollars): 0,000
Penalty for minimum sales violations (in 100k dollars): 0,000
Penalty for maximum sales violations (in 100k dollars): 0,000
Alpha, Beta, Gamma values: 862.2740421318402, 862.2740421318402, 862.2740421318402
Objective value: 5.875,897




--- Running MBLP on dataset: GM5 --------------------------------------------------------------------------------------
Mapping for b_a: {'direct mail': 0, 'email': 1}
Mapping for b_b: {'call center': 0, ('direct mail', 'email', 'text message'): 1}
Mapping for b_a_bar: {'call center': 0, 'text message': 1}
Mapping for b_s: {'ALL': 0}
Mapping for b_s_bar: {'ALL': 0}
Mapping for b_m_bar: {'ALL': 0}
Gurobi Optimizer version 11.0.2 build v11.0.2rc0 (win64 - Windows 11+.0 (26100.2))

CPU model: 12th Gen Intel(R) Core(TM) i7-12700KF, instruction set [SSE2|AVX|AVX2]
Thread count: 12 physical cores, 20 logical processors, using up to 20 threads

Optimize a model with 201565 rows, 25000004 columns and 2946964 nonzeros
Model fingerprint: 0x0b9ca488
Variable types: 4 continuous, 25000000 integer (25000000 binary)
Coefficient statistics:
  Matrix range     [3e-03, 9e+00]
  Objective range  [3e-06, 7e+02]
  Bounds range     [1e+00, 2e+04]
  RHS range        [1e+00, 2e+05]
Found heuristic solution: objective 5251241.1371
Found heuristic solution: objective 5251241.1371
Presolve removed 37532 rows and 23861654 columns
Presolve time: 3.80s
Presolved: 164033 rows, 1138350 columns, 2420663 nonzeros
Found heuristic solution: objective 6555771.7962
Variable types: 1 continuous, 1138349 integer (1138347 binary)
Found heuristic solution: objective 6783604.5384
Root relaxation: objective 8.101844e+06, 55085 iterations, 19.09 seconds (11.96 work units)

    Nodes    |    Current Node    |     Objective Bounds      |     Work
 Expl Unexpl |  Obj  Depth IntInf | Incumbent    BestBd   Gap | It/Node Time

     0     0 8101844.26    0    5 6783604.54 8101844.26  19.4%     -   37s
H    0     0                    8101796.8443 8101844.26  0.00%     -   38s

Explored 1 nodes (55085 simplex iterations) in 39.45 seconds (28.57 work units)
Thread count was 20 (of 20 available processors)

Solution count 5: 8.1018e+06 6.7836e+06 6.61721e+06 ... 5.25124e+06

Optimal solution found (tolerance 1.00e-04)
Best objective 8.101796844335e+06, best bound 8.101844263592e+06, gap 0.0006%
Slack Variables for Min assignment:
z_a[direct mail] = -0.0
z_a[email] = -0.0

Slack Variables for Min sales:
z_s[ALL] = 0.0

Slack Variables for Max sales:
z_s_bar[ALL] = 0.0

Constraints:
Total number of constraints: 201565
Total number of variables: 25000004

Optimization complete for Mixed Binary Linear Model
Maximum potential profit (in 100k dollars): 81,018
Penalty for minimum assignment violations (in 100k dollars): 0,000
Penalty for minimum sales violations (in 100k dollars): 0,000
Penalty for maximum sales violations (in 100k dollars): 0,000
Alpha, Beta, Gamma values: 656.7651257762399, 656.7651257762399, 656.7651257762399
Objective value: 8.101,797
"""

