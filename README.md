![Python](https://img.shields.io/badge/Python-3.12.3-blue.svg)
![Gurobi](https://img.shields.io/badge/Gurobi-11.0.2-blue)
![License](https://img.shields.io/badge/License-MIT-green.svg)

# Python Implementation of the paper "A Matheuristic for a Customer Assignment Problem in Direct Marketing"

This repository contains a Python implementation of the article **"A Matheuristic for a Customer Assignment Problem in Direct Marketing"** by T. Bigler, M. Kammermann, and P. Baumann, published in **European Journal of Operational Research**, Volume 304, 2023, Pages 689–708  
**DOI:** [10.1016/j.ejor.2022.04.009](https://doi.org/10.1016/j.ejor.2022.04.009)

## GitHub Repository with Generated Instances

For the generated instances, please refer to:  
[https://github.com/phil85/customer-assignment-instances](https://github.com/phil85/customer-assignment-instances)

### Article Summary

The article **"A Matheuristic for a Customer Assignment Problem in Direct Marketing"** addresses a real-world challenge faced by a telecommunications company running multiple direct marketing campaigns. The goal is to assign millions of customers to hundreds of marketing activities (like calls, emails, or offers), while respecting a complex set of business and customer-specific constraints.

The complexity arises from:
- Budget and sales targets
- Avoiding excessive customer contact
- Managing conflicts between activities (e.g., overlapping schedules)

Given the enormous scale of the problem, exact optimization methods (like full mixed-binary linear programming) become computationally impractical. To overcome this, the authors propose:
1. **MBLP (Mixed Binary Linear Program):**  
   A complete formulation capturing all constraints but scaling poorly.
2. **Alternative MBLP:**  
   An improved version of the MBLP that applies preprocessing techniques to significantly reduce the number of conflict constraints, enhancing model scalability.
3. **Matheuristic:**  
   A hybrid approach combining optimization and heuristic methods:
   - Customers are first grouped by eligibility patterns.
   - Groups are further split into clusters using Mini-batch k-means.
   - A group-level LP is solved, followed by an iterative algorithm assigning individual customers.
   - The approach balances solution quality and computational time, controlled by parameter **k**.

Key innovations include a **preprocessing technique** that efficiently reduces redundant conflict constraints and a **new modeling technique** to better handle group-level conflicts.

The matheuristic consistently produces high-quality solutions with minimal optimality gap and runs significantly faster than exact methods. In practice, it improves the company’s campaign profitability and scalability, proving effective even on real-world datasets with millions of customers.

Note: Datasets vary by eligibility fraction and the number of eligibility patterns, affecting complexity and scalability

## Repository Structure

- **main.py**  
  Prompts the user for a dataset path (e.g., `Datasets/GS1`) and to choose one of the three models: **MBLP**, **alternative MBLP**, or **Matheuristic**

- **loader.py**  
  Loads and processes datasets to derive constraints, related sets, bounds, and other necessary data

- **models/**  
  - **mblp_class.py** and **mblp.py**  
    Implement the Mixed-Binary Linear Program (MBLP) formulation  
  - **preprocessing.py**, **LP.py**, **iterative_algorithm.py**, and **matheuristic.py**  
    Implement the matheuristic. (The methods `run_without_step_5()` or `run_without_new_modeling()` run the matheuristic with modifications in handling redundant cliques or group-level conflict constraints, respectively)  
  - **alternative_mblp.py**  
    Contains the alternative MBLP formulation that uses preprocessing

- **test.py**  
  A script to run the models on a toy dataset or on small instances for testing

- **scalability_analysis.py**  
  A script to perform scalability analysis and generate plots. Due to computational and memory limitations, it is designed to run on one instance at a time.  
**There is no need to run this script again**, as all the results have already been generated and are available in the `Plots` folder and the `results.txt` file.

- **Plots/**  
  Contains the resulting plots from the scalability analysis on the small and medium instances

- **results.csv**  
  Collects the Objective Function Value, MipGap (%), CPU time (sec), Memory Usage (MB), and Number of Constraints for the datasets that have been analyzed: GS1 to GS8 and GM1 to GM8
  Download file: https://media.githubusercontent.com/media/Psylocibe23/MathematicalOptimizationExam/refs/heads/main/results.csv

## Datasets Description

- **Generated Instances:**  
  - **GS (Small):** Up to 20,000 customers and 75 activities  
  - **GM (Medium):** Up to 200,000 customers and 125 activities  
  - **GL (Large):** Up to 1,000,000 customers and 175 activities

- **Real-World Instances:**  
  - **RL (Large):** Up to 1.4 million customers and 385 activities  
  - **RVL (Very Large):** Over 2 million customers and up to 295 activities 

## How to Use

1. Run `main.py`
2. Enter the path to the dataset (e.g., `Datasets/GS1`)
3. Select the desired model: **MBLP**, **alternative MBLP**, or **Matheuristic**
4. (Optional) Use `test.py` for testing on toy datasets or small instances
5. (Optional) Run `scalability_analysis.py` (on one instance at a time) for scalability experiments


## Experimental Setup

- **Time limit:** 30 minutes per instance  
- **Implementation:** Python 3.12.3, Gurobi 11.0.2  
- **Processor:** Intel Core i7-12700KF (12th Gen) @ 3.61 GHz  
- **Memory:** 32 GB RAM


## License

This project is open source and available under the [MIT License](LICENSE)
