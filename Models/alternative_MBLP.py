from Models.MBLP_class import base_MBLP
from Models.Preprocessing import Preprocessing
from Models.Matheuristic import Matheuristic
from gurobipy import quicksum


class AlternativeMBLP(base_MBLP):
    # Second formulation of the MBLP that uses the LP solution
    def __init__(self, data):
        super().__init__('Mixed Binary Linear Model - Alternative', data)
        # If needed, you can set a default k value for clustering here.
        self.k = 20


    def data_preparation(self):
        # Build eligibility patterns
        M = Matheuristic.build_eligibility_patterns(self.data['I'], self.data['J'], self.data['J_i'])
        M_sorted, sorted_indices = Matheuristic.sort_matrix(M)
        # Create eligibility groups
        self.groups = Matheuristic.create_eligibility_groups(M_sorted, sorted_indices, self.data['I'])
        # Build the profit matrix
        A = Matheuristic.build_profit_matrix(self.data['I'], self.data['J'], self.data['e_ij'], self.data['J_i'])
        # Cluster groups into subgroups
        self.subgroups = Matheuristic.cluster_eligibility_groups(A, self.groups, self.data['I'], self.k)
        prep = Preprocessing(self.data, self.groups, self.subgroups)
        prep.run()


    def add_constraints(self):
        super().add_constraints()
        # Group-level conflict constraints
        self.model.addConstrs(
            (quicksum(self.x[(i, j)] for j in self.data['J_c'][p][l]) <= 1
             for p in self.subgroups
             for l in range(self.data['n_c'][p])
             for subgroup in self.subgroups[p]
             for i in subgroup)
        )

    def count_conflict_constraints(self):
        total_conflicts = 0
        for p, subgroup_list in self.subgroups.items():
            num_conflict_sets = self.data['n_c'][p]
            for subgroup in subgroup_list:
                total_conflicts += len(subgroup) * num_conflict_sets
        print("AlternativeMBLP conflict constraints count:", total_conflicts)


    def run(self):
        self.model.setParam('NodefileStart', 0.5)
        self.add_variables()
        self.data_preparation()
        self.add_constraints()
        self.count_conflict_constraints()
        self.set_objective()
        self.optimize()