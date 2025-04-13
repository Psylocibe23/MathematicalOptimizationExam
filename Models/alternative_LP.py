from Models.LP import LPMix
from gurobipy import Model, GRB, quicksum


class AlternativeLP(LPMix):
    # Second LP formulation used to test the impact of group-level constraints (LPMix) vs pairs of conflicting
    # activities constraints (AlternativeLP)
    def __init__(self, data, groups, subgroups):
        super().__init__(data, groups, subgroups)


    def add_variables(self):
        super().add_variables()


    def add_constraints(self):
        super().add_constraints()


    def add_conflict(self):
        self.model.addConstrs(
            (
                quicksum(self.x[(p, k, j)] for j in [j1, j2]) <= len(subgroup)
                for p in self.subgroups.keys()
                for k, subgroup in enumerate(self.subgroups[p])
                for (j1, j2) in self.data['T']
                if j1 in self.eligibility_map[p] and j2 in self.eligibility_map[p]
            )
        )

    def count_conflict_constraints(self):
        total_conflicts = 0
        for p, subgroup_list in self.subgroups.items():
            num_subgroups = len(subgroup_list)
            num_conflict_sets = self.data['n_c'][p]
            total_conflicts += num_subgroups * num_conflict_sets
        print("LP (Matheuristic) conflict constraints count:", total_conflicts)


    def run(self):
        self.add_variables()
        self.add_constraints()
        self.add_conflict()
        self.count_conflict_constraints()
        self.set_objective()
        self.optimize()
        nonzero_count = sum(1 for var in self.model.getVars() if abs(var.X) > 1e-6)
        print("Total number of nonzero assignments in the model:", nonzero_count)
        print("Total number of constraints in the model:", self.model.NumConstrs)