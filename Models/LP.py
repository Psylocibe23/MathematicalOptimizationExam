from gurobipy import Model, GRB, Var, LinExpr, quicksum
from Models.MBLP_class import base_MBLP


class LPMix:

    def __init__(self, data, groups, subgroups):
        self.model = Model('LP')
        self.data = data
        self.groups = groups         # Dictionary: {eligibility_pattern (tuple): [list of customer IDs]}
        self.subgroups = subgroups   # Dictionary: {eligibility_pattern (tuple): [list of subgroups, each is a list of customer IDs]}
        self.x = {}                  # Decision variables
        self.variables = {}

        # Create an eligibility map: pattern -> eligible activities
        # The eligibility pattern is a tuple of binary values (e.g., (0, 0, 1, 1, 0, 0, 1, 0))
        self.eligibility_map = {
            pattern: [j for j, flag in zip(self.data['J'], pattern) if flag == 1]
            for pattern in groups.keys()
        }


    def add_variables(self):
        """
        Continuous variables x_gj indicate how many customers from subgroup g are assigned to activity j
        Subgroup g is identified by (pattern, subgroup_index)
        """
        for pattern, subgroup_list in self.subgroups.items():
            eligible_activities = self.eligibility_map.get(pattern, [])
            for k, subgroup in enumerate(subgroup_list):
                o_pk = len(subgroup)
                for j in eligible_activities:
                    self.x[(pattern, k, j)] = self.model.addVar(
                        lb=0,
                        ub=o_pk,
                        vtype=GRB.CONTINUOUS
                    )

        # Minimum assignment slack variables
        self.z_a = {
            key: self.model.addVar(
                lb=0,
                ub=self.data['b_a'][key],
                vtype=GRB.CONTINUOUS
            ) for key in self.data['b_a'].keys()
        }

        # Minimum sales slack variables
        self.z_s = {
            key: self.model.addVar(
                lb=0,
                ub=self.data['b_s'][key],
                vtype=GRB.CONTINUOUS
            ) for key in self.data['b_s'].keys()
        }

        # Maximum sales slack variables
        self.z_s_bar = {
            key: self.model.addVar(
                lb=0,
                ub=self.data['q_tot'],
                vtype=GRB.CONTINUOUS
            ) for key in self.data['b_s_bar'].keys()
        }

        # Minimum contact slack variables
        if self.data.get('has_minimum_contact', False):
            self.z_m_global = {}
            for l in self.data['b_m'].keys():
                self.z_m_global[l] = self.model.addVar(
                    lb=0,
                    ub=self.data['b_m'][l] * len(self.data['I']),
                    vtype=GRB.CONTINUOUS
                )

        self.model.update()


    def generate_key_mapping(self, b_something, J_something):
        key_mapping = {}
        b_keys = list(b_something.keys())
        J_keys = list(J_something.keys())

        if len(b_keys) != len(J_keys):
            raise ValueError(f"Mismatch in lengths: {b_something.keys()} vs. {J_something.keys()}")

        for i, b_key in enumerate(b_keys):
            key_mapping[b_key] = J_keys[i]

        return key_mapping


    def add_constraints(self):
        dummy = base_MBLP('dummy', self.data)

        self.mappings = {
            'b_a': dummy.generate_key_mapping(self.data['b_a'], self.data['J_a']),
            'b_b': dummy.generate_key_mapping(self.data['b_b'], self.data['J_b']),
            'b_a_bar': dummy.generate_key_mapping(self.data['b_a_bar'], self.data['J_a_bar']),
            'b_s': dummy.generate_key_mapping(self.data['b_s'], self.data['J_s']),
            'b_s_bar': dummy.generate_key_mapping(self.data['b_s_bar'], self.data['J_s_bar']),
            'b_m_bar': dummy.generate_key_mapping(self.data['b_m_bar'], self.data['J_m_bar']),
        }
        # 1. Minimum assignment constraints:
        self.model.addConstrs(
            (
                quicksum(
                    self.x[(p, k, j)]
                    for p in self.subgroups.keys()
                    for k, subgroup in enumerate(self.subgroups[p])
                    for j in (set(self.eligibility_map[p]) & self.data['J_a'][self.mappings['b_a'][l]])
                ) + self.z_a[l] >= self.data['b_a'][l]
                for l in self.data['b_a'].keys()
            )
        )

        # 2. Maximum assignment constraints:
        self.model.addConstrs(
            (
                quicksum(
                    self.x[(p, k, j)]
                    for p in self.subgroups.keys()
                    for k, subgroup in enumerate(self.subgroups[p])
                    for j in (set(self.eligibility_map[p]) & self.data['J_a_bar'][self.mappings['b_a_bar'][l]])
                ) <= self.data['b_a_bar'][l]
                for l in self.data['b_a_bar'].keys()
            )
        )

        # 3. Maximum budget constraints:
        self.model.addConstrs(
            (
                quicksum(
                    self.data['c_j'][j] * self.x[(p, k, j)]
                    for p in self.subgroups.keys()
                    for k, subgroup in enumerate(self.subgroups[p])
                    for j in (set(self.eligibility_map[p]) & self.data['J_b'][self.mappings['b_b'][l]])
                ) <= self.data['b_b'][l]
                for l in self.data['b_b'].keys()
            )
        )

        # 4. Minimum sales constraints:
        self.model.addConstrs(
            (
                quicksum(
                    (
                        sum(self.data['q_ij'].get((i, j), 0) for i in subgroup) / len(subgroup)
                        if len(subgroup) > 0 else 0
                    ) * self.x[(p, k, j)]
                    for p in self.subgroups.keys()
                    for k, subgroup in enumerate(self.subgroups[p])
                    for j in (set(self.eligibility_map[p]) & self.data['J_s'][self.mappings['b_s'][l]])
                ) + self.z_s[l] >= self.data['b_s'][l]
                for l in self.data['b_s'].keys()
            )
        )

        # 5. Maximum sales constraints:
        self.model.addConstrs(
            (
                quicksum(
                    (
                        sum(self.data['q_ij'].get((i, j), 0) for i in subgroup) / len(subgroup)
                        if len(subgroup) > 0 else 0
                    ) * self.x[(p, k, j)]
                    for p in self.subgroups.keys()
                    for k, subgroup in enumerate(self.subgroups[p])
                    for j in (set(self.eligibility_map[p]) & self.data['J_s_bar'][self.mappings['b_s_bar'][l]])
                ) - self.z_s_bar[l] <= self.data['b_s_bar'][l]
                for l in self.data['b_s_bar'].keys()
            )
        )

        # 6. Minimum contact constraints:
        if 'J_m' in self.data and self.data['J_m'] and self.data['has_minimum_contact']:
            if 'b_m' not in self.mappings:
                self.mappings['b_m'] = self.generate_key_mapping(self.data['b_m'], self.data['J_m'])
            self.model.addConstrs(
                (
                    quicksum(
                        self.x[(p, k, j)]
                        for j in (set(self.eligibility_map[p]) &
                                  self.data['J_m'][self.mappings['b_m'][list(self.data['b_m'].keys())[0]]])
                    ) >= len(subgroup) * self.data['b_m'][list(self.data['b_m'].keys())[0]]
                    for p in self.subgroups.keys()
                    for k, subgroup in enumerate(self.subgroups[p])
                )
            )

        # 7. Maximum contact constraints:
        self.model.addConstrs(
            (
                quicksum(
                    self.x[(p, k, j)]
                    for j in (set(self.eligibility_map[p]) & self.data['J_m_bar'][self.mappings['b_m_bar'][key]])
                ) <= self.data['b_m_bar'][key] * len(subgroup)
                for key in self.data['b_m_bar'].keys()
                for p in self.subgroups.keys()
                for k, subgroup in enumerate(self.subgroups[p])
            )
        )

        self.model.update()


    def add_conflict(self):
        # Conflict constraints
        self.model.addConstrs(
            (
                sum(self.x[(p, k, j)] for j in self.data['J_c'][p][lp]) <= len(subgroup)
                for p in self.subgroups.keys()
                for k, subgroup in enumerate(self.subgroups[p])
                for lp in range(self.data['n_c'][p])
            )
        )

        self.model.update()


    def set_objective(self):
        self.profit = sum(
            (
                sum(self.data['e_ij'].get((i, j), 0) for i in subgroup) / len(subgroup) if len(subgroup) else 0
            ) * self.x[(p, k, j)]
            for p, subgroup_list in self.subgroups.items()
            for k, subgroup in enumerate(subgroup_list)
            for j in set(self.eligibility_map[p])
        )

        penalty_a = sum(self.data['alpha'] * self.z_a[l] for l in self.data['b_a'])
        penalty_s = sum(self.data['beta'] * self.z_s[l] for l in self.data['b_s'])
        penalty_s_bar = sum(self.data['gamma'] * self.z_s_bar[l] for l in self.data['b_s_bar'])
        if self.data.get('has_minimum_contact', False):
            penalty_m = sum(self.data['delta'] * self.z_m_global[l] for l in self.data['b_m'].keys())
        else:
            penalty_m = 0

        self.model.setObjective(
            self.profit - (penalty_a + penalty_s + penalty_s_bar + penalty_m),
            GRB.MAXIMIZE
        )

        self.model.update()


    def count_conflict_constraints(self):
        total_conflicts = 0
        for p, subgroup_list in self.subgroups.items():
            num_subgroups = len(subgroup_list)
            num_conflict_sets = self.data['n_c'][p]
            total_conflicts += num_subgroups * num_conflict_sets
        print("LP (Matheuristic) conflict constraints count:", total_conflicts)
        return total_conflicts


    def optimize(self):
        self.model.optimize()

        if self.model.status == GRB.OPTIMAL:
            self.lp_solution = {key: var.X for key, var in self.x.items()}
            print("\nOptimal LP objective value:", self.model.objVal)

            print("\nSlack variables for Minimum Assignment:")
            for key in self.data['b_a'].keys():
                print(f"z_a[{key}] = {self.z_a[key].X}")
            print("\nSlack variables for Minimum Sales:")
            for key in self.data['b_s'].keys():
                print(f"z_s[{key}] = {self.z_s[key].X}")
            print("\nSlack variables for Maximum Sales:")
            for key in self.data['b_s_bar'].keys():
                print(f"z_s_bar[{key}] = {self.z_s_bar[key].X}")
            lp_assignment_count = sum(var_value for var_value in self.lp_solution.values() if var_value > 0)
            print(f"\nLP assignments and constraints:")
            print(f"LP total number of assignments: {lp_assignment_count}")


    def run(self):
        self.add_variables()
        self.add_constraints()
        self.add_conflict()
        self.count_conflict_constraints()
        self.set_objective()
        self.optimize()
        print(f"Total number of constraints in the model: {self.model.NumConstrs}\n")