from gurobipy import Model, GRB, Var, LinExpr


class base_MBLP:
    # Base class containing the common variables and constraints for the MBLP and the alternative_MBLP
    def __init__(self, model_name, data):
        self.model = Model(model_name)
        self.data = data
        self.variables = {}
        self.constraints = {}


    def add_variables(self):
        """
        Add decision variables and slack variables to the model

        Decision Variables:
        - x[i,j]: Binary variable indicating if customer i is assigned to activity j

        Slack Variables:
        - z_a[l]: Slack variable for minimum assignment constraints
        - z_s[l]: Slack variable for minimum sales constraints
        - z_s_bar[l]: Slack variable for maximum sales constraints
        -z_m[l]: slack variable for minimum contact constraints
        """
        self.x = self.model.addVars(
            self.data['I'],
            self.data['J'],
            vtype=GRB.BINARY,
            name='x'
        )

        self.z_a = {
            key: self.model.addVar(
                lb=0,
                ub=bound,
                vtype=GRB.CONTINUOUS,
                name=f"z_a{key}"
            ) for key, bound in self.data['b_a'].items()
        }

        self.z_s = {
            key: self.model.addVar(
                lb=0,
                ub=bound,
                vtype=GRB.CONTINUOUS,
                name=f"z_s{key}"
            ) for key, bound in self.data['b_s'].items()
        }

        self.z_s_bar = {
            key: self.model.addVar(
                lb=0,
                ub=self.data['q_tot'],
                vtype=GRB.CONTINUOUS,
                name=f"z_s_bar{key}"
            ) for key, bound in self.data['b_s_bar'].items()
        }

        # Only add z_m if minimum contact constraint exists
        if self.data['has_minimum_contact']:
            self.z_m = {}
            for key, bound in self.data['b_m'].items():
                for i in self.data['I']:
                    self.z_m[(i, key)] = self.model.addVar(
                        lb=0,
                        ub=bound,
                        vtype=GRB.CONTINUOUS,
                        name=f"z_m[{i},{key}]"
                    )

        self.model.update()


    def generate_key_mapping(self, b_something, J_something):
        # Keys mappings (to map the string keys of b_a etc. to the integer keys of J_a etc.)
        key_mapping = {}
        b_keys = list(b_something.keys())
        J_keys = list(J_something.keys())

        if len(b_keys) != len(J_keys):
            raise ValueError(f"Mismatch between {b_something} and {J_something} length!")
        for i, b_key in enumerate(b_keys):
            key_mapping[b_key] = J_keys[i]

        return key_mapping


    def add_constraints(self):
        """
        Add constraints to the model
        - Minimum assignment
        - Maximum assignment
        - Budget
        - Minimum sales
        - Maximum sales
        - Minimum contact
        - Maximum contact
        """
        # generate mappings in order to iterate through dictionaries b_something, J_something which have different keys
        self.mappings = {
            'b_a': self.generate_key_mapping(self.data['b_a'], self.data['J_a']),
            'b_b': self.generate_key_mapping(self.data['b_b'], self.data['J_b']),
            'b_a_bar': self.generate_key_mapping(self.data['b_a_bar'], self.data['J_a_bar']),
            'b_s': self.generate_key_mapping(self.data['b_s'], self.data['J_s']),
            'b_s_bar': self.generate_key_mapping(self.data['b_s_bar'], self.data['J_s_bar']),
            'b_m_bar': self.generate_key_mapping(self.data['b_m_bar'], self.data['J_m_bar']),
        }

        # Minimum assignment (2)
        self.model.addConstrs(
            (
                sum(self.x[i, j] for j in self.data['J_a'][self.mappings['b_a'][key]] for i in self.data['I_j'][j]) + self.z_a[key] >= self.data['b_a'][key]
                for key in self.data['b_a'].keys()
            ),
            name='MinAssignment'
        )

        # Maximum assignment (3)
        self.model.addConstrs(
            (
                sum(self.x[i, j] for j in self.data['J_a_bar'][self.mappings['b_a_bar'][key]] for i in self.data['I_j'][j]) <= self.data['b_a_bar'][key]
                for key in self.data['b_a_bar'].keys()
            ),
            name="MaxAssignment"
        )

        # Maximum Budget (4)
        self.model.addConstrs(
            (
                sum(self.data['c_j'][j] * self.x[i, j] for j in self.data['J_b'][self.mappings['b_b'][key]] for i in self.data['I_j'][j]) <= self.data['b_b'][key]
                for key in self.data['b_b'].keys()
            ),
            name="MAxBudget"
        )

        # Minimum sales (5)
        self.model.addConstrs(
            (
                sum(self.data['q_ij'][i, j] * self.x[i, j] for j in self.data['J_s'][self.mappings['b_s'][key]] for i in self.data['I_j'][j]) + self.z_s[key] >= self.data['b_s'][key]
                for key in self.data['b_s'].keys()
            ),
            name="MinSales"
        )

        # Maximum sales (6)
        self.model.addConstrs(
            (
                sum(self.data['q_ij'][i, j] * self.x[i, j] for j in self.data['J_s_bar'][self.mappings['b_s_bar'][key]] for i in self.data['I_j'][j]) -
                self.z_s_bar[key] <= self.data['b_s_bar'][key]
                for key in self.data['b_s_bar'].keys()
            ),
            name="MaxSales"
        )

        # Minimum contact constraint (7)
        if 'J_m' in self.data and self.data['J_m'] and self.data['has_minimum_contact']:
            if 'b_m' not in self.mappings:
                self.mappings['b_m'] = self.generate_key_mapping(self.data['b_m'], self.data['J_m'])

            self.model.addConstrs(
                (
                    sum(self.x[i, j] for j in self.data['J_m'][self.mappings['b_m'][key]]
                        if i in self.data['I_j'][j])
                    + self.z_m[(i, key)] >= self.data['b_m'][key]
                    for key in self.data['b_m']
                    for i in self.data['I']
                ),
                name="MinContact"
            )

        # Maximum contact constraint (8)
        for key in self.data['b_m_bar']:
            for i in self.data['I']:
                if len(self.data['J_m_bar'][self.mappings['b_m_bar'][key]] & set(self.data['J_i'][i])) > \
                        self.data['b_m_bar'][key]:
                    self.model.addConstr(
                        sum(
                            self.x[i, j]
                            for j in (self.data['J_m_bar'][self.mappings['b_m_bar'][key]] & set(self.data['J_i'][i]))
                        ) <= self.data['b_m_bar'][key],
                        name=f"MaxContact_{i}_{key}"
                    )


    def set_objective(self):
        self.profit = 0
        for (i, j) in self.data['e_ij']:
            revenue_ij = self.data['e_ij'][(i, j)]
            self.profit += (revenue_ij) * self.x[i, j]

        # 2) Penalty for minimum assignment constraints
        self.penalty_a = sum(
            self.data['alpha'] * self.z_a[key]
            for key in self.data['b_a']
        )

        # 3) Penalty for minimum sales constraints
        self.penalty_s = sum(
            self.data['beta'] * self.z_s[key]
            for key in self.data['b_s']
        )

        # 4) Penalty for maximum sales constraints
        self.penalty_s_bar = sum(
            self.data['gamma'] * self.z_s_bar[key]
            for key in self.data['b_s_bar']
        )

        # 5) Penalty for minimum contact constraints (only if present)
        self.penalty_m = 0
        if self.data['has_minimum_contact']:
            self.penalty_m = sum(
                self.data['delta'] * self.z_m[(i, key)]
                for (i, key) in self.z_m
            )

        self.model.setObjective(
            self.profit
            - self.penalty_a
            - self.penalty_s
            - self.penalty_s_bar
            - self.penalty_m,
            GRB.MAXIMIZE
        )


    def optimize(self):
        self.model.setParam('Seed', 0)
        self.model.optimize()

        if self.model.status == GRB.OPTIMAL:
            print("\nSlack Variables for Min assignment:")
            for key in self.data['b_a'].keys():
                print(f"z_a[{key}] = {self.z_a[key].X if isinstance(self.z_a[key], Var) else self.z_a[key]}")

            print("\nSlack Variables for Min sales:")
            for key in self.data['b_s'].keys():
                print(f"z_s[{key}] = {self.z_s[key].X if isinstance(self.z_s[key], Var) else self.z_s[key]}")

            print("\nSlack Variables for Max sales:")
            for key in self.data['b_s_bar'].keys():
                print(f"z_s_bar[{key}] = {self.z_s_bar[key].X if isinstance(self.z_s_bar[key], Var) else self.z_s_bar[key]}")

            if self.data['has_minimum_contact']:
                print("\nSlack Variables for Min contact (nonzero only):")
                tol = 1e-6  # tolerance level for zero
                for key in self.data['b_m'].keys():
                    for i in self.data['I']:
                        var_key = (i, key)
                        if var_key in self.z_m:
                            val = self.z_m[var_key].X if hasattr(self.z_m[var_key], "X") else self.z_m[var_key]
                            if abs(val) > tol:
                                print(f"z_m[({i}, {key})] = {val}")

            print("\nConstraints:")
            print(f"Total number of constraints: {self.model.NumConstrs}")
            print(f"Total number of variables: {self.model.NumVars}")
            profit = sum(
                (self.data['e_ij'][(i, j)]) * self.x[i, j].X
                for (i, j) in self.data['e_ij']
            )
            penalty_a = sum(self.data['alpha'] * self.z_a[key].X for key in self.data['b_a'].keys()) if isinstance(
                self.penalty_a, LinExpr) else self.penalty_a
            penalty_s = sum(self.data['beta'] * self.z_s[key].X for key in self.data['b_s'].keys()) if isinstance(
                self.penalty_s, LinExpr) else self.penalty_s
            penalty_s_bar = sum(
                self.data['gamma'] * self.z_s_bar[key].X for key in self.data['b_s_bar'].keys()) if isinstance(
                self.penalty_s_bar, LinExpr) else self.penalty_s_bar
            penalty_m = sum(
                self.data['delta'] * self.z_m[(i, key)].X
                for (i, key) in self.z_m
            ) if isinstance(self.penalty_m, LinExpr) else self.penalty_m

            # Formatting results
            formatted_profit = f"{profit / 100000:,.5f}".replace(",", "X").replace(".", ",").replace("X", ".")
            formatted_penalty_a = f"{penalty_a / 100000:,.3f}".replace(",", "X").replace(".", ",").replace("X", ".")
            formatted_penalty_s = f"{penalty_s / 100000:,.3f}".replace(",", "X").replace(".", ",").replace("X", ".")
            formatted_penalty_s_bar = f"{penalty_s_bar / 100000:,.3f}".replace(",", "X").replace(".", ",").replace("X", ".")

            print("\nOptimization complete for", self.model.ModelName)
            print(f"Maximum potential profit (in 100k dollars): {formatted_profit}")
            print(f"Penalty for minimum assignment violations (in 100k dollars): {formatted_penalty_a}")
            print(f"Penalty for minimum sales violations (in 100k dollars): {formatted_penalty_s}")
            print(f"Penalty for maximum sales violations (in 100k dollars): {formatted_penalty_s_bar}")
            if self.data['has_minimum_contact']:
                formatted_penalty_m = f"{penalty_m / 100000:,.3f}".replace(",", "X").replace(".", ",").replace("X", ".")
                print(f"Penalty for minimum contact violations (in 100k dollars): {formatted_penalty_m}")
            formatted_obj = f"{self.model.objVal / 1000:,.3f}".replace(",", "X").replace(".", ",").replace("X", ".")
            print(f"\nOBJECTIVE FUNCTION VALUE: {formatted_obj}")
            print(f"MipGap: {self.model.MIPGap}")