from gurobipy import Model, GRB


# Base class containing the common variables and constraints for the MBLP and the alternative_MBLP
class base_MBLP:
    def __init__(self, model_name, data):
        self.model = Model(model_name)
        self.data = data
        self.variables = {}
        self.constraints = {}

    def add_variables(self):
        """
        Add decision variables and slack variables to the model.

        Decision Variables:
        - x[i,j]: Binary variable indicating if customer i is assigned to activity j.

        Slack Variables:
        - z_a[l]: Slack variable for minimum assignment constraints.
        - z_s[l]: Slack variable for minimum sales constraints.
        - z_s_bar[l]: Slack variable for maximum sales constraints.
        -z_m[l]: slack variable for minimum contact constraints (NOT INCLUDED)
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
                ub=bound,
                vtype=GRB.CONTINUOUS,
                name=f"z_s_bar{key}"
            ) for key, bound in self.data['b_s_bar'].items()
        }

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
        - Minimum contact (COMMENTED OUT)
        - Maximum contact
        """

        # generate mappings in order to iterate through dictionaries b_a etc. and J_a etc. which have different keys name
        mappings = {
            'b_a': self.generate_key_mapping(self.data['b_a'], self.data['J_a']),
            'b_b': self.generate_key_mapping(self.data['b_b'], self.data['J_b']),
            'b_a_bar': self.generate_key_mapping(self.data['b_a_bar'], self.data['J_a_bar']),
            'b_s': self.generate_key_mapping(self.data['b_s'], self.data['J_s']),
            'b_s_bar': self.generate_key_mapping(self.data['b_s_bar'], self.data['J_s_bar']),
            'b_m_bar': self.generate_key_mapping(self.data['b_m_bar'], self.data['J_m_bar']),
        }

        # Debugging: Print all generated mappings
        for name, mapping in mappings.items():
            print(f"Mapping for {name}: {mapping}")


        # Minimum assignment (2)
        self.model.addConstrs(
            (
                sum(self.x[i, j] for j in self.data['J_a'][mappings['b_a'][key]] for i in self.data['I_j'][j]) + self.z_a[key] >= self.data['b_a'][key]
                for key in self.data['b_a'].keys()
            ),
            name = 'MinAssignment'
        )

        # Maximum assignment (3)
        self.model.addConstrs(
            (
                sum(self.x[i, j] for j in self.data['J_a_bar'][mappings['b_a_bar'][key]] for i in self.data['I_j'][j]) <= self.data['b_a_bar'][key]
                for key in self.data['b_a_bar'].keys()
            ),
            name = "MaxAssignment"
        )

        # Maximum Budget (4)
        self.model.addConstrs(
            (
                sum(self.data['c_j'][j] * self.x[i, j] for j in self.data['J_b'][mappings['b_b'][key]] for i in self.data['I_j'][j]) <= self.data['b_b'][key]
                for key in self.data['b_b'].keys()
            ),
            name = "MAxBudget"
        )

        # Minimum sales (5)
        self.model.addConstrs(
            (
                sum(self.data['q_ij'][i, j] * self.x[i, j] for j in self.data['J_s'][mappings['b_s'][key]] for i in self.data['I_j'][j]) + self.z_s[key] >= self.data['b_s'][key]
                for key in self.data['b_s'].keys()
            ),
            name = "MinSales"
        )

        # Maximum sales (6)
        self.model.addConstrs(
            (
                sum(self.data['q_ij'][i, j] * self.x[i, j] for j in self.data['J_s_bar'][mappings['b_s_bar'][key]] for i in self.data['I_j'][j]) -
                self.z_s[key] <= self.data['b_s_bar'][key]
                for key in self.data['b_s_bar'].keys()
            ),
            name="MaxSales"
        )

        # Minimum contact constraints (7) #(MIN CONTACT RULE NON PRESENTE NEI DATI FORNITI DAGLI AUTORI)
        """self.model.addConstrs(
            (
                sum(self.x[i, j] for j in (set(self.data['J_m'][mappings['b_m'][key]]) & set(self.data['J_i'][i]))
                    + self.z_m[key] >= self.data['b_m'][key]
                    for key in self.data['b_m'].keys())
            ),
            name="MinContact"
        )"""

        # Maximum contact constraints (8)
        self.model.addConstrs(
            (
                sum(self.x[i, j] for j in (set(self.data['J_m_bar'][mappings['b_m_bar'][key]]) & set(self.data['J_i'][i])))
                <= self.data['b_m_bar'][key]
                for i in self.data['I']
                for key in self.data['b_m_bar'].keys()
            ),
            name="MaxContact"
        )

    def set_objective(self):
        """
        set the penalties and the objective function for the model
        """
        self.profit = sum(
            self.data['e_ij'][i, j] * self.x[i, j]
            for (i,j) in self.data['e_ij']
        )

        # Penalty for Min assignment constraints violation
        self.penalty_a = sum(
            self.data['alpha'] * self.z_a[key]
            for key in self.data['b_a'].keys()
        )

        # Penalty for Min sales constraints violation
        self.penalty_s = sum(
            self.data['beta'] * self.z_s[key]
            for key in self.data['b_s'].keys()
        )

        # Penalty for Max sales constraints violation
        self.penalty_s_bar = sum(
            self.data['gamma'] * self.z_s_bar[key]
            for key in self.data['b_s_bar'].keys()
        )

        # Penalty for Min contact constraints violation
        """
        self.penalty_m = sum(
            self.data['delta'] * self.z_m[key]
            for key in self.data['b_m'].keys()
        )"""

        self.model.setObjective(self.profit - self.penalty_a - self.penalty_s - self.penalty_s_bar, GRB.MAXIMIZE)

    def optimize(self):
        self.model.optimize()

        if self.model.status == GRB.OPTIMAL:
            print("Slack Variables for Min assignment:")
            for key in self.data['b_a'].keys():
                print(f"z_a[{key}] = {self.z_a[key].x}")

            print("\nSlack Variables for Min sales:")
            for key in self.data['b_s'].keys():
                print(f"z_s[{key}] = {self.z_s[key].x}")

            print("\nSlack Variables for Max sales:")
            for key in self.data['b_s_bar'].keys():
                print(f"z_s_bar[{key}] = {self.z_s[key].x}")

            print("\nConstraints:")
            print(f"Total number of constraints: {self.model.NumConstrs}")
            print(f"Total number of variables: {self.model.NumVars}")


            """print("\nDecision Variables:")
            for (i, j), var in self.x.items():
                print(f"x[{i},{j}] = {var.x}")"""

            # Formatting results
            formatted_profit = f"{self.profit.getValue() / 100000:,.3f}".replace(",", "X").replace(".", ",").replace(
                "X", ".")
            formatted_penalty_a = f"{self.penalty_a.getValue() / 100000:,.3f}".replace(",", "X").replace(".",
                                                                                                         ",").replace(
                "X", ".")
            formatted_penalty_s = f"{self.penalty_s.getValue() / 100000:,.3f}".replace(",", "X").replace(".",
                                                                                                         ",").replace(
                "X", ".")
            formatted_penalty_s_bar = f"{self.penalty_s_bar.getValue() / 100000:,.3f}".replace(",", "X").replace(".",
                                                                                                                 ",").replace(
                "X", ".")

            # Printing optimization results
            print("\nOptimization complete for", self.model.ModelName)
            print(f"Maximum potential profit (in 100k dollars): {formatted_profit}")
            print(f"Penalty for minimum assignment violations (in 100k dollars): {formatted_penalty_a}")
            print(f"Penalty for minimum sales violations (in 100k dollars): {formatted_penalty_s}")
            print(f"Penalty for maximum sales violations (in 100k dollars): {formatted_penalty_s_bar}")
            print(
                f"Alpha, Beta, Gamma values: {self.data['alpha']}, {self.data['beta']}, {self.data['gamma']}")

            formatted_obj = f"{self.model.objVal / 1000:,.3f}".replace(",", "X").replace(".", ",").replace("X", ".")
            print(f"Objective value: {formatted_obj}")




