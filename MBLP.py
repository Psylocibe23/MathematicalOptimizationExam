from MBLP_class import base_MBLP

# The first MBLP presented in the article
class MBLP(base_MBLP):
    def __init__(self, data):
        super().__init__('Mixed Binary Linear Model', data)

    def add_constraints(self):
        """
        Add the conflict constraint to the base MBLP
        """
        super().add_constraints()
        # Find customers eligible for pairs of conflicting activities
        for (j1, j2) in self.data['T']:
            # Get customers eligible for both activities j1, j2
            eligible_customers = set(self.data['J_i'].get(j1, [])) & set(self.data['J_i'].get(j2, []))
            for i_float in eligible_customers:
                i = int(i_float)
                # Check if (i, j1) and (i, j2) exist in self.x
                if (i, j1) in self.x and (i, j2) in self.x:
                    self.model.addConstr(
                        self.x[i, j1] + self.x[i, j2] <= 1,
                        name=f"Conflict_{i}_{j1}_{j2}"
                    )

    def run(self):
        self.add_variables()
        self.add_constraints()
        self.set_objective()
        self.optimize()


