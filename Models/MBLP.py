from Models.MBLP_class import base_MBLP


class MBLP(base_MBLP):
    # The first MBLP presented in the article
    def __init__(self, data):
        super().__init__('Mixed Binary Linear Model', data)


    def add_constraints(self):
        """
        Add the conflict constraint to the base MBLP
        """
        super().add_constraints()
        # Initialize conflict constraints count
        conflicts_count = 0
        # Find customers eligible for pairs of conflicting activities
        for (j1, j2) in self.data['T']:
            # Get customers eligible for both activities j1, j2
            eligible_customers = set(self.data['I_j'].get(j1, [])) & set(self.data['I_j'].get(j2, []))

            for i_float in eligible_customers:
                i = int(i_float)
                if (i, j1) in self.x and (i, j2) in self.x:
                    self.model.addConstr(
                        self.x[i, j1] + self.x[i, j2] <= 1
                    )
                    conflicts_count += 1

        self.model.update()
        print(f"Total number of conflict constraints: {conflicts_count}")


    def run(self):
        self.add_variables()
        self.add_constraints()
        self.set_objective()
        self.optimize()