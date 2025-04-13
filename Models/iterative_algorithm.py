import math
from Models.Preprocessing import Preprocessing
from collections import defaultdict
from collections import deque

class IterativeAlgo:
    def __init__(self, data, groups, subgroups, lp_solution):
        self.data = data
        self.groups = groups
        self.subgroups = subgroups
        self.lp_solution = lp_solution

        prep = Preprocessing(self.data, self.groups, self.subgroups)
        self.G = prep.build_conflict_graph(self.data['T'], self.data['J'])
        # Dictionaries to map customers and patterns to activities
        self.eligibility_mapping = {pattern: [j for j, flag in zip(self.data['J'], pattern) if flag == 1] for pattern in self.groups.keys()}
        self.customer_eligibilities = {customer: [j for j in self.data['J_i'].get(customer, [])] for customer in self.data['I']}
        # Dictionaries to keep track of assignments during iterations
        self.assignment_counts = {customer: 0 for customer in self.data['I']}
        self.assignment_dict = {customer: [] for customer in self.data['I']}
        # Dictionaries to enforce hard customer-level constraints
        self.max_sales_dict = {}
        for raw_key, bound in self.data['b_s_bar'].items():
            key = raw_key if isinstance(raw_key, tuple) else raw_key
            self.max_sales_dict[key] = bound

        self.budget_dict = {}
        for raw_key, bound in self.data['b_b'].items():
            key = raw_key if isinstance(raw_key, tuple) else raw_key
            self.budget_dict[key] = bound

        self.max_assignments_dict = {}
        for raw_key, bound in self.data['b_a_bar'].items():
            key = raw_key if isinstance(raw_key, tuple) else raw_key
            self.max_assignments_dict[key] = bound

        self.max_contact_dict = {}
        for raw_key, bound in self.data['b_m_bar'].items():
            key = raw_key if isinstance(raw_key, tuple) else raw_key
            self.max_contact_dict[key] = bound

        self.mappings = {}


    def generate_key_mapping(self, b_something, J_something):
        key_mapping = {}
        b_keys = list(b_something.keys())
        J_keys = list(J_something.keys())

        if len(b_keys) != len(J_keys):
            raise ValueError(f"Mismatch in lengths: {b_something.keys()} vs. {J_something.keys()}")

        for i, b_key in enumerate(b_keys):
            key_mapping[b_key] = J_keys[i]

        return key_mapping


    def generate_reverse_mapping(self):
        """
        Creates an inverted index so that for each activity j, we can directly find
        which bound_keys in b_something contain that j
        """
        self.b_s_bar_inverted = defaultdict(list)
        for bound_key in self.data['b_s_bar']:
            j_sub_idx = self.mappings['b_s_bar'][bound_key]
            act_set = self.data['J_s_bar'][j_sub_idx]
            for a in act_set:
                self.b_s_bar_inverted[a].append(bound_key)

        self.b_a_bar_inverted = defaultdict(list)
        for bound_key in self.data['b_a_bar']:
            j_sub_idx = self.mappings['b_a_bar'][bound_key]
            act_set = self.data['J_a_bar'][j_sub_idx]
            for a in act_set:
                self.b_a_bar_inverted[a].append(bound_key)

        self.b_b_inverted = defaultdict(list)
        for bound_key in self.data['b_b']:
            j_sub_idx = self.mappings['b_b'][bound_key]
            act_set = self.data['J_b'][j_sub_idx]
            for a in act_set:
                self.b_b_inverted[a].append(bound_key)


    def populate_mappings_dict(self):
        self.mappings = {
            'b_a': self.generate_key_mapping(self.data['b_a'], self.data['J_a']),
            'b_b': self.generate_key_mapping(self.data['b_b'], self.data['J_b']),
            'b_a_bar': self.generate_key_mapping(self.data['b_a_bar'], self.data['J_a_bar']),
            'b_s': self.generate_key_mapping(self.data['b_s'], self.data['J_s']),
            'b_s_bar': self.generate_key_mapping(self.data['b_s_bar'], self.data['J_s_bar']),
            'b_m_bar': self.generate_key_mapping(self.data['b_m_bar'], self.data['J_m_bar']),
        }


    def prepare_subgroup_sequence(self, pattern, subgroup, subgroup_index, lp_solution):
        """
        Given a subgroup it returns an ordered sequence of subgroup-activity pairs to be passed
        to the iterative algorithm
        :param pattern: tuple
        :param subgroup:
        :param subgroup_index: int
        :param lp_solution: variables assignments of the LP model
        :return: list of subgroup-activity pairs
        """
        eligible_acts = self.eligibility_mapping.get(pattern, [])
        # keep only the activities that have LP val >=1
        filtered = [j for j in eligible_acts if lp_solution.get((pattern, subgroup_index, j), 0) >= 1]
        if not filtered:
            return []

        G_sub = self.G.subgraph(filtered)
        degs = dict(G_sub.degree())
        max_deg = max(degs.values())
        top_nodes = [j for j, d in degs.items() if d == max_deg]
        initial = min(top_nodes)
        seq = [initial]
        remain = set(filtered) - {initial}

        while remain:
            best_act = None
            best_conn = -1
            for a in remain:
                conn = sum(1 for neigh in G_sub.neighbors(a) if neigh in seq)
                if conn > best_conn or (conn == best_conn and (best_act is None or a < best_act)):
                    best_act = a
                    best_conn = conn
            seq.append(best_act)
            remain.remove(best_act)

        return [(pattern, subgroup_index, act) for act in seq]


    def check_hard_constraints(self, activity, customer):
        """
        Given an activity and a customer it returns True if assigning customer to activity would
        violate any hard constraint
        """
        for bound_key in self.data['b_m_bar'].keys():
            j_sub_idx = self.mappings['b_m_bar'][bound_key]
            act_set = self.data['J_m_bar'][j_sub_idx]
            if activity in act_set:
                max_count = self.max_contact_dict[bound_key]
                num_contacts = self.assignment_counts[customer]
                if max_count - num_contacts <= 0:
                    return True

        return False


    def get_conflicting_activities(self, activity):
        conflicts = []

        for (j1, j2) in self.data['T']:
            if activity == j1:
                conflicts.append(j2)
            elif activity == j2:
                conflicts.append(j1)

        return set(conflicts)

    def find_candidates(self, pattern, subgroup_idx, activity):
        """
        Returns a list of customers that can be assigned to an activity without violating hard constraints
        """
        candidates = []

        for customer in self.subgroups[pattern][subgroup_idx]:
            if activity in self.customer_eligibilities[customer]:
                if not self.check_hard_constraints(activity, customer):
                    candidates.append(customer)

        return candidates


    def assign_candidates(self, pattern, subgroup_idx, activity, candidates):
        """
        Given a list of candidates it assigns the floor(x_pkj) candidates with the highest expected profit
        for the activity considered
        """
        lp_val = self.lp_solution.get((pattern, subgroup_idx, activity), 0.0)
        num_to_assign = math.floor(lp_val)
        if num_to_assign < 1:
            return []
        # Sort candidates in ascending order of potential profit
        sorted_candidates = sorted(candidates,
                                   key=lambda i: self.data['e_ij'].get((i,activity), 0.0),
                                   reverse=True
                                   )

        assigned_count = min(num_to_assign, len(sorted_candidates))
        assigned_customers = sorted_candidates[:assigned_count]

        for cust in assigned_customers:
            self.assignment_dict[cust].append(activity)
            self.assignment_counts[cust] += 1

        return assigned_customers


    def update_eligibilities(self, assigned_customers, activity):
        conflicts = self.get_conflicting_activities(activity)

        for customer in assigned_customers:
            if activity in self.customer_eligibilities[customer]:
                self.customer_eligibilities[customer].remove(activity)
            for act in conflicts:
                if act in self.customer_eligibilities[customer]:
                    self.customer_eligibilities[customer].remove(act)


    def update_constraints_dictionaries(self, assigned_customers, activity):
        for customer in assigned_customers:
            if activity in self.b_b_inverted:
                bound_keys = self.b_b_inverted[activity]
                for bound_key in bound_keys:
                    self.budget_dict[bound_key] -= self.data['c_j'].get(activity, 0.0)
            if activity in self.b_s_bar_inverted:
                bound_keys = self.b_s_bar_inverted[activity]
                for bound_key in bound_keys:
                    self.max_sales_dict[bound_key] -= self.data['q_ij'].get((customer, activity), 0.0)
            if activity in self.b_a_bar_inverted:
                bound_keys = self.b_a_bar_inverted[activity]
                for bound_key in bound_keys:
                    self.max_assignments_dict[bound_key] -= 1


    def iterative_algorithm(self, sequences):
        """
        Iteratively assign customers to eligible activities based on the LP solution without violating hard customer
        level constraints
        """
        queue = deque(sequences)

        while queue:
            current_sequence = queue.popleft()
            if not current_sequence:
                continue  # Skip empty sequences

            for current_pair in current_sequence:
                pattern, subgroup_idx, activity = current_pair
                candidates = self.find_candidates(pattern, subgroup_idx, activity)
                if candidates:
                    assigned_candidates = self.assign_candidates(pattern, subgroup_idx, activity, candidates)
                    self.update_eligibilities(assigned_candidates, activity)
                    self.update_constraints_dictionaries(assigned_candidates, activity)
                else:
                    print(f"No candidates found for activity {activity}")

        return self.assignment_dict


    def compute_potential_profit(self):
        total = 0

        for customer, list_act in self.assignment_dict.items():
            for act in list_act:
                total += self.data['e_ij'].get((customer, act), 0.0)
        print(f"Total potential profit:{total}")

        return total


    def compute_penalties(self):
        # Total penalty
        penalty = 0.0
        # Min assignment penalty
        z_a = {key: 0 for key in self.data['b_a']}
        assignment_totals = {key: 0 for key in self.data['b_a']}
        # Count assignments for each activity set corresponding to each constraint
        for bound_key in self.data['b_a']:
            j_sub_idx = self.mappings['b_a'][bound_key]
            act_set = self.data['J_a'][j_sub_idx]
            for customer, activities in self.assignment_dict.items():
                assignment_totals[bound_key] += sum(1 for a in activities if a in act_set)

        # Compute penalties for each constraint
        penalty_a = {}
        for bound_key in self.data['b_a']:
            target_assignments = self.data['b_a'][bound_key]
            actual_assignments = assignment_totals[bound_key]
            if actual_assignments < target_assignments:
                penalty_a[bound_key] = (actual_assignments - target_assignments) * self.data['alpha']
            else:
                penalty_a[bound_key] = 0
            penalty += penalty_a[bound_key]
        # Min sales penalty
        z_s = {key: 0 for key in self.data['b_s']}
        min_sales_total = {key: 0 for key in self.data['b_s']}
        for bound_key in self.data['b_s']:
            j_sub_idx = self.mappings['b_s'][bound_key]
            act_set = self.data['J_s'][j_sub_idx]
            for customer, activities in self.assignment_dict.items():
                min_sales_total[bound_key] += sum(self.data['q_ij'].get((customer, a), 0.0) for a in activities if a in act_set)
        penalty_s = {}
        for bound_key in self.data['b_s']:
            target_min_sales = self.data['b_s'][bound_key]
            actual_min_sales = min_sales_total[bound_key]
            if target_min_sales > actual_min_sales:
                penalty_s[bound_key] = (actual_min_sales - target_min_sales) * self.data['beta']
            else:
                penalty_s[bound_key] = 0
            penalty += penalty_s[bound_key]
        # Max sales penalty
        z_s_bar = {key: 0 for key in self.data['b_s_bar']}
        max_sales_total = {key: 0 for key in self.data['b_s_bar']}
        for bound_key in self.data['b_s_bar']:
            j_sub_idx = self.mappings['b_s_bar'][bound_key]
            act_set = self.data['J_s_bar'][j_sub_idx]
            for customer, activities in self.assignment_dict.items():
                max_sales_total[bound_key] += sum(self.data['q_ij'].get((customer, a), 0.0) for a in activities if a in act_set)
        penalty_s_bar = {}
        for bound_key in self.data['b_s_bar']:
            target_max_sales = self.data['b_s_bar'][bound_key]
            actual_max_sales = max_sales_total[bound_key]
            if target_max_sales < actual_max_sales:
                penalty_s_bar[bound_key] = (actual_max_sales - target_max_sales) * self.data['gamma']
            else:
                penalty_s_bar[bound_key] = 0
            penalty += penalty_s_bar[bound_key]

        return penalty


    def compute_objective_value(self):
        total_profit = self.compute_potential_profit()
        penalty = self.compute_penalties()
        obj_val = total_profit - abs(penalty)
        print(f"Final objective = {obj_val}")

        return obj_val