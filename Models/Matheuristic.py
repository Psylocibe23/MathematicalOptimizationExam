import numpy as np
from sklearn.cluster import MiniBatchKMeans
from Models.Preprocessing import Preprocessing
from Models.LP import LPMix
from Models.iterative_algorithm import IterativeAlgo
from Models.alternative_LP import AlternativeLP


class Matheuristic:

    def __init__(self, data, k=20):
        self.data = data
        self.k = k  # Number of subgroups for k-means clustering


    @staticmethod
    def build_eligibility_patterns(I, J, J_i):
        """
        Build a binary eligibility matrix with |I| rows and |J| columns
        :param I: list of customers
        :param J: list of activities
        :param J_i: dictionary keys (customer) : values (list of eligible activities)
        :return: M binary eligibility matrix (np.array)
        """
        n_cust = len(I)
        n_act = len(J)
        M = np.zeros((n_cust, n_act), dtype=int)

        for row_idx, i in enumerate(I):
            for col_idx, j in enumerate(J):
                if j in J_i[i]:
                    M[row_idx, col_idx] = 1
                else:
                    M[row_idx, col_idx] = 0

        return M


    @staticmethod
    def sort_matrix(M):
        """
        :param M: binary matrix (2D numpy array)
        :return: sorted matrix according to reverse lexicographical order
        :return: sorted indices (to map back to customer IDs, since row order is changed)
        """
        sorted_indices = sorted(range(M.shape[0]), key=lambda i: tuple(M[i]), reverse=True)
        M_sorted = M[sorted_indices, :]

        return M_sorted, sorted_indices


    @staticmethod
    def create_eligibility_groups(M_sorted, sorted_indices, I):
        """
        Create eligibility groups where customers with the same eligibility pattern are grouped together

        Returns:
          Dictionary with keys as tuples (eligibility patterns) and values as lists of customer IDs
        """
        unique_rows, reverse_indices = np.unique(M_sorted, axis=0, return_inverse=True)
        groups = {}

        for unique_idx, unique_row in enumerate(unique_rows):
            group_indices_sorted = np.where(reverse_indices == unique_idx)[0]
            # Map these indices back to the original customer IDs
            original_indices = [sorted_indices[idx] for idx in group_indices_sorted]
            cust_ids = [I[orig_idx] for orig_idx in original_indices]
            groups[tuple(unique_row)] = cust_ids

        return groups


    @staticmethod
    def build_profit_matrix(I, J, e_ij, J_i):
        """
        Build a profit matrix whose entries are the expected profit for customer i and activity j
        """
        n_cust = len(I)
        n_act = len(J)
        A = np.zeros((n_cust, n_act), dtype=float)

        for row_idx, cust in enumerate(I):
            eligible = J_i[cust]
            for col_idx, act in enumerate(J):
                if act in eligible:
                    A[row_idx, col_idx] = e_ij.get((cust, act), 0)

        return A


    @staticmethod
    def cluster_eligibility_groups(A, groups, I, k=20):
        """
        For each eligibility group, perform clustering on the rows of A

        Returns:
          Dictionary mapping each eligibility pattern to a list of clusters (each cluster is a tuple of customer IDs)
        """
        all_subgroups = {}
        cust_to_index = {cust: idx for idx, cust in enumerate(I)}

        for pattern, cust_ids in groups.items():
            indices = [cust_to_index[cust] for cust in cust_ids]
            A_group = A[indices, :]
            k_group = min(k, A_group.shape[0])

            kmeans = MiniBatchKMeans(n_clusters=k_group, random_state=0)
            kmeans.fit(A_group)
            labels = kmeans.labels_

            clusters = {}

            for label in np.unique(labels):
                clusters[label] = [cust_ids[i] for i in range(len(cust_ids)) if labels[i] == label]

            all_subgroups[pattern] = [tuple(cluster) for cluster in clusters.values()]

        return all_subgroups


    def run(self):
        # Build the eligibility patterns
        M = Matheuristic.build_eligibility_patterns(self.data['I'], self.data['J'], self.data['J_i'])
        M_sorted, sorted_indices = Matheuristic.sort_matrix(M)
        # Build groups based on eligibility patterns
        groups = Matheuristic.create_eligibility_groups(M_sorted, sorted_indices, self.data['I'])
        # Build the profit matrix
        A = Matheuristic.build_profit_matrix(self.data['I'], self.data['J'], self.data['e_ij'], self.data['J_i'])
        # Cluster the eligibility groups
        subgroups = Matheuristic.cluster_eligibility_groups(A, groups, self.data['I'], self.k)
        # Retrieve the group level constraints dictionary from preprocessing
        prep = Preprocessing(self.data, groups, subgroups)
        prep.run()
        # Linear Program to assign customers at group level
        lp = LPMix(self.data, groups, subgroups)
        lp.run()
        self.lp = lp
        lp_sol = lp.lp_solution

        iter_algo = IterativeAlgo(self.data, groups, subgroups, lp_sol)
        iter_algo.populate_mappings_dict()
        iter_algo.generate_reverse_mapping()

        sequence = []

        for pattern, subgroup_list in subgroups.items():
            for k, subgroup in enumerate(subgroup_list):
                seq = iter_algo.prepare_subgroup_sequence(pattern, subgroup, k, lp_sol)
                if seq:
                    sequence.append(seq)
        # Assign customers to activities iteratively (customer level)
        iter_algo.iterative_algorithm(sequence)
        iter_algo.compute_objective_value()
        self.iterative_algorithm = iter_algo


    def run_without_new_modeling(self):
        """
            Run the Matheuristic model without the new modeling technique which define constraints for sets of
            conflicting activities (instead of forcing pairs of conflicting activities)
        """
        M = Matheuristic.build_eligibility_patterns(self.data['I'], self.data['J'], self.data['J_i'])
        M_sorted, sorted_indices = Matheuristic.sort_matrix(M)
        groups = Matheuristic.create_eligibility_groups(M_sorted, sorted_indices, self.data['I'])

        A = Matheuristic.build_profit_matrix(self.data['I'], self.data['J'], self.data['e_ij'], self.data['J_i'])
        subgroups = Matheuristic.cluster_eligibility_groups(A, groups, self.data['I'], self.k)

        prep = Preprocessing(self.data, groups, subgroups)
        prep.run()

        lp = AlternativeLP(self.data, groups, subgroups)
        lp.run()
        self.lp = lp
        lp_sol = lp.lp_solution

        iter_algo = IterativeAlgo(self.data, groups, subgroups, lp_sol)
        iter_algo.populate_mappings_dict()
        iter_algo.generate_reverse_mapping()

        sequence = []
        for pattern, subgroup_list in subgroups.items():
            for k, subgroup in enumerate(subgroup_list):
                seq = iter_algo.prepare_subgroup_sequence(pattern, subgroup, k, lp_sol)
                if seq:
                    sequence.append(seq)

        iter_algo.iterative_algorithm(sequence)
        iter_algo.compute_objective_value()
        self.iterative_algorithm = iter_algo
        
        
    def run_without_step5(self):
        """
            Run the Matheuristic model without step 5 of preprocessing (i.e. without removing redundant cliques
            before retrieving J_c[p] and n_c[p]
        """
        M = Matheuristic.build_eligibility_patterns(self.data['I'], self.data['J'], self.data['J_i'])
        M_sorted, sorted_indices = Matheuristic.sort_matrix(M)
        groups = Matheuristic.create_eligibility_groups(M_sorted, sorted_indices, self.data['I'])

        A = Matheuristic.build_profit_matrix(self.data['I'], self.data['J'], self.data['e_ij'], self.data['J_i'])
        subgroups = Matheuristic.cluster_eligibility_groups(A, groups, self.data['I'], self.k)

        prep = Preprocessing(self.data, groups, subgroups)
        prep.run_without_step_5()

        lp = LPMix(self.data, groups, subgroups)
        lp.run()
        self.lp = lp
        lp_sol = lp.lp_solution

        iter_algo = IterativeAlgo(self.data, groups, subgroups, lp_sol)
        iter_algo.populate_mappings_dict()
        iter_algo.generate_reverse_mapping()

        sequence = []
        for pattern, subgroup_list in subgroups.items():
            for k, subgroup in enumerate(subgroup_list):
                seq = iter_algo.prepare_subgroup_sequence(pattern, subgroup, k, lp_sol)
                if seq:
                    sequence.append(seq)

        iter_algo.iterative_algorithm(sequence)
        iter_algo.compute_objective_value()
        self.iterative_algorithm = iter_algo