import networkx as nx
import numpy as np


class Preprocessing:
    def __init__(self, data, groups, subgroups):
        self.data = data
        self.groups = groups
        self.subgroups = subgroups
        self.n_c = {}
        self.J_c = {}
        self.conflict_graph = None


    def build_conflict_graph(self, conflict_set, activities):
        """
        Construct a conflict graph G from the set of conflicting activities T
        :param conflict_set: list of pairs of conflicting activities (int)
        :param activities: list of activities (int)
        :return G: conflict graph
        """
        G = nx.Graph()
        G.add_nodes_from(activities)
        G.add_edges_from(conflict_set)

        return G


    def find_maximal_cliques(self):
        """
        find all maximal cliques in the conflict graph G and store them in a binary matrix A
        """
        if self.conflict_graph is None:
            self.conflict_graph = self.build_conflict_graph(self.data['T'], self.data['J'])

        cliques = list(nx.find_cliques(self.conflict_graph))
        A = np.zeros((len(cliques), len(self.data['J'])), dtype=int)

        for row_idx, clique in enumerate(cliques):
            for col_idx, j in enumerate(self.data['J']):
                A[row_idx, col_idx] = 1 if j in clique else 0

        return A


    def build_eligibility_matrix(self, A, pattern):
        """
        Derive the clique matrix B from the selected pattern p
        :param A: Binary maximal cliques matrix
        :param pattern: binary tuple
        :return: B Clique matrix
        """
        p_array = np.array(pattern)
        if p_array.ndim != 1 or p_array.shape[0] != A.shape[1]:
            raise ValueError("the eligibility pattern must be a 1D array with the same length as the number of columns of A.")
        B = A * p_array
        return B


    def clean_clique_matrix(self, B):
        """
        Remove redundant cliques from clique matrix B

        Process:
          (1) Remove rows with <2 nonzero entries and remove duplicate rows (preserving first occurrence)
          (2) Compute C = B * B^T
          (3) Build matrix D where each row equals diag(C)
          (4) Compute E = C - D

          Phase 1:
             (5) Set the diagonal and lower triangular part of E to -1
             (6) Check the strictly upper triangular part of E for zeros
                 Any row i with a zero indicates that the corresponding clique is a subset of another
                 remove that row from B then update C, D and E

          Phase 2:
             (7) On the updated E, set the diagonal and the upper triangular part to -1
             (8) Check the strictly lower triangular part for zeros
                 Remove any row i with a zero then update C, D and E

          The procedure repeats (starting from Phase 1) until no further rows are removed
        """
        nonzero_counts = np.count_nonzero(B, axis=1)
        B_filtered = B[nonzero_counts >= 2, :]
        B_unique, indices = np.unique(B_filtered, axis=0, return_index=True)
        order = np.argsort(indices)
        B_clean = B_unique[order]

        changed = True
        iteration = 0
        while changed:
            changed = False
            m = B_clean.shape[0]
            if m == 0:
                break

            C = np.dot(B_clean, B_clean.T)
            diag = np.diag(C)
            D = np.repeat(diag.reshape(-1, 1), m, axis=1)
            E = C - D

            E_phase1 = E.copy()
            lower_idx = np.tril_indices(m, k=0)
            E_phase1[lower_idx] = -1

            upper_idx = np.triu_indices(m, k=1)
            rows_to_remove_phase1 = set()
            for i, j in zip(*upper_idx):
                if E_phase1[i, j] == 0:
                    rows_to_remove_phase1.add(i)

            if rows_to_remove_phase1:
                keep_idx = [i for i in range(m) if i not in rows_to_remove_phase1]
                B_clean = B_clean[keep_idx, :]
                C = C[keep_idx, :][:, keep_idx]
                m = B_clean.shape[0]
                diag = np.diag(C)
                D = np.repeat(diag.reshape(-1, 1), m, axis=1)
                E = C - D
                changed = True

            m = B_clean.shape[0]
            if m == 0:
                break

            E_phase2 = E.copy()
            upper_idx_full = np.triu_indices(m, k=0)
            E_phase2[upper_idx_full] = -1
            lower_idx_strict = np.tril_indices(m, k=-1)
            rows_to_remove_phase2 = set()

            for i, j in zip(*lower_idx_strict):
                if E_phase2[i, j] == 0:
                    rows_to_remove_phase2.add(i)

            if rows_to_remove_phase2:
                keep_idx = [i for i in range(m) if i not in rows_to_remove_phase2]
                B_clean = B_clean[keep_idx, :]
                C = C[keep_idx, :][:, keep_idx]
                m = B_clean.shape[0]
                diag = np.diag(C)
                D = np.repeat(diag.reshape(-1, 1), m, axis=1)
                E = C - D
                changed = True

            iteration += 1
        return B_clean


    def compute_group_constraints(self, B):
        """
        Given a processed clique matrix B for a specific eligibility pattern p
        returns:
          - n_c: the number of constraints for the group (i.e. number of rows in B)
          - J_c: a list of sets, where each set represents the activities corresponding
                 to one row of B
        """
        n_c = B.shape[0]
        J_c = []

        for row in B:
            conflict_set = {j for j, flag in zip(self.data['J'], row) if flag == 1}
            J_c.append(conflict_set)

        return n_c, J_c


    def run(self):
        self.conflict_graph = self.build_conflict_graph(self.data['T'], self.data['J'])

        A = self.find_maximal_cliques()
        if A.shape[0] == 0 or A.shape[1] == 0:
            print("Warning: Clique matrix is empty or improperly formed.")

        for pattern in self.groups.keys():
            B = self.build_eligibility_matrix(A, pattern)
            B_cleaned = self.clean_clique_matrix(B)
            n_c_val, J_c_val = self.compute_group_constraints(B_cleaned)
            self.n_c[pattern] = n_c_val
            self.J_c[pattern] = J_c_val

        self.data['n_c'] = self.n_c
        self.data['J_c'] = self.J_c


    def run_without_step_5(self):
        self.conflict_graph = self.build_conflict_graph(self.data['T'], self.data['J'])

        A = self.find_maximal_cliques()
        if A.shape[0] == 0 or A.shape[1] == 0:
            print("Warning: Clique matrix is empty or improperly formed.")

        for pattern in self.groups.keys():
            B = self.build_eligibility_matrix(A, pattern)
            n_c_val, J_c_val = self.compute_group_constraints(B)
            self.n_c[pattern] = n_c_val
            self.J_c[pattern] = J_c_val

        self.data['n_c'] = self.n_c
        self.data['J_c'] = self.J_c