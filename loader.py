import pandas as pd
import os
from collections import defaultdict


class DataLoader:
    def __init__(self, base_directory):
        self.base_directory = base_directory
        self.load_data(self.base_directory)
        self.initialize_data_structure()
        self.generate_conflict_rules()
        self.generate_conflict_set()
        self.generate_activities_constraints_sets()
        self.check_minimum_contact()

    def load_data(self, base_directory):
        """
            load data from a CSV file and explode columns
        """
        self.activities = self.read_csv_file('table1.csv')
        self.customers = self.read_csv_file('table2.csv')
        self.constraints = self.read_csv_file('table3.csv')
        self.campaigns = self.read_csv_file('table4.csv')

        # Explode columns for table1.csv
        if 'Act_TargetProducts' in self.activities.columns:
            activities_target_products = self.activities['Act_TargetProducts'].str.split(';', expand=True)
            # Rename the columns
            activities_target_products.columns = [f"Act_TargetProduct{i}" for i in range(1, activities_target_products.shape[1] + 1)]
            # Concatenate the new columns to the original DataFrame
            self.activities = pd.concat([self.activities.drop('Act_TargetProducts', axis=1), activities_target_products], axis=1)

        # Explode columns for table3.csv
        if 'Camp_Channels' in self.constraints.columns:
            constraints_channels = self.constraints['Camp_Channels'].str.split(';', expand=True)
            constraints_channels.columns = [f"Camp_Channel{i}" for i in range(1, constraints_channels.shape[1] + 1)]
            self.constraints = pd.concat([self.constraints.drop('Camp_Channels', axis=1), constraints_channels], axis=1)

        if 'Camp_TargetProducts' in self.constraints.columns:
            constraints_target_products = self.constraints['Camp_TargetProducts'].str.split(';', expand=True)
            constraints_target_products.columns = [f"Camp_TargetProduct{i}" for i in range(1, constraints_target_products.shape[1] + 1)]
            self.constraints = pd.concat([self.constraints.drop('Camp_TargetProducts', axis=1), constraints_target_products], axis=1)

        self.activities.sort_values(by="Act_Activity", inplace=True)

    def read_csv_file(self, file_name):
        """
        Read CSV file and return a pd DataFrame
        input: file name (str)
        output: pd DataFrame
        """
        file_path = os.path.join(self.base_directory, file_name)
        try:
            if os.path.exists(file_path):
                return pd.read_csv(file_path)
            else:
                print(f"File {file_path} does not exist.")
                return pd.DataFrame()
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
            return pd.DataFrame()

    def check_minimum_contact(self):
        self.data['has_minimum_contact'] = 'Minimum contact' in self.constraints['Camp_Type'].values

    def initialize_data_structure(self):
        """
        Extrapolate sets, bounds and number of constraints from the Datasets
        input: a Dataframe
        output: populate the data dictionary (self.data)
        """
        # A list of all customers
        self.I = self.customers['Cust_Customer'].unique().tolist()
        # A list of all activities
        self.J = self.activities['Act_Activity'].unique().tolist()
        # A dictionary containing the response probabilities of customer i wrt activity j
        self.q_ij = {
            (row['Cust_Customer'], row['Cust_Activity']): row['Cust_ResponseProbability']
            for idx, row in self.customers.iterrows()
        }
        # Sum of the q_ij values, needed for the mblp formulation
        self.q_tot = sum(self.q_ij.values())
        # A dictionary containing the expected profit of customer i for activity j
        self.e_ij = {
            (row['Cust_Customer'], row['Cust_Activity']): row['Cust_ExpectedProfit']
            for idx, row in self.customers.iterrows()
        }
        # A dictionary containing the cost of each activity
        self.c_j = {
            row['Act_Activity']: row['Act_Cost']
            for idx, row in self.activities.iterrows()
        }
        # Penalty coefficients (set to max absolute expected profit of the dataset as in the article)
        max_abs_net = max(abs(self.e_ij[(i, j)]) for (i, j) in self.e_ij)
        # Penalty coefficients based on the maximum absolute expected profit
        self.alpha = max_abs_net
        self.beta = max_abs_net
        self.gamma = max_abs_net
        self.delta = max_abs_net

        def build_J_i():
            # Associate each customer to its eligible activities
            activities_dict = defaultdict(list)
            for idx, row in self.customers.iterrows():
                activities_dict[row['Cust_Customer']].append(row['Cust_Activity'])
            return activities_dict

        # A dictionary with customers (keys) and related eligible activities (values: lists)
        self.J_i = build_J_i()

        def build_I_j():
            # Associates each activity with the eligible customers
            customers_dict = defaultdict(list)
            for idx, row in self.customers.iterrows():
                customers_dict[row['Cust_Activity']].append(row['Cust_Customer'])
            return customers_dict

        # A dictionary with activities (keys) and related eligible customers (values: lists)
        self.I_j = build_I_j()

        def count_constraints(constraint_name):
            """
            Counts the number of constraints for the dataset
            input: constraint_name (str)
            output: number of appearances of constraint_name in constraints.csv (int)
            """
            count = 0
            for row in self.constraints['Camp_Type']:
                if row == constraint_name:
                    count += 1
            return count

        # Number of Minimum assignment constraints
        self.n_a = count_constraints('Minimum assignment')
        # Number of Maximum assignment constraints
        self.n_a_bar = count_constraints('Maximum assignment')
        # Number of Budget constraints
        self.n_b = count_constraints('Budget')
        # Number of Minimum sales constraints
        self.n_s = count_constraints('Minimum sales')
        # Number of Maximum sales constraints
        self.n_s_bar = count_constraints('Maximum sales')
        # Number of minimum contact constraints
        self.n_m = count_constraints('Minimum contact')
        # Number of Maximum contact constraints
        self.n_m_bar = count_constraints('Maximum contact')

        def populate_bound_dictionaries(constraint_type):
            """
            For each constraint type create a dictionary with the relative bounds
            """
            bound_dict = {}
            num_channels = sum(1 for col in self.constraints.columns if col.startswith("Camp_Channel"))
            channel_columns = [f"Camp_Channel{i}" for i in range(1, num_channels + 1)]
            filtered_rows = self.constraints[self.constraints['Camp_Type'] == constraint_type]

            for _, row in filtered_rows.iterrows():
                channels = []
                for col in channel_columns:
                    if pd.notna(row[col]):
                        for val in row[col].split(';'):
                            cleaned_value = val.strip()
                            if cleaned_value == "ALL":
                                channels = ["ALL"]
                                break
                            else:
                                channels.append(cleaned_value)
                        if channels == ["ALL"]:
                            break
                channels = sorted(set(channels))

                if 'Camp_TargetProducts' in self.constraints.columns and pd.notna(row['Camp_TargetProducts']):
                    target = row['Camp_TargetProducts'].strip()
                elif 'Camp_TargetProduct1' in self.constraints.columns and pd.notna(row['Camp_TargetProduct1']):
                    target = row['Camp_TargetProduct1'].strip()
                elif 'Camp_TargetProduct2' in self.constraints.columns and pd.notna(row['Camp_TargetProduct2']):
                    target = row['Camp_TargetProduct2'].strip()
                else:
                    target = "ALL"
                if target == "ALL":
                    if channels == ["ALL"]:
                        key = "ALL"
                    else:
                        key = channels[0] if len(channels) == 1 else tuple(channels)
                else:
                    try:
                        if constraint_type in ['Minimum sales', 'Maximum sales']:
                            target_val = int(target)
                        else:
                            target_val = target
                    except:
                        target_val = target
                    if channels == ["ALL"]:
                        key = ("ALL", target_val)
                    else:
                        ch_key = channels[0] if len(channels) == 1 else tuple(channels)
                        key = (ch_key, target_val)
                bound_dict[key] = row['Camp_Bound']
            return bound_dict

        # Budget assignment bounds dictionary
        self.b_b = populate_bound_dictionaries('Budget')
        # Minimum assignment bounds dictionary
        self.b_a = populate_bound_dictionaries('Minimum assignment')
        # Maximum assignment bounds dictionary
        self.b_a_bar = populate_bound_dictionaries('Maximum assignment')
        # Minimum sales bounds dictionary
        self.b_s = populate_bound_dictionaries('Minimum sales')
        # Maximum sales bounds dictionary
        self.b_s_bar = populate_bound_dictionaries('Maximum sales')
        # Minimum contact bounds dictionary
        self.b_m = populate_bound_dictionaries('Minimum contact')
        # Maximum contact bounds dictionary
        self.b_m_bar = populate_bound_dictionaries('Maximum contact')

        self.data = {
            'I': self.I,
            'J': self.J,
            'q_ij': self.q_ij,
            'e_ij': self.e_ij,
            'c_j': self.c_j,
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'delta': self.delta,
            'J_i': self.J_i,
            'I_j': self.I_j,
            'n_a': self.n_a,
            'n_a_bar': self.n_a_bar,
            'n_b': self.n_b,
            'n_s': self.n_s,
            'n_s_bar': self.n_s_bar,
            'n_m': self.n_m,
            'n_m_bar': self.n_m_bar,
            'b_b': self.b_b,
            'b_a': self.b_a,
            'b_a_bar': self.b_a_bar,
            'b_s': self.b_s,
            'b_s_bar': self.b_s_bar,
            'b_m': self.b_m,
            'b_m_bar': self.b_m_bar,
            'q_tot': self.q_tot
        }

    def generate_conflict_rules(self):
        """
        Read conflict rules from table4.csv, used to generate T (set of pairs of conflicting activities)
        """
        self.conflict_rules = []

        for _, row in self.campaigns.iterrows():
            if pd.isna(row['Const_Channel1']) or pd.isna(row['Const_TargetProduct1']) \
                    or pd.isna(row['Const_Channel2']) or pd.isna(row['Const_TargetProduct2']) \
                    or pd.isna(row['Const_Lag']):
                continue

            chan1 = row['Const_Channel1'].strip()
            prod1 = row['Const_TargetProduct1'].strip()
            chan2 = row['Const_Channel2'].strip()
            prod2 = row['Const_TargetProduct2'].strip()
            lag = int(row['Const_Lag'])
            rule_tuple = (chan1, prod1, chan2, prod2, lag)
            self.conflict_rules.append(rule_tuple)

        return self.conflict_rules

    def generate_conflict_set(self):
        """
        Builds the set T of conflicting activity pairs (j1, j2) by applying each
        conflict rule to each pair of activities in self.activities
        """
        self.T = []
        product_columns = [c for c in self.activities.columns if c.startswith("Act_TargetProduct")]

        # Compare each pair of distinct activity rows
        for i, row1 in self.activities.iterrows():
            for j, row2 in self.activities.iterrows():
                if i >= j:
                    continue  # skip same or duplicated pairs

                for conflict_rule in self.conflict_rules:
                    # 1) Channel check
                    c1_ok = (row1['Act_Channel'] == conflict_rule[0] or conflict_rule[0] == 'ALL')
                    c2_ok = (row2['Act_Channel'] == conflict_rule[2] or conflict_rule[2] == 'ALL')
                    # 2) Day difference check
                    days_within_lag = abs(row1['Act_Day'] - row2['Act_Day']) <= conflict_rule[4]
                    # 3) Product checks
                    row1_products = []
                    for pcol in product_columns:
                        val = row1.get(pcol, None)
                        if pd.notna(val):
                            row1_products.append(val.strip())

                    row2_products = []
                    for pcol in product_columns:
                        val = row2.get(pcol, None)
                        if pd.notna(val):
                            row2_products.append(val.strip())

                    p1_ok = (conflict_rule[1] == 'ALL' or conflict_rule[1] in row1_products)
                    p2_ok = (conflict_rule[3] == 'ALL' or conflict_rule[3] in row2_products)

                    if c1_ok and c2_ok and days_within_lag and p1_ok and p2_ok:
                        conflicting_pair = (row1['Act_Activity'], row2['Act_Activity'])
                        self.T.append(conflicting_pair)

        self.T = sorted(list(set(self.T)), key=lambda pair: (pair[0], pair[1]))
        self.data['T'] = self.T

        return self.T

    def generate_activities_constraints_sets(self):
        """
        Build dictionaries of activity sets for each constraint type
        For example, self.data['J_a'][l] = {set of activities} for the l-th minimum assignment constraint
        """
        types = {
            'Minimum assignment': 'J_a',
            'Maximum assignment': 'J_a_bar',
            'Budget': 'J_b',
            'Minimum sales': 'J_s',
            'Maximum sales': 'J_s_bar',
            'Maximum contact': 'J_m_bar',
            'Minimum contact': 'J_m',
        }

        num_channels = sum(1 for col in self.constraints.columns if col.startswith("Camp_Channel"))
        camp_channels = [f"Camp_Channel{i}" for i in range(1, num_channels + 1)]

        # For each constraint type, build a dictionary of sets of activities
        for ctype, key in types.items():
            self.data[key] = {}
            filtered_constraints = self.constraints[self.constraints['Camp_Type'] == ctype]
            if len(filtered_constraints) == 0:
                # If no constraints of this type exist in the dataset, skip
                continue

            counter = 0
            for _, row in filtered_constraints.iterrows():
                channels = set()
                for channel_col in camp_channels:
                    channel_value = row[channel_col]
                    if pd.notna(channel_value):
                        for value in channel_value.split(';'):
                            cleaned_value = value.strip()
                            if cleaned_value != 'ALL':
                                channels.add(cleaned_value)
                            else:
                                channels = set(self.activities['Act_Channel'].unique())
                                break

                target_products = set()
                if pd.notna(row['Camp_TargetProduct1']) and row['Camp_TargetProduct1'].strip() != 'ALL':
                    target_products.add(row['Camp_TargetProduct1'].strip())
                    if 'Camp_TargetProduct2' in self.constraints.columns:
                        tp2 = row['Camp_TargetProduct2']
                        if pd.notna(tp2) and tp2.strip() != '':
                            target_products.add(tp2.strip())
                else:
                    for col in ['Act_TargetProduct1', 'Act_TargetProduct2', 'Act_TargetProduct3']:
                        if col in self.activities.columns:
                            cleaned_vals = [val.strip() for val in self.activities[col].dropna().astype(str).unique()]
                            target_products.update(cleaned_vals)

                activity_conditions = self.activities['Act_Channel'].isin(channels)
                if 'Act_TargetProduct1' in self.activities.columns:
                    activity_conditions &= self.activities['Act_TargetProduct1'].astype(str).isin(target_products)
                if 'Act_TargetProduct2' in self.activities.columns:
                    activity_conditions |= self.activities['Act_TargetProduct2'].astype(str).isin(target_products)

                time_condition = (
                        (self.activities['Act_Day'] >= row['Camp_StartDay']) &
                        (self.activities['Act_Day'] <= row['Camp_EndDay'])
                )

                # Collect all activities that satisfy both conditions
                constrained_activities = set(
                    self.activities.loc[activity_conditions & time_condition, 'Act_Activity']
                )

                self.data[key][counter] = constrained_activities
                counter += 1