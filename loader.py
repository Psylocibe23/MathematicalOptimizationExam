import pandas as pd
import os
from collections import defaultdict

class DataLoader():
    def __init__(self, base_directory):
        self.base_directory = base_directory
        self.load_data(self.base_directory)
        self.initialize_data_structure()
        self.generate_conflict_rules()
        self.generate_conflict_set()
        self.generate_activities_constraints_sets()


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


    def initialize_data_structure(self):
        """
        Extrapolate sets, bounds and number of constraints from the Datasets
        input: a Dataframe
        output: data dictionary
        """

        # A list of all customers
        self.I = self.customers['Cust_Customer'].unique().tolist()
        # A list of all activities
        self.J = self.activities['Act_Activity'].unique().tolist()
        # A dictionary containing the response probabilities of customer i wrt activity j
        self.q_ij = {
            (row['Cust_Customer'], row['Cust_Activity']) : row['Cust_ResponseProbability']
            for idx, row in self.customers.iterrows()
        }
        # A dictionary containing the expected profit of customer i for activity j
        self.e_ij = {
            (row['Cust_Customer'], row['Cust_Activity']) : row['Cust_ExpectedProfit']
            for idx, row in self.customers.iterrows()
        }
        # A dictionary containing the cost of each activity
        self.c_j = {
            row['Act_Activity'] : row['Act_Cost']
            for idx, row in self.activities.iterrows()
        }
        # Penalty coefficients (set to max absolute expected profit of the dataset as in the article)
        max_abs_e_ij = max(abs(profit) for profit in self.e_ij.values())
        self.alpha = max_abs_e_ij
        self.beta = max_abs_e_ij
        self.gamma = max_abs_e_ij


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
        # Number of Minimum contact constraints
        """self.n_m = count_constraints('Minimum contact') """ #(MIN CONTACT RULE NON PRESENTE NEI DATI FORNITI DAGLI AUTORI)
        # Number of Maximum contact constraints
        self.n_m_bar = count_constraints('Maximum contact')

        def populate_bound_dictionaries(constraint_type):
            """
            Create a dictionary with the campaign channels for keys and bounds (float) for values
            input: constraint_type (str)
            output: bounds dictionary
            """
            bound_dict = {}
            channel_columns = ['Camp_Channel1', 'Camp_Channel2', 'Camp_Channel3']  # List of possible channel columns
            filtered_rows = self.constraints[self.constraints['Camp_Type'] == constraint_type]

            for _, row in filtered_rows.iterrows():
                channels = [row[col].strip() for col in channel_columns if
                            pd.notna(row[col])]
                channels = sorted(set(channels))  # Remove duplicates and sort to standardize across entries
                channel_key = tuple(channels)
                if channel_key:
                    if len(channel_key) == 1:
                        channel_key = channel_key[0]
                    bound_dict[channel_key] = row['Camp_Bound']

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
        # Maximum contact bounds dictionary
        self.b_m_bar = populate_bound_dictionaries('Maximum contact')
        # Minimum contact bounds dictionary
        """self.b_m = populate_bound_dictionaries('Minimum contact')""" #(MIN CONTACT RULE NON PRESENTE NEI DATI FORNITI DAGLI AUTORI)




        self.data = {
            'I': self.I,
            'J': self.J,
            'q_ij': self.q_ij,
            'e_ij': self.e_ij,
            'c_j': self.c_j,
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'J_i': self.J_i,
            'I_j': self.I_j,
            'n_a': self.n_a,
            'n_a_bar': self.n_a_bar,
            'n_b': self.n_b,
            'n_s': self.n_s,
            'n_s_bar': self.n_s_bar,
            'n_m_bar': self.n_m_bar,
            'b_b': self.b_b,
            'b_a': self.b_a,
            'b_a_bar': self.b_a_bar,
            'b_s': self.b_s,
            'b_s_bar': self.b_s_bar,
            'b_m_bar': self.b_m_bar
        }

    def generate_conflict_rules(self):
        """
        Extrapolate the conflicts rules from table3.csv
        output: a list containing the conflict rules for the dataframe stored in tuples (channel1, channel2, time_lag)
        """
        self.conflict_rules = []
        for _, row in self.campaigns.iterrows():  # Unpack the tuple into index (_) and row (Series)
            rule_tuple = (row['Const_Channel1'], row['Const_Channel2'], row['Const_Lag'])  # Access columns from the Series
            self.conflict_rules.append(rule_tuple)

        return self.conflict_rules


    def generate_conflict_set(self):
        """
        Enforce the conflict rules on the activities set in order to identify the conflicting activities
        input: conflict_rules (list of tuples)
        output: T set of pairs of conflicting activities
        """
        self.T = []
        for i, row1 in self.activities.iterrows():
            for j, row2 in self.activities.iterrows():
                # ensure we are not comparing a row with itself
                if i >= j:
                    continue
                for conflict_rule in self.conflict_rules:
                    if (
                        row1['Act_Channel'] == conflict_rule[0]
                        and row2['Act_Channel'] == conflict_rule[1]
                        and abs(row1['Act_Day'] - row2['Act_Day']) < conflict_rule[2]
                    ):
                        conflicting_pair = (row1['Act_Activity'], row2['Act_Activity'])
                        self.T.append(conflicting_pair)

        self.T = list(set(self.T))  # first convert to set in order to remove eventual duplicates
        self.data['T'] = self.T
        return self.T


    def generate_activities_constraints_sets(self):
        """
        Enforce the business and contact constraints of table3 (self.constraints) and generates the associated
        sets of activities
        """

        # Define constraint types and their corresponding keys in the data dictionary
        types = {
            'Minimum assignment': 'J_a',
            'Maximum assignment': 'J_a_bar',
            'Budget': 'J_b',
            'Minimum sales': 'J_s',
            'Maximum sales': 'J_s_bar',
            'Maximum contact': 'J_m_bar',
        }

        camp_channels = ['Camp_Channel1', 'Camp_Channel2', 'Camp_Channel3']

        for ctype, key in types.items():
            self.data[key] = {}  # Initialize the dictionary for this constraint type
            filtered_constraints = self.constraints[self.constraints['Camp_Type'] == ctype]

            # Initialize a counter to track the index for multiple constraints
            counter = 0

            for _, row in filtered_constraints.iterrows():
                # Check if the constraint applies to ALL activities
                is_full_time = (row['Camp_StartDay'] == 1) and (row['Camp_EndDay'] == 120)
                is_all_channels = all(row[channel] == 'ALL' or pd.isna(row[channel]) for channel in camp_channels)
                is_all_products = row['Camp_TargetProduct1'] == 'ALL'

                if is_full_time and is_all_channels and is_all_products:
                    # If all activities are included, skip filtering and include all activities
                    constrained_activities = set(self.activities['Act_Activity'])
                else:
                    # Determine applicable channels
                    channels = set()
                    for channel_col in camp_channels:
                        channel_value = row[channel_col]
                        if pd.notna(channel_value) and channel_value != 'ALL':
                            channels.add(channel_value)
                        elif channel_value == 'ALL':
                            channels.update(self.activities['Act_Channel'].unique().tolist())

                    # Determine applicable target products
                    target_products = set()
                    if pd.notna(row['Camp_TargetProduct1']) and row['Camp_TargetProduct1'] != 'ALL':
                        target_products.add(str(row['Camp_TargetProduct1']))
                    else:
                        target_products.update(
                            str(product)
                            for product in self.activities['Act_TargetProduct1'].unique()
                            if pd.notna(product)
                        )
                        target_products.update(
                            str(product)
                            for product in self.activities['Act_TargetProduct2'].unique()
                            if pd.notna(product)
                        )

                    # Determine activities within the campaign's time period
                    activities_time_constrained = set(
                        self.activities[
                            (self.activities['Act_Day'] >= row['Camp_StartDay']) &
                            (self.activities['Act_Day'] <= row['Camp_EndDay'])
                            ]['Act_Activity']
                    )

                    # Determine activities subjected to constraints
                    matched_activities = self.activities[
                        self.activities['Act_Channel'].apply(lambda x: str(x).strip() in channels) &
                        (
                                self.activities['Act_TargetProduct1'].apply(lambda x: str(x) in target_products) |
                                self.activities['Act_TargetProduct2'].apply(lambda x: str(x) in target_products)
                        )
                        ]['Act_Activity'].tolist()

                    # Filter matched activities to include only time-constrained ones
                    constrained_activities = set(matched_activities) & activities_time_constrained

                # Store the constrained activities in the appropriate dictionary key
                self.data[key][counter] = constrained_activities
                counter += 1