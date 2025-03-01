from functools import partial
import pdb
import concurrent
from multiprocessing import Pool

from graphviz import Digraph


class ExtendedBooster:
    def __init__(self, booster, params=None, cache=()):
        self.booster = booster
        self.node_ranges = {}  # Dictionary to store ranges for each node

    # Define a function to get the leaf value for a specific node
    def get_leaf_value(self, df, tree, node):
        """
        Get the leaf value of a node if it is a leaf.
        Returns None if it's not a leaf.
        """
        node_info = df[(df["Tree"] == tree) & (df["ID"] == node)]
        if node_info.iloc[0]["Feature"] == "Leaf":
            return node_info.iloc[0]["Gain"]
        return None

    # Define a function to get the yes/no children of a node
    def get_yes_no_nodes(self, df, tree, node):
        """
        Get the yes and no child nodes for a given node in a given tree.
        """
        node_info = df[(df["Tree"] == tree) & (df["ID"] == node)]
        return node_info.iloc[0]["Yes"], node_info.iloc[0]["No"]

    # Function to calculate the range of possible output changes by changing the outcome at a node
    def calculate_output_change(self, df, tree, node):
        """
        Calculate the change in output by altering the decision at a specific node.
        """
        if node in self.node_ranges:
            return self.node_ranges[node]
        yes_node, no_node = self.get_yes_no_nodes(df, tree, node)

        # Get leaf values if the sample goes down the "yes" or "no" path
        leaf_value = self.get_leaf_value(df, tree, node)

        # If both yes and no are leaves, calculate the difference
        if leaf_value is not None:
            lval = leaf_value
            rval = leaf_value
        else:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Submit tasks for all children
                futures = {
                    executor.submit(
                        self.calculate_output_change, (df), (tree), (child)
                    ): child
                    for child in [yes_node, no_node]
                }

                # Wait for all futures to complete and set labels
                future = list(concurrent.futures.as_completed(futures))

                ry, rn = future[0].result(), future[1].result()
            # ry= self.calculate_output_change(df, tree, yes_node)
            # rn= self.calculate_output_change(df, tree, no_node)
            rval = max(ry[1], rn[1])
            lval = min(ry[0], rn[0])
        self.node_ranges[node] = (lval, rval)
        return lval, rval

    def compute_node_ranges(self):
        """

        Compute the range of output changes for each internal node across all trees.
        This method populates the `node_ranges` dictionary with the range of output change.
        """
        # Convert the trees to a pandas DataFrame
        tree_df = self.booster.trees_to_dataframe()

        # Iterate over all trees in the model
        for tree in tree_df["Tree"].unique():
            # Get all internal nodes (non-leaf nodes)
            self.calculate_output_change(tree_df, tree, f"{tree}-0")

    def custom_to_graphviz(self, num_trees=0):
        tree_df = self.booster.trees_to_dataframe()
        tree_df = tree_df[tree_df["Tree"] == num_trees]
        graph = Digraph()
        node_ranges = self.node_ranges
        for _, row in tree_df.iterrows():
            node_id = row["ID"]
            if row["Feature"] == "Leaf":
                label = f"Leaf\nValue: {row['Gain']:.2f}"
            else:
                label = f"Node {node_id}\n{row['Feature']} <= {row['Split']:.2f}"

            # Add node with range information
            if node_id in node_ranges:
                values = node_ranges[node_id]
                label += f"\n<{values[0]:.2f}, {values[1]:.2f}>"
            graph.node(str(f"{node_id}"), label=label)
        for _, row in tree_df.iterrows():
            node_id = row["ID"]
            # Add edges
            if row["Feature"] != "Leaf":
                graph.edge(str(node_id), str(row["Yes"]), label="Yes")
                graph.edge(str(node_id), str(row["No"]), label="No")
        return graph

    def get_range(self, node_id):
        return self.node_ranges.get(node_id)
