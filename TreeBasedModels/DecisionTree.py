class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        # for leaf node
        self.value = value

class DecisionTreeClassifier:
    def __init__(self, min_samples_split=2, max_depth=2):
        self.root = None
        # set the stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def gini_index(self, y):
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += (p_cls ** 2)
        return 1 - gini

    def entropy(self, y):
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy

    def information_gain(self, parent, l_child, r_child, mode='entropy'):
        ## parent: the list of ground truth labels for all the points in the region of the parent node
        ## l_child: the list of ground truth labels for all the points in the region l_child
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)

        if mode == 'gini':
            gain = self.gini_index(parent) - ((weight_l * self.gini_index(l_child)) + (weight_r * self.gini_index(r_child)))
        else:
            gain = self.entropy(parent) - ((weight_l * self.entropy(l_child)) + (weight_r * self.entropy(r_child)))

        return gain

    def split(self, dataset, feature_index, threshold):
        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
        return dataset_left, dataset_right   

    def get_best_split(self, dataset, num_samples, num_features):
        # define a dictionary to store the best split
        best_split = {}
        max_info_gain = -float("inf")

        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)

            # loop over all possible split values (thresholds) for the current feature
            for threshold in possible_thresholds:
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)

                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y_parent, y_left_child, y_right_child = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    curr_info_gain = self.information_gain(y_parent, y_left_child, y_right_child, mode='gini')
                    if curr_info_gain > max_info_gain:
                        best_split['feature_index'] = feature_index
                        best_split['threshold'] = threshold
                        best_split['dataset_left'] = dataset_left
                        best_split['dataset_right'] = dataset_right
                        best_split['info_gain'] = curr_info_gain
                        max_info_gain = curr_info_gain

        return best_split

    def calculate_leaf_value(self, Y):
        Y = list(Y)
        return max(Y, key=Y.count)    

    def build_tree(self, dataset, curr_depth=0):
        # recursive function to build the tree
        X, Y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(X)
        # we will split until the stopping conditions are met
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            # getting the best split; it returns a dictionary
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # check if the information gain after the best split is positive
            if best_split['info_gain'] > 0:
                left_subtree = self.build_tree(best_split['dataset_left'], curr_depth + 1)
                right_subtree = self.build_tree(best_split['dataset_right'], curr_depth + 1)
                return Node(feature_index=best_split['feature_index'], threshold=best_split['threshold'],
                            left=left_subtree, right=right_subtree, info_gain=best_split['info_gain'], value=None)  

        # compute the leaf node only if we are out of the stopping condition
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)    

    def fit(self, X, Y):
        Y = Y.reshape(-1, 1)  # Reshape Y to be a column vector
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)

    def make_prediction(self, x, tree):
        if tree.value is not None:
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

    def predict(self, X):
        predictions = [self.make_prediction(x, self.root) for x in X]
        return predictions     

    def print_tree(self, tree=None, indent=" "):
        '''function to print the tree'''

        if not tree:
            tree = self.root      

        if tree.value is not None:
            print(tree.value)
        else:
            print("X_" + str(tree.feature_index), " <= ", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
