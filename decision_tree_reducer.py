from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

def get_k_depth_values(features, classes, min_depth, max_depth, random_state):
    k_depth_values = []
    previous_k = -1
    for depth in range(min_depth, max_depth + 1):
        print('.', end='', flush=True)
        tree = DecisionTreeClassifier(max_depth=depth, random_state=random_state)
        tree.fit(features, classes)
        if tree.get_depth() < depth:
            # we have reached the maximum depth for these set of features/classes
            break
        predicted_classes = tree.predict(features)
        score = f1_score(classes, predicted_classes, average='macro')
        k = len(get_features(tree))
        if k < previous_k:
            for i in range(0, len(k_depth_values)):
                if k < k_depth_values[i][0]:
                    k_depth_values.insert(i, (k, depth, score))
                    break
                elif k == k_depth_values[i][0]:
                    if score > k_depth_values[i][2]:
                        k_depth_values[i] = (k, depth, score)
                    break
        elif k == previous_k:
            # this may be better than the previous, but still has the same k, otherwise, keep previous
            if k_depth_values[len(k_depth_values) - 1][2] < score:
                k_depth_values[len(k_depth_values) - 1] = (k, depth, score)
        else:
            k_depth_values.append((k, depth, score))
            previous_k = k
    return k_depth_values

def get_features(tree):
    features = {}
    for i in range(0, tree.tree_.node_count):
        feature = tree.tree_.feature[i]
        features[feature] = True
    return list(features.keys())

class DecisionTreeDimReducer:
    
    def __init__(self, depth, random_state):
        self.tree = DecisionTreeClassifier(max_depth=depth, random_state=random_state)
        self.features = []
        self.k = -1
        self.score = 0
    
    def fit(self, features, classes):
        self.tree.fit(features, classes)
        self.features = get_features(self.tree)
        self.k = len(self.features)
        predicted_classes = self.tree.predict(features)
        self.score = f1_score(classes, predicted_classes, average='macro')

    def transform(self, features):
        return features[:,self.features]
