from xgboost import XGBClassifier


def train_classifier(features, targets, max_depth):
    clf = XGBClassifier(max_depth=max_depth)
    clf.fit(features, targets)
    return clf