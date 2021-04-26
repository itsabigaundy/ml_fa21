import utils
from sklearn import tree


class DecisionTree:
    def __init__(self) -> None:
        self.classifier = tree.DecisionTreeClassifier()

    def train(self, features, labels):
        self.classifier.fit(features, labels)


if __name__ == '__main__':
    dataset = utils.get_churn_data(.9)
    model = DecisionTree()