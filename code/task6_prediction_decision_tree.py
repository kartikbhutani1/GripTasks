import sklearn.datasets as datasets
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    iris = datasets.load_iris()
    return iris


def create_decision_tree_classifier(X, y):
    dtree = DecisionTreeClassifier()
    dtree.fit(X, y)

    print('Decision Tree Classifer Created')
    return dtree


def visualize_decision_tree(dtree, iris):
    fig = plt.figure(figsize=(30, 30))
    _ = tree.plot_tree(dtree,
                       feature_names=iris.feature_names,
                       class_names=iris.target_names,
                       filled=True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}task6_decision_tree.png')


def process_data(iris):
    iris_features = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_labels = iris.target
    print(iris_features.head())
    return iris_features, iris_labels


def predict_new_data(dtree, X_test):
    y_pred = dtree.predict(X_test)
    return y_pred


def calculate_prediction_accuracy(y_test, y_pred):
    return accuracy_score(y_test, y_pred)


def prediction_decision_trees():
    iris = load_data()
    X, y = process_data(iris)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=1)
    dtree = create_decision_tree_classifier(X_train, y_train)
    visualize_decision_tree(dtree, iris)

    # Prediction
    y_pred = predict_new_data(dtree, X_test)

    # Accuracy
    acc = calculate_prediction_accuracy(y_test, y_pred)
    print(f'Accuracy of prediction using decision trees : {round(acc, 2)}')


if __name__ == "__main__":
    output_dir = '../outputs/'
    prediction_decision_trees()
