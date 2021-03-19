# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


def load_data(url):
    data = pd.read_csv(url)
    print("Data imported successfully")
    plot_data(data)
    return data


def plot_data(data):
    data.plot(x='Hours', y='Scores', style='o')
    plt.title('Hours vs Percentage')
    plt.xlabel('Hours Studied')
    plt.ylabel('Percentage Score')
    plt.show()


def pre_process_data(data):
    attributes = data.iloc[:, :-1].values
    labels = data.iloc[:, 1].values

    return attributes, labels


def plot_regression_line(X, y, regressor):
    # Plotting the regression line
    line = regressor.coef_ * X + regressor.intercept_
    plt.scatter(X, y)
    plt.plot(X, line)
    plt.savefig(f'{output_dir}task1_regression.png')
    # plt.show()


def train_data(X_train, y_train):
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    print("Training complete.")
    return regressor


def predict_student_score(regressor, hours):
    pred_score = regressor.predict(hours)
    print("No of Hours = {}".format(hours))
    print("Predicted Score = {}".format(pred_score))
    return pred_score


def generate_student_study_score_model(X_train, y_train):
    regressor = train_data(X_train, y_train)
    return regressor


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test


def evaluate_model(y_test, y_pred):
    print('Mean Absolute Error:',
          metrics.mean_absolute_error(y_test, y_pred))


def student_study_score_task():
    url = "http://bit.ly/w-data"
    data = load_data(url)
    X, y = pre_process_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    regressor = generate_student_study_score_model(X_train, y_train)
    plot_regression_line(X, y, regressor)
    predict_student_score(regressor, hours=[[9.25]])
    y_pred = predict_student_score(regressor, hours=X_test)
    evaluate_model(y_test, y_pred)


if __name__ == "__main__":
    output_dir = './outputs/'
    student_study_score_task()
