# Imports
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib



# ----------------- //// Train model/ model Functions //// -----------------



# using datasets
def load_dataset():
    x, y = fetch_openml(name="mnist_784", version=1, return_X_y=True, as_frame=False, parser="liac-arff")
    x = x.astype("float32") / 255
    return x, y


# split dataset
def split_dataset():
    x , y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y) 
    return X_train, X_test, y_train, y_test


# create model 
def create_model():
    model = MLPClassifier(hidden_layer_sizes=(64, 128), activation="relu", max_iter=20, random_state=42, verbose=True)
    return model


# train model
def train_model(model, x_train, y_train):
    trained_model = model.fit(x_train, y_train)
    return trained_model


# test and calculate accuracy
def evaluate_model(model, x_test, y_test):
    tested_model = model.predict(x_test)
    tested_model_accuracy = accuracy_score(y_test, tested_model)
    print(tested_model_accuracy)
    return tested_model_accuracy


# wrong answers analyze
def analyze_answers(tested_model, y_test):
    analyzrd_model = confusion_matrix(y_test, tested_model)
    sns.heatmap(data=analyzrd_model ,annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    return analyzrd_model


# save trained model
def save_model(model):
    joblib.dump(model, "digit_model.pkl")



# ----------------- //// Train //// -----------------

if __name__ == "__main__":

    # x, y for train and testing model
    x_train, x_test, y_train, y_test = split_dataset()

    # training
    model = create_model()
    trained_model = train_model(model, x_train, y_train)
    save_model(trained_model)

    evaluated_model = evaluate_model(trained_model, x_test, y_test)

    tested_model = trained_model.predict(x_test)

    analyzed_answers = analyze_answers(tested_model, y_test)
