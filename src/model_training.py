import joblib
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class IrisClassifier:
    def __init__(self, experiment_name="Iris Classification"):
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.model = LogisticRegression(random_state=42)
        self.experiment_name = experiment_name
        mlflow.set_experiment(self.experiment_name)

    def load_data(self):
        iris = datasets.load_iris()
        self.X = iris.data
        self.y = iris.target

    def preprocess_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def train_model(self):
        with mlflow.start_run():
            self.model.fit(self.X_train, self.y_train)

            # Log model parameters
            mlflow.log_params(self.model.get_params())

            # Log the model
            mlflow.sklearn.log_model(self.model, "model")

    def evaluate_model(self):
        with mlflow.start_run(nested=True):
            y_pred = self.model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)

            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            print(f"Model accuracy: {accuracy:.2f}")

    def predict(self, features):
        features_scaled = self.scaler.transform(np.array(features).reshape(1, -1))
        with mlflow.start_run(nested=True):
            prediction = self.model.predict(features_scaled)

            # Log the prediction
            mlflow.log_param("input_features", features)
            mlflow.log_metric("prediction", prediction[0])

        return prediction[0]

    def save_model(self, filename='model.joblib'):
        joblib.dump(self.model, filename)
        print(f"Model saved as {filename}")


def main():
    classifier = IrisClassifier()
    classifier.load_data()
    classifier.preprocess_data()
    classifier.train_model()
    classifier.evaluate_model()
    classifier.save_model('model.joblib')  # Save the model after

    # Example prediction
    new_flower = [[5.1, 3.5, 1.4, 0.2]]  # Example features of a new flower
    prediction = classifier.predict(new_flower)
    print(f"Predicted class for the new flower: {prediction}")


if __name__ == "__main__":
    main()
