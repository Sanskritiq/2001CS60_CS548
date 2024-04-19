import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import yaml
import matplotlib.pyplot as plt
from fs import fs
import argparse

with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


class FeatureSelection:
    def __init__(self, filename, algo):
        self.filename = filename
        self.algo = algo

    def read_data(self, train_file, test_file):
        self.train_data = pd.read_csv(train_file, sep=",", encoding="utf-8")
        self.test_data = pd.read_csv(test_file, sep=",", encoding="utf-8")

    def split_data(self):
        self.X_train = self.train_data.drop(columns=["label"], axis=1)
        self.y_train = self.train_data["label"]
        self.X_test = self.test_data.drop(columns=["label"], axis=1)
        self.y_test = self.test_data["label"]
        self.X_t, _, self.y_t, _ = train_test_split(
            self.X_train, self.y_train, train_size=0.01, random_state=7
        )

    def perform_feature_selection(self):
        feat = np.asarray(self.X_t)
        label = np.asarray(self.y_t)
        xtrain, xtest, ytrain, ytest = train_test_split(
            feat, label, test_size=0.3, stratify=label
        )
        fold = {"xt": xtrain, "yt": ytrain, "xv": xtest, "yv": ytest}

        opts = {
            "k": config["K"],
            "fold": fold,
            "N": config["N"],
            "T": config["T"],
            "P": config["P"],
        }

        start_time = time.time()
        self.fso = fs(self.algo, feat, label, xtest, ytest, opts)
        end_time = time.time()

        self.selected_features = self.fso["sf"]
        self.execution_time = end_time - start_time

    def plot(self):
        # Plot convergence
        plt.plot(self.fso["c"])
        plt.xlabel("Number of Iterations")
        plt.ylabel("Fitness Value")
        plt.title(self.algo + " Convergence")
        plt.grid(True)
        plt.show()

        # save the plot in images folder
        plt.savefig( self.filename + f"_{self.algo}_convergence.png")

    def write_feature_file(self):
        feature_name = self.filename + f"_{str(self.algo)}_feature.csv"
        with open(feature_name, "w") as file:
            file.write(
                "optimization,execution time of optimzier,no of feature selected,selected feature\n"
            )
            file.write(
                self.algo
                + ","
                + str(self.execution_time)
                + ","
                + str(len(self.selected_features))
                + ',"'
            )
            column_headers = list(self.X_train.columns.values)
            for i in self.selected_features:
                file.write(column_headers[i] + ",")
            file.write('"\n')

    def select_features(self):
        feature_df = pd.read_csv(
            self.filename + f"_{self.algo}_feature.csv", sep=",", encoding="utf-8"
        )
        selected_feature = feature_df.iat[0, 3]
        selected_feature = selected_feature[0:-1]
        return list(selected_feature.split(","))


class DataPreprocessing:
    @staticmethod
    def preprocess_data(X_train, X_test, selected_features):
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]
        return X_train, X_test


# Usage example
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename", type=str, help="Filename of the dataset", default="NF-BOT-IOT"
    )
    parser.add_argument("--algo", type=str, help="Algorithm to use", default="wsa")

    args = parser.parse_args()

    feature_selector = FeatureSelection(args.filename, args.algo)
    feature_selector.read_data(
        "NF-BOT-IOT_train_preprocessed.csv", "NF-BOT-IOT_test_preprocessed.csv"
    )
    feature_selector.split_data()
    feature_selector.perform_feature_selection()
    feature_selector.write_feature_file()
    feature_selector.plot()
    # selected_features = feature_selector.select_features()
    # X_train_selected, X_test_selected = DataPreprocessing.preprocess_data(
    #     feature_selector.X_train, feature_selector.X_test, selected_features
    # )
