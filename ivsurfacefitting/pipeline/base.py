import torch

from pathlib import Path

import pandas as pd

from ivsurfacefitting.models.base import IVSurfaceModel
from ivsurfacefitting.metrics.base import IVMetric


class IVSurfacePipeline:
    """
    Pipeline for implied volatility surface fitting.

    Note that everything is very senseible to the names of the datasets and the models.

    Training results are made by pairs (train data, model).

    Testing results are mad eby triplets (train data, test data, model).

    Attributes:
    TODO
    """

    def __init__(
            self,
            train_locations: list[str],
            test_locations: list[str],
            models: list[IVSurfaceModel],
            metrics: list[IVMetric],
            ):

        self.train_locations = train_locations
        self.test_locations = test_locations
        self.models = models
        self.metrics = metrics

        self.path = Path(__file__).resolve().parent.parent

        self.results_path = self.path / "results"
        self.results_path.mkdir(parents=True, exist_ok=True)

        self.train_results_path = self.results_path / "train"
        self.train_results_path.mkdir(parents=True, exist_ok=True)

        self.test_results_path = self.results_path / "test"
        self.test_results_path.mkdir(parents=True, exist_ok=True)

        # Create training directories
        for train_data in self.train_locations:
            for m in models:
                p = self.train_results_path / f"{train_data}_{m.name}"
                p.mkdir(parents=True, exist_ok=True)

        # Create testing directories
        for train_data in self.train_locations:
            for test_data in self.test_locations:
                for m in models:
                    p = self.test_results_path / f"{train_data}_{test_data}_{m.name}"
                    p.mkdir(parents=True, exist_ok=True)

        # Load datasets.

        self.train_datasets = []
        for train_loc in train_locations:
            self.train_datasets.append(pd.read_csv(f"ivsurfacefitting/datasets/{train_loc + ".csv"}"))


        self.test_datasets = []
        for test_loc in test_locations:
            self.test_datasets.append(pd.read_csv(f"ivsurfacefitting/datasets/{test_loc + ".csv"}"))


    def run(self, forcetrain = False):
        """
        Runs the pipeline.
        """
        results = pd.DataFrame(columns = ["train_dataset","test_dataset","model","metric","value"])


        for d,dataset in enumerate(self.train_datasets):

            dname = self.train_locations[d]

            for model in self.models:

                path = self.train_results_path / f"{dname}_{model.name}"
                model_path = path / "trained_model.pt"

                if model.learnable:
                    if (not model_path.exists()) or forcetrain:
                        
                        print(f"Learning {model.name} for dataset {dname}.")
                        train_stats = model.learn(dataset) # TODO: Save training statistics.
                        print("Learnt.")

                        print(f"Saving {model.name} for dataset {dname} to {model_path}.")
                        model.save(model_path)
                        print("Saved.")

        # Get the predictions and evaluate the metrics.
        for model in self.models:

            for t,tdataset in enumerate(self.test_datasets):

                tname = self.test_locations[t]

                if not model.learnable:

                    prediction = model.fit(tdataset, tdataset[["id","logmoneyness","maturity"]])
                    # May need to add another fit on a gird for easier computing of no arbitrage.

                    prediction.to_csv(self.test_results_path / f"notrain_{tname}_{model.name}" / "predictions.csv",index=False)

                    for metric in self.metrics:

                        val = metric(prediction, tdataset, "id")

                        results.loc[len(results)] = ["not trainable", "", model.name , metric.name, val]

                if model.learnable:

                    for train_loc in self.train_locations:

                        path = self.train_results_path / f"{train_loc}_{model.name}"
                        model_path = path / "trained_model.pt"

                        print(f"Loading {model.name} for dataset {train_loc}.")
                        model.load(model_path)
                        print("Loaded.")

                        prediction = prediction = model.fit(tdataset, tdataset[["id","logmoneyness","maturity"]])

                        prediction.to_csv(self.test_results_path / f"{train_loc}_{tname}_{model.name}" / "predictions.csv",index=False)

                        for metric in self.metrics:

                            val = metric(prediction, tdataset, "id")

                            results.loc[len(results)] = [train_loc, tname, model.name , metric.name, val]

        print(results)
        results.to_csv(self.results_path / "final_statistics.csv", index = False)




