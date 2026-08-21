
import pandas as pd

class IVSurfaceModel:
    """
    Abstract data class for models that fit iv surfaces.
    """

    learnable: bool
    name: str

    def __init__(self, name: str, learnable: bool) -> None:
        self.name = name
        self.learnable = learnable

    def fit(self, data: pd.DataFrame, coordinates: pd.DataFrame) -> pd.DataFrame:
        """
        Handles the fitting for the model.

        Note that it is the models fit method responsability to move the dataframes into whatever data structure
        the model uses for fitting.

        Args:
            data (pd.DataFrame): Observed data used to make the fit, has columns [id, "logmoneyness", "maturity", "iv"].
            coordinates (pd.DataFrame): Coordinates to evaluate the function at, has columns [id, "logmoneyness", "maturity"].

        Returns:
            pd.Dataframe: Same dataframe as columns with the extra "iv" column of fitted values.
        """
        return pd.DataFrame()

    def learn(self, train_data: pd.DataFrame, **kwargs) -> dict:
        """
        Learning algorithm.

        Does nothing if not learnable.
        """
        if self.learnable == True:
            print(f"Learning algorithm for {self.name} not implemented.")
        return {}

    def load(self, path):
        """
        Loads the model from a file.

        Does nothing if not learnable.
        """
        if self.learnable == True:
            print(f"Loading algorithm for {self.name} not implemented.")
        pass

    def save(self, path):
        """
        Saves the model to a file.

        Does nothing if not learnable.
        """
        if self.learnable == True:
            print(f"Saving algorithm for {self.name} not implemented.")
        pass
