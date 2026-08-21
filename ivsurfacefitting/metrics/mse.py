import numpy as np
import pandas as pd

from ivsurfacefitting.metrics.base import IVMetric

class RMSE(IVMetric):

    def __init__(self) -> None:
        super().__init__("rmse")

    def __call__(self, real: pd.DataFrame, prediction: pd.DataFrame, indexcol: str, valuecol: str = "iv"):
        """
        Measures mrse of predicted. 
        """

        if not real[indexcol].equals(prediction[indexcol]):
            raise ValueError("Index columns must match.")

        error = real[valuecol].to_numpy() - prediction[valuecol].to_numpy()

        return np.linalg.norm(error)






