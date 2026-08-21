from numpy import float32
import torch
import pandas as pd

def df_to_tensor(indexcol: str, indeces: pd.Series, valuescols: list[str], df: pd.DataFrame) -> torch.Tensor:
    """
    Converts a data frame with columns valuescols into a tensor.

    Note that for each unique value on the indexing coumn, there must be the same amount of rows, so that
    it can be converted to a tensor.

    Args:
        indexcol (str): Name of indexing column.
        indeces (pd.Series): Indeces to be converted to tensor.
        valuescols (list[str]): Columns to transform.
        df (pd.DataFrame): Dataframe to convert.
        
    Returns:
        torch.Tensor
    """
    for col in valuescols:
        if col not in df.columns:
            raise ValueError(f"Column {col} not in the dataframe.")

    counts = df[indexcol].value_counts()
    if counts.nunique() != 1:
        raise ValueError("Not all indices have the same number of rows.")

    n_indices = len(counts)

    filtered_df = df[df[indexcol].isin(indeces)]

    tensor = torch.tensor(filtered_df.loc[:,valuescols].to_numpy(),dtype=torch.float32).reshape(n_indices, -1, len(valuescols))

    return tensor

def tensor_to_df(indexcol: str, indeces: pd.Series, valuescols: list[str], tensor: torch.Tensor) -> pd.DataFrame:
    """
    Converts tensor into a dataframe with given columns and indices.

    Tensor must have shape (number of indices, rows per index, len of valuescols).

    Args:
        indexcol (str): Name of indexing column.
        indeces (pd.Series): Indeces to be converted to tensor.
        valuescols (list[str]): Columns to transform.
        tensor (torch.Tensor): Tensor to convert.
        
    Returns:
        pd.DataFrame
    """
    indeces = indeces.drop_duplicates()
    n_indices, n_rows, n_values = tensor.shape

    if n_indices != len(indeces):
        raise ValueError(
            f"Tensor has {n_indices} indices, but {len(indeces)} indices were provided."
        )

    if n_values != len(valuescols):
        raise ValueError(
            f"Tensor has {n_values} value columns, but {len(valuescols)} were provided."
        )

    values = tensor.detach().cpu().numpy().reshape(-1, n_values)

    index_values = indeces.to_numpy().repeat(n_rows)

    df = pd.DataFrame(values, columns=valuescols)
    df.insert(0, indexcol, index_values)

    return df


