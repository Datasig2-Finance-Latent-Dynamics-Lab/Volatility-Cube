import torch

import pandas as pd

import numpy as np

from typing import cast

from tqdm import tqdm

from torch import nn

from ivsurfacefitting.models.base import IVSurfaceModel

from ivsurfacefitting.data.utils import df_to_tensor, tensor_to_df


class CrossAttnEncodeMLPDecoder(IVSurfaceModel, nn.Module):
    """
    Transformer model for fitting gridless surfaces.

    The idea is that, given a set of observations X = {x_1, x_2, ...}, the decoder will use a
    set of learneable vectors Z = {z_1,..}, such that the first attention matrix is 

        A = softmax(QK^T / sqrt(d))

    Where Q comes from WQ*Z and K from WK*X, then doing V*A whete V = WV*X, ensures
    permutation invariance, since it would permute the rows of both K and V.

    After this we perform more transofrmer layers for the encoding, finally one
    uses a fully connected neural network for the decoder.

    Note that it doesnt guarantee nor try to enforce no arbitrage in any way.
    """

    def __init__(self, latent_dim: int = 16) -> None:
        """
        Initializes the module and defines the architecture.

        Remember that it takes log moneyness and maturity as inputs,
        and outputs iv, so the input dimension is 2 and output is one.

        Args:
            latent_dim (int): Dimension to encode into.
        """

        IVSurfaceModel.__init__(self,name = "CrossAttnEncodeMLPDecoder", learnable=True)
        nn.Module.__init__(self)

        # Embedding of initial observations
        # After this shape is (batch, N, 64)
        self.embedding = nn.Sequential(
            nn.Linear(3, 64), # 3 is input dim plus output dim
            nn.GELU(),
            nn.Linear(64, 64),
        )

        # Create learneable vectors Z
        self.learneable = nn.Parameter(torch.randn(16, 64))

        # Create permutation invariant cross-attention layer
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=8,
            batch_first=True,
        )

        # Make multiple transformer layer
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=8,
            dim_feedforward=128,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=4,
        )

        # Final linear layers to get low dimensional representation.

        self.final_encoding_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 64, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim),
        )

        # Decoding

        self.network = nn.Sequential(
            nn.Linear(latent_dim + 2, 256), # latent dim + input dim
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1), # output dim
        )

    def forward(self, samples: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Encodes the samples, and then evaluates decoder on the given coordinates.

        Args:
            samples (torch.Tensor): samples used for fitting
            coordinates (torch.Tensor): coordinates at which to evaluate the model.
        """
        # Encoding
        samples = self.embedding(samples)

        # Copy latent vectors by batch
        batch_size = samples.shape[0]
        latents = self.learneable.unsqueeze(0)
        latents = latents.expand(batch_size, -1, -1)

        # Perform attention, returns weights too, which are usefull sometimes, not currently needed
        x, _ = self.cross_attention(
            query=latents,
            key=samples,
            value=samples,
        )

        # Now transformer layers
        x = self.transformer(x)

        # FInal layer
        encoding = self.final_encoding_layer(x)

        #Decoding
        _, N, _ = coordinates.shape

        # Turn latent into shape (batch, points, latent_dim) so it can be concatenated with coordinates.
        z = encoding[:, None, :].expand(-1, N, -1)

        x = torch.cat([z, coordinates], dim=-1)

        return self.network(x)

    def learn(self, train_data: pd.DataFrame, **kwargs) -> dict:
        """
        Handles the learning/training.
        
        Args:
            train_data (pd.DataFrame)
        """

        if "epochs" in kwargs:
            epochs = kwargs["epochs"]
        else:
            epochs = 100

        train_tensor = df_to_tensor(
                "id",
                cast(pd.Series, train_data["id"]), # cast only needed so type checker doesnt complain
                ["logmoneyness","maturity","iv"],
                train_data,            
                )

        input_dim = 2
        output_dim = 1

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        criterion = torch.nn.MSELoss()
        n_surfaces, n_points, dimensions = train_tensor.shape

        surface_indeces = np.arange(n_surfaces)

        batch_size = 64

        if dimensions != input_dim + output_dim:
            raise ValueError("Tensor dimension doesnt match.")

        for _ in tqdm(range(epochs)):
            np.random.shuffle(surface_indeces)

            for batch in range(n_surfaces // batch_size):
                p = np.random.randint(5, n_points)

                start_index = batch * batch_size

                batch_indices = surface_indeces[start_index : start_index + batch_size]

                batch_data = train_tensor[batch_indices].to(device)

                point_indices = np.random.choice(np.arange(n_points), size=p, replace=False)

                samples = batch_data[:, point_indices, :]

                coordinates = batch_data[:, :, :input_dim]

                values = batch_data[:, :, input_dim:]

                predictions = self(samples, coordinates)

                loss = criterion(predictions, values)

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

        self.to("cpu")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.eval()

        #TODO: Return training statistics

        return {}

    def fit(self, data: pd.DataFrame, coordinates: pd.DataFrame) -> pd.DataFrame:

         indices = cast(pd.Series, data["id"])

         data_tensor = df_to_tensor(
                "id",
                indices,
                ["logmoneyness","maturity","iv"],
                data,            
                )

         coords_tensor = df_to_tensor(
                "id",
                indices,
                ["logmoneyness","maturity"],
                coordinates,            
                )

         results_tensor = self.forward(data_tensor, coords_tensor)

         results = tensor_to_df(
                 "id",
                 indices,
                 ["iv"],
                 results_tensor,                 
                 )

         final = coordinates.copy()

         final["iv"] = results["iv"]

         return final

    def load(self, path):
        self.load_state_dict(torch.load(path)) 

    def save(self, path):
        torch.save(self.state_dict(), path)


