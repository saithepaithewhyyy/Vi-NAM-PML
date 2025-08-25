import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

class MeanFieldLayer(nn.Module):
    def __init__(self, input_dim, output_dim, init_var=1.0):  # Changed from 0.1 to 1.0
        super(MeanFieldLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Prior parameters p(theta)
        self.w_mu_p = torch.zeros(input_dim, output_dim)
        self.w_log_var_p = torch.zeros(input_dim, output_dim)
        self.b_mu_p = torch.zeros(output_dim)
        self.b_log_var_p = torch.zeros(output_dim)

        # Variational parameters q(theta)
        self.w_mu_q = nn.Parameter(torch.normal(mean=0, std=0.01, size=(input_dim, output_dim)), requires_grad=True)
        self.w_log_var_q = nn.Parameter(torch.ones(input_dim, output_dim) * torch.log(torch.tensor(0.1)), requires_grad=True)
        self.b_mu_q = nn.Parameter(torch.normal(mean=0, std=0.01, size=(output_dim,)), requires_grad=True)
        self.b_log_var_q = nn.Parameter(torch.ones(output_dim) * torch.log(torch.tensor(0.1)), requires_grad=True)
    # Rest of the class remains unchanged

    # the priors do not change so could be stored as attributes, but
    # it feels cleaner to access them in the same way as the posteriors
    def p_w(self):
        """weight prior distribution"""
        return torch.distributions.Normal(self.w_mu_p, (0.5 * self.w_log_var_p).exp())

    def p_b(self):
        """bias prior distribution"""
        return torch.distributions.Normal(self.b_mu_p, (0.5 * self.b_log_var_p).exp())

    def q_w(self):
        """variational weight posterior"""
        return torch.distributions.Normal(self.w_mu_q, (0.5 * self.w_log_var_q).exp())

    def q_b(self):
        """variational bias posterior"""
        return torch.distributions.Normal(self.b_mu_q, (0.5 * self.b_log_var_q).exp())

    def kl(self):
        weight_kl = torch.distributions.kl.kl_divergence(self.q_w(), self.p_w()).sum()
        bias_kl = torch.distributions.kl.kl_divergence(self.q_b(), self.p_b()).sum()
        return weight_kl + bias_kl

    def forward(self, x):
        """Propagates x through this layer by sampling weights from the posterior"""
        assert (len(x.shape) == 3), "x should be shape (num_samples, batch_size, input_dim)."
        assert x.shape[-1] == self.input_dim

        num_samples = x.shape[0]
        # rsample carries out reparameterisation trick for us
        weights = self.q_w().rsample((num_samples,))  # (num_samples, input_dim, output_dim).
        biases = self.q_b().rsample((num_samples,)).unsqueeze(1)  # (num_samples, batch_size, output_dim)
        return x @ weights + biases # (num_samples, batch_size, output_dim).


class MeanFieldBNN(nn.Module):
    """Mean-field variational inference BNN with VI only in the last layer."""

    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        activation=nn.ELU(),
        noise_std=1,
    ):
        super(MeanFieldBNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        self.log_noise_var = torch.log(torch.tensor(noise_std**2))

        self.network = nn.ModuleList()
        for i in range(len(hidden_dims) + 1):
            if i == 0:
                # First layer: deterministic nn.Linear
                self.network.append(nn.Linear(self.input_dim, self.hidden_dims[i]))
                self.network.append(self.activation)
            elif i == len(hidden_dims):
                # Last layer: MeanFieldLayer for VI
                self.network.append(MeanFieldLayer(self.hidden_dims[i - 1], self.output_dim))
            else:
                # Hidden layers: deterministic nn.Linear
                self.network.append(nn.Linear(self.hidden_dims[i - 1], self.hidden_dims[i]))
                self.network.append(self.activation)

    def forward(self, x, num_samples=1):
        """Propagate the inputs through the network using num_samples weights for the last layer.

        Args:
            x (torch.tensor): Inputs to the network, shape (batch_size, input_dim).
            num_samples (int, optional): Number of samples for the last layer. Defaults to 1.
        """
        assert len(x.shape) == 2, "x.shape must be (batch_size, input_dim)."

        # Propagate through deterministic layers
        for layer in self.network[:-1]:  # Exclude the last layer
            if isinstance(layer, nn.Linear):
                x = layer(x)  # Shape: (batch_size, layer_output_dim)
            else:
                x = layer(x)  # Activation, same shape

        # Expand for the last MeanFieldLayer
        x = torch.unsqueeze(x, 0).repeat(num_samples, 1, 1)  # Shape: (num_samples, batch_size, hidden_dims[-1])

        # Apply the last layer (MeanFieldLayer)
        x = self.network[-1](x)  # Shape: (num_samples, batch_size, output_dim)

        assert len(x.shape) == 3, "x.shape must be (num_samples, batch_size, output_dim)"
        assert x.shape[-1] == self.output_dim

        return x

    def ll(self, y_obs, y_pred, num_samples=1):
        """Computes the log likelihood of the outputs of self.forward(x)"""
        l = torch.distributions.normal.Normal(y_pred, torch.sqrt(torch.exp(self.log_noise_var)))

        # Take mean over num_samples dim, sum over batch_size dim
        return l.log_prob(y_obs.unsqueeze(0).repeat(num_samples, 1, 1)).mean(0).sum(0).squeeze()

    def kl(self):
        """Computes the KL divergence for the last layer (MeanFieldLayer)."""
        # Only the last layer is a MeanFieldLayer
        return self.network[-1].kl()

    def loss(self, x, y, num_samples=1):
        """Computes the ELBO and returns its negative"""
        y_pred = self.forward(x, num_samples=num_samples)

        exp_ll = self.ll(y, y_pred, num_samples=num_samples)
        kl = self.kl()

        return kl - exp_ll, exp_ll, kl


class viNAM(nn.Module):
    """Neural Additive Model with feature group support"""
    def __init__(
        self,
        feature_dims,  # List of dimensions for each feature group
        hidden_dims,
        output_dim=1,
        activation=nn.ELU(),
        noise_std=1.0,
    ):
        super().__init__()
        self.feature_dims = feature_dims
        self.total_input_dim = sum(feature_dims)
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        self.log_noise_var = nn.Parameter(torch.log(torch.tensor(noise_std**2)))

        # Create BNNs for each feature group with respective input dimensions
        self.feature_networks = nn.ModuleList([
            MeanFieldBNN(
                input_dim=dim,
                hidden_dims=hidden_dims,
                output_dim=output_dim,
                activation=activation,
                noise_std=noise_std
            ) for dim in feature_dims
        ])

        # Track feature group boundaries
        self.feature_boundaries = torch.cumsum(
            torch.tensor([0] + feature_dims), dim=0
        ).tolist()

    def forward(self, x, num_samples=1):
        """Process feature groups through their respective networks"""
        assert len(x.shape) == 2, "x.shape must be (batch_size, total_input_dim)"
        assert x.shape[1] == self.total_input_dim, f"Input dim mismatch: {x.shape[1]} vs {self.total_input_dim}"

        outputs = []
        for i, (start, end) in enumerate(zip(
            self.feature_boundaries[:-1],
            self.feature_boundaries[1:]
        )):
            # Extract feature group slice
            x_group = x[:, start:end]
            # Process through corresponding BNN
            outputs.append(self.feature_networks[i](x_group, num_samples))

        # Sum all feature contributions
        return torch.stack(outputs).sum(dim=0)

    def ll(self, y_obs, y_pred, num_samples=1):
        """Log likelihood with heteroskedastic noise support"""
        noise_var = torch.exp(self.log_noise_var)
        return torch.distributions.Normal(y_pred, noise_var.sqrt()).log_prob(y_obs.unsqueeze(0)).mean(0).sum()

    def kl(self):
        """Aggregate KL across all feature networks"""
        return sum(net.kl() for net in self.feature_networks)

    def loss(self, x, y, num_samples=1):
        """ELBO computation with feature group support"""
        y_pred = self.forward(x, num_samples)
        exp_ll = self.ll(y, y_pred, num_samples)
        kl = self.kl()
        return kl - exp_ll, exp_ll, kl
class CustomMixDataset(Dataset):
    def __init__(self, X, y):
        """
        Modified for viNAM compatibility with ordinal encoding and feature preservation
        Args:
            X (pd.DataFrame): Raw features including datetime
            y (pd.Series): Target values (bike count)
        """
        self.raw_X = X.copy()

        bool_cols = X.select_dtypes(include=['bool']).columns.tolist()
        numerical = X.select_dtypes(include=['number']).columns.tolist()
        categorical = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Convert boolean columns to categorical strings
        X = X.copy()
        # bool_cols = ['holiday', 'workingday']
        for col in bool_cols:
            if col in X.columns:
                X[col] = X[col].astype(str)

        # Define preprocessing
        # numerical = ['year', 'month', 'hour', 'temp', 'feel_temp', 'humidity', 'windspeed', 'weekday']
        # categorical = ['season', 'holiday', 'workingday', 'weather']

        # Ordinal encoding for categorical features (preserves single dimension)
        if categorical:
          self.ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
          X_cat = pd.DataFrame(self.ordinal_encoder.fit_transform(X[categorical]),columns=categorical,index=X.index)

        # Standard scaling for numerical features
        self.scaler = StandardScaler()
        X_num = pd.DataFrame(self.scaler.fit_transform(X[numerical]),
                            columns=numerical,
                            index=X.index)

        # Combine processed features
        if categorical:
          self.X_processed = pd.concat([X_num, X_cat], axis=1)
          self.X = torch.tensor(self.X_processed.values, dtype=torch.float32)
        else:
          self.X = torch.tensor(X_num.values, dtype=torch.float32)


        self.y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)
        self.w = torch.ones_like(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.w[idx]

    def get_feature_names(self):
        """Get original feature names in processing order"""
        return self.X_processed.columns.tolist()

    def get_feature_mapping(self):
        """Get ordinal encoding categories for interpretation"""
        return {col: self.ordinal_encoder.categories_[i]
                for i, col in enumerate(self.ordinal_encoder.feature_names_in_)}

# Modified training pipeline for bike sharing data
def load_dataset(name, version = None, test_size=0.1):
    """Load data with temporal validation split"""
    if version is not None:
        ver = fetch_openml(name=name, version=version, as_frame=True)
    else:
        ver = fetch_openml(name=name, as_frame=True)
    df = ver.data
    y = ver.target
    # Time-based split (critical for temporal data)
    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = df.iloc[:split_idx], df.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test

dataset_list = ["yacht_hydrodynamics", "concrete_compressive_strength", "kin8nm","Bike_Sharing_Demand"]
avg_nll = {}
for i,name in enumerate(dataset_list):
  elbo_list = []
  print(name)
  X_train, X_test, y_train, y_test = load_dataset(name = name)
  train_dataset = CustomMixDataset(X_train, y_train)
  test_dataset = CustomMixDataset(X_test, y_test)
  # print(X_test)
  train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
  test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

  # Model initialization with dynamic input dimension and tuned noise_std
  std_y = train_dataset.y.std().item()
  # print(f"Target standard deviation: {std_y}")


  feature_dims = [1] * train_dataset.X.shape[1]
  hidden_dims = [64,32]
  model = viNAM(feature_dims=feature_dims, hidden_dims=hidden_dims, noise_std=1)

  # Optimizer with gradient clipping and smaller initial learning rate
  optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-4)  # Reduced lr for better convergence
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
  num_samples = 500  # Increased MC samples for better ELBO estimation
  n_epochs = 20  # Increased from 50
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
  anneal_epochs = 20
  for epoch in range(n_epochs):
      model.train()
      total_loss = 0.0
      total_ll = 0.0
      total_kl = 0.0
      kl_anneal = min(1.0, (epoch + 1) / anneal_epochs) if epoch < anneal_epochs else 1.0

      for x, y, _ in train_dataloader:
          optimizer.zero_grad()
          loss, exp_ll, kl = model.loss(x, y, num_samples=num_samples)
          loss = kl * kl_anneal - exp_ll  # Apply annealing explicitly
          loss.backward()
          nn.utils.clip_grad_norm_(model.parameters(), 1.0)
          optimizer.step()
          total_loss += loss.item()
          total_ll += exp_ll.item()
          total_kl += kl.item()

      # Validation phase
      model.eval()
      val_rmse = 0.0
      val_nll = 0.0
      total_val_samples = 0
      with torch.no_grad():
          for x, y, _ in test_dataloader:
              batch_size = x.shape[0]
              total_val_samples += batch_size
              preds = model(x, num_samples=100).mean(0)
              val_rmse += torch.sqrt(nn.functional.mse_loss(preds, y)).item() * batch_size
              y_pred = model.forward(x, num_samples=100)
              log_likelihood = model.ll(y, y_pred, num_samples=100)
              val_nll += -log_likelihood.item()  # Corrected accumulation

      val_rmse = val_rmse / total_val_samples
      val_nll = val_nll / total_val_samples

      # Schedule based on val_nll
      scheduler.step(val_nll)

      elbo_list.append(total_loss/len(train_dataloader))

      if epoch % 2 == 0:
          print(f"\nEpoch {epoch+1}/{n_epochs}")
          print(f"Train Loss (ELBO): {total_loss/len(train_dataloader):.2f}")
          print(f"Log Likelihood: {total_ll/len(train_dataloader):.2f}")
          print(f"KL Divergence: {total_kl/len(train_dataloader):.2f}")
          print(f"Val RMSE: {val_rmse:.2f}")
          print(f"Val NLL: {val_nll:.2f}")
          print(f"KL Anneal Factor: {kl_anneal:.2f}")
          print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")

  # Final evaluation (corrected NLL computation)
  model.eval()
  total_nll = 0.0
  total_samples = 0
  with torch.no_grad():
      for x, y, _ in test_dataloader:
          batch_size = x.shape[0]
          total_samples += batch_size
          y_pred = model.forward(x, num_samples=500)
          log_likelihood = model.ll(y, y_pred, num_samples=500)
          total_nll += -log_likelihood.item()  # Sum log-likelihood over batch

      avg_nll[name] = total_nll / total_samples
      print(f"\nFinal Test Negative Log-Likelihood (NLL): {avg_nll[name]:.2f}")

print(avg_nll)
