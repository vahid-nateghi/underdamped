import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    def __init__(self, dim: int, max_length: float=50) -> None:
        super().__init__()
        assert dim % 2 == 0, "dim must be even for positional encoding for sin/cos"

        self.dim = dim
        self.max_length = max_length
        self.max_rank = dim // 2

    def forward(self, x: torch.tensor) -> torch.tensor:
        encodings = [self.positional_encoding(x, rank) for rank in range(1, self.max_rank + 1)]
        return torch.cat(
            encodings,
            axis=1,
        )

    def positional_encoding(self, x: torch.tensor, rank: int) -> torch.tensor:
        sin = torch.sin(x / self.max_length * rank * torch.pi)
        cos = torch.cos(x / self.max_length * rank * torch.pi)
        return torch.stack((cos, sin), axis=1)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, activation=nn.SiLU):
        super(MLP, self).__init__()
        assert num_layers >= 2, "Must have at least two layers (input + output)."

        self.activation = activation()

        # input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # hidden layers
        self.hidden_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers - 2):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        # output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.input_norm(x)
        x = self.activation(x)

        for layer, norm in zip(self.hidden_layers, self.norms):
            x = layer(x)
            x = norm(x)
            x = self.activation(x)

        x = self.output_layer(x)
        return x


class VelocityModel(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size, num_layers, n_betas=1, pos_length=50, beta_length=350, time_length=10):
        super(VelocityModel, self).__init__()

        self.net = MLP(
            input_dim=(n_betas+in_dim)*hidden_size,
            output_dim=out_dim,
            hidden_dim=hidden_size,
            num_layers=num_layers,
            activation=nn.SiLU,
        )

        self.pos_embed = nn.Linear(in_dim*hidden_size, hidden_size)
        self.beta_embed = nn.Linear(n_betas*hidden_size, n_betas*hidden_size)
        self.time_embed = nn.Linear(hidden_size, hidden_size)

        self.pos_encoder = PositionalEncoder(dim=hidden_size, max_length=pos_length)
        self.beta_encoder = PositionalEncoder(dim=hidden_size, max_length=beta_length)
        self.time_encoder = PositionalEncoder(dim=hidden_size, max_length=time_length)

    def forward(self, pos_t, t, beta0, beta1=None):
        if beta1 is not None:
            beta = torch.cat([beta0, beta1], dim=-1)
        else:
            beta = beta0

        beta = self.beta_encoder(beta).flatten(start_dim=1)
        t = self.time_encoder(t).squeeze()

        beta_embed = self.beta_embed(beta)
        time_embed = self.time_embed(t)

        pos_t = self.pos_encoder(pos_t).flatten(start_dim=1)
        pos_embed = self.pos_embed(pos_t)
        
        model_input = torch.cat([pos_embed, time_embed, beta_embed], dim=1)
        return self.net(model_input)
