"""Neural network models for intrusion detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    """Standard autoencoder for anomaly detection baseline.
    
    This is a simple autoencoder that learns to reconstruct normal traffic
    patterns. Anomalies are detected based on high reconstruction error.
    """

    def __init__(
        self,
        input_dim: int,
        encoding_dim: int = 32,
        hidden_dims: list = None,
    ) -> None:
        """Initialize autoencoder.
        
        Args:
            input_dim: Number of input features
            encoding_dim: Dimension of the latent encoding
            hidden_dims: List of hidden layer dimensions for encoder/decoder
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 32]
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        encoder_layers.append(nn.Linear(prev_dim, encoding_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = encoding_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Reconstructed tensor of shape (batch_size, input_dim)
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation.
        
        Args:
            x: Input tensor
            
        Returns:
            Latent encoding
        """
        return self.encoder(x)


class SiameseAutoencoder(nn.Module):
    """Siamese Autoencoder for metric learning-based anomaly detection.
    
    This architecture combines the reconstruction capability of autoencoders
    with the discriminative power of Siamese networks. It learns a distance
    function between normal and anomalous states by processing pairs of samples
    through shared encoder-decoder networks.
    
    The model extends PyTorch's nn.Module as required by the project guidelines.
    """

    def __init__(
        self,
        input_dim: int,
        encoding_dim: int = 32,
        hidden_dims: list = None,
    ) -> None:
        """Initialize Siamese Autoencoder.
        
        Args:
            input_dim: Number of input features
            encoding_dim: Dimension of the latent encoding space
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 32]
        
        # Shared encoder (used by both branches of Siamese network)
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        encoder_layers.append(nn.Linear(prev_dim, encoding_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Shared decoder
        decoder_layers = []
        prev_dim = encoding_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor = None
    ) -> tuple[torch.Tensor, ...]:
        """Forward pass through Siamese network.
        
        Args:
            x1: First input tensor of shape (batch_size, input_dim)
            x2: Optional second input tensor for pair processing.
                If None, only processes x1.
            
        Returns:
            If x2 is None: (reconstruction, encoding)
            If x2 is provided: (reconstruction1, reconstruction2, encoding1, encoding2)
        """
        # Encode first input
        encoding1 = self.encoder(x1)
        reconstruction1 = self.decoder(encoding1)
        
        if x2 is not None:
            # Encode second input (Siamese pair)
            encoding2 = self.encoder(x2)
            reconstruction2 = self.decoder(encoding2)
            return reconstruction1, reconstruction2, encoding1, encoding2
        
        return reconstruction1, encoding1

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation.
        
        Args:
            x: Input tensor
            
        Returns:
            Latent encoding
        """
        return self.encoder(x)

    def compute_distance(
        self, encoding1: torch.Tensor, encoding2: torch.Tensor
    ) -> torch.Tensor:
        """Compute Euclidean distance between encodings.
        
        Args:
            encoding1: First encoding tensor
            encoding2: Second encoding tensor
            
        Returns:
            Euclidean distance between encodings
        """
        return torch.norm(encoding1 - encoding2, dim=1)


def contrastive_loss(
    encoding1: torch.Tensor,
    encoding2: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Contrastive loss for Siamese network training.
    
    This loss function encourages similar samples (label=0) to have
    encodings close together, and dissimilar samples (label=1) to have
    encodings far apart (beyond a margin).
    
    Args:
        encoding1: Encodings from first branch of Siamese network
        encoding2: Encodings from second branch of Siamese network
        labels: Similarity labels (0 for similar, 1 for dissimilar)
        margin: Margin for dissimilar pairs
        
    Returns:
        Contrastive loss value
    """
    # Compute Euclidean distance between encodings
    distance = torch.norm(encoding1 - encoding2, dim=1)
    
    # Similar pairs (label=0): minimize distance
    similar_loss = (1 - labels) * torch.pow(distance, 2)
    
    # Dissimilar pairs (label=1): maximize distance up to margin
    dissimilar_loss = labels * torch.pow(
        torch.clamp(margin - distance, min=0.0), 2
    )
    
    # Total loss
    loss = torch.mean(similar_loss + dissimilar_loss)
    
    return loss

