"""Training functions with curriculum learning for Siamese Autoencoder."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path


class CurriculumScheduler:
    """Curriculum learning scheduler for progressive difficulty training.
    
    This scheduler implements a curriculum learning strategy where training
    starts with easier samples (normal-normal pairs) and gradually introduces
    harder samples (normal-anomaly pairs) as training progresses.
    """

    def __init__(
        self,
        total_epochs: int,
        easy_epochs: int = 5,
        medium_epochs: int = 10,
        hard_epochs: int = 15,
    ) -> None:
        """Initialize curriculum scheduler.
        
        Args:
            total_epochs: Total number of training epochs
            easy_epochs: Number of epochs for easy phase (normal-normal pairs only)
            medium_epochs: Number of epochs for medium phase (mix of pairs)
            hard_epochs: Number of epochs for hard phase (all pairs, including hard negatives)
        """
        self.total_epochs = total_epochs
        self.easy_epochs = easy_epochs
        self.medium_epochs = medium_epochs
        self.hard_epochs = hard_epochs
        self.current_phase = "easy"

    def get_phase(self, epoch: int) -> str:
        """Get current curriculum phase based on epoch.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            Phase name: 'easy', 'medium', or 'hard'
        """
        if epoch < self.easy_epochs:
            return "easy"
        elif epoch < self.easy_epochs + self.medium_epochs:
            return "medium"
        else:
            return "hard"

    def get_pair_sampling_ratio(self, epoch: int) -> Dict[str, float]:
        """Get sampling ratios for different pair types in current phase.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            Dictionary with sampling ratios for different pair types
        """
        phase = self.get_phase(epoch)
        
        if phase == "easy":
            # Easy phase: mostly normal-normal pairs
            return {"normal_normal": 0.9, "normal_anomaly": 0.1, "anomaly_anomaly": 0.0}
        elif phase == "medium":
            # Medium phase: balanced mix
            return {"normal_normal": 0.5, "normal_anomaly": 0.4, "anomaly_anomaly": 0.1}
        else:
            # Hard phase: all types, including hard negatives
            return {"normal_normal": 0.3, "normal_anomaly": 0.5, "anomaly_anomaly": 0.2}


def create_siamese_pairs(
    features: np.ndarray,
    labels: np.ndarray,
    num_pairs: int,
    sampling_ratios: Dict[str, float],
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create pairs for Siamese network training.
    
    Args:
        features: Feature matrix
        labels: Label array (0=normal, 1=anomaly)
        num_pairs: Number of pairs to generate
        sampling_ratios: Dictionary with ratios for pair types
        random_state: Random seed
        
    Returns:
        Tuple of (pairs_x1, pairs_x2, pair_labels) where pair_labels
        indicate similarity (0=similar, 1=dissimilar)
    """
    np.random.seed(random_state)
    
    normal_indices = np.where(labels == 0)[0]
    anomaly_indices = np.where(labels == 1)[0]
    
    pairs_x1 = []
    pairs_x2 = []
    pair_labels = []
    
    # Calculate number of pairs for each type
    n_normal_normal = int(num_pairs * sampling_ratios["normal_normal"])
    n_normal_anomaly = int(num_pairs * sampling_ratios["normal_anomaly"])
    n_anomaly_anomaly = int(num_pairs * sampling_ratios["anomaly_anomaly"])
    
    # Normal-Normal pairs (similar, label=0)
    if n_normal_normal > 0 and len(normal_indices) >= 2:
        idx1 = np.random.choice(normal_indices, n_normal_normal)
        idx2 = np.random.choice(normal_indices, n_normal_normal)
        pairs_x1.append(features[idx1])
        pairs_x2.append(features[idx2])
        pair_labels.extend([0] * n_normal_normal)
    
    # Normal-Anomaly pairs (dissimilar, label=1)
    if n_normal_anomaly > 0 and len(normal_indices) > 0 and len(anomaly_indices) > 0:
        idx1 = np.random.choice(normal_indices, n_normal_anomaly)
        idx2 = np.random.choice(anomaly_indices, n_normal_anomaly)
        pairs_x1.append(features[idx1])
        pairs_x2.append(features[idx2])
        pair_labels.extend([1] * n_normal_anomaly)
    
    # Anomaly-Anomaly pairs (similar, label=0)
    if n_anomaly_anomaly > 0 and len(anomaly_indices) >= 2:
        idx1 = np.random.choice(anomaly_indices, n_anomaly_anomaly)
        idx2 = np.random.choice(anomaly_indices, n_anomaly_anomaly)
        pairs_x1.append(features[idx1])
        pairs_x2.append(features[idx2])
        pair_labels.extend([0] * n_anomaly_anomaly)
    
    # Combine and shuffle
    if pairs_x1:
        pairs_x1 = np.vstack(pairs_x1)
        pairs_x2 = np.vstack(pairs_x2)
        pair_labels = np.array(pair_labels)
        
        # Shuffle
        indices = np.random.permutation(len(pair_labels))
        pairs_x1 = pairs_x1[indices]
        pairs_x2 = pairs_x2[indices]
        pair_labels = pair_labels[indices]
    else:
        # Fallback: create random pairs
        indices = np.random.choice(len(features), (num_pairs, 2))
        pairs_x1 = features[indices[:, 0]]
        pairs_x2 = features[indices[:, 1]]
        pair_labels = (labels[indices[:, 0]] != labels[indices[:, 1]]).astype(int)
    
    return pairs_x1, pairs_x2, pair_labels


def train_siamese_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = "cpu",
    curriculum_scheduler: Optional[CurriculumScheduler] = None,
    save_path: Optional[str] = None,
) -> Dict[str, List[float]]:
    """Train Siamese Autoencoder with curriculum learning.
    
    This function implements the training loop with curriculum learning,
    where training difficulty increases progressively through phases.
    
    Args:
        model: Siamese Autoencoder model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cpu' or 'cuda')
        curriculum_scheduler: Optional curriculum learning scheduler
        save_path: Optional path to save best model
        
    Returns:
        Dictionary with training history (train_loss, val_loss per epoch)
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Combined loss: reconstruction + contrastive
    reconstruction_criterion = nn.MSELoss()
    
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    
    # Get training data for pair generation
    train_features = []
    train_labels = []
    for batch_features, batch_labels in train_loader:
        train_features.append(batch_features.numpy())
        train_labels.append(batch_labels.numpy())
    train_features = np.vstack(train_features)
    train_labels = np.hstack(train_labels)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        # Get curriculum phase
        if curriculum_scheduler:
            phase = curriculum_scheduler.get_phase(epoch)
            sampling_ratios = curriculum_scheduler.get_pair_sampling_ratio(epoch)
        else:
            phase = "hard"
            sampling_ratios = {"normal_normal": 0.3, "normal_anomaly": 0.5, "anomaly_anomaly": 0.2}
        
        # Create pairs for this epoch
        pairs_x1, pairs_x2, pair_labels = create_siamese_pairs(
            train_features, train_labels, num_pairs=len(train_features), sampling_ratios=sampling_ratios
        )
        
        # Convert to tensors
        pairs_x1_tensor = torch.FloatTensor(pairs_x1).to(device)
        pairs_x2_tensor = torch.FloatTensor(pairs_x2).to(device)
        pair_labels_tensor = torch.LongTensor(pair_labels).to(device)
        
        # Training step
        optimizer.zero_grad()
        
        # Forward pass
        recon1, recon2, enc1, enc2 = model(pairs_x1_tensor, pairs_x2_tensor)
        
        # Combined loss
        recon_loss = reconstruction_criterion(recon1, pairs_x1_tensor) + \
                     reconstruction_criterion(recon2, pairs_x2_tensor)
        
        from finalproject.models import contrastive_loss
        contrastive_loss_val = contrastive_loss(enc1, enc2, pair_labels_tensor)
        
        loss = recon_loss + contrastive_loss_val
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        train_loss = loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                
                # Create validation pairs
                batch_size = batch_features.size(0)
                indices = torch.randperm(batch_size)
                pairs_x1_val = batch_features
                pairs_x2_val = batch_features[indices]
                pair_labels_val = (batch_labels != batch_labels[indices]).long().to(device)
                
                recon1, recon2, enc1, enc2 = model(pairs_x1_val, pairs_x2_val)
                
                recon_loss_val = reconstruction_criterion(recon1, pairs_x1_val) + \
                                reconstruction_criterion(recon2, pairs_x2_val)
                contrastive_loss_val = contrastive_loss(enc1, enc2, pair_labels_val)
                
                val_loss += (recon_loss_val + contrastive_loss_val).item()
        
        val_loss /= len(val_loader)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss and save_path:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Phase: {phase}, "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return history


def train_autoencoder_baseline(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = "cpu",
    save_path: Optional[str] = None,
) -> Dict[str, List[float]]:
    """Train standard autoencoder baseline.
    
    Args:
        model: Autoencoder model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        save_path: Optional path to save best model
        
    Returns:
        Dictionary with training history
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_features, _ in train_loader:
            batch_features = batch_features.to(device)
            
            optimizer.zero_grad()
            reconstructed = model(batch_features)
            loss = criterion(reconstructed, batch_features)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_features, _ in val_loader:
                batch_features = batch_features.to(device)
                reconstructed = model(batch_features)
                loss = criterion(reconstructed, batch_features)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        if val_loss < best_val_loss and save_path:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return history

