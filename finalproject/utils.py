"""Utility functions for evaluation, visualization, and robustness testing."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional, List
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from captum.attr import IntegratedGradients, GradientShap, Saliency
import pandas as pd


def compute_anomaly_scores_autoencoder(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute anomaly scores using reconstruction error.
    
    Args:
        model: Trained autoencoder model
        data_loader: Data loader
        device: Device to run on
        
    Returns:
        Tuple of (anomaly_scores, true_labels)
    """
    model.eval()
    scores = []
    labels = []
    
    with torch.no_grad():
        for batch_features, batch_labels in data_loader:
            batch_features = batch_features.to(device)
            reconstructed = model(batch_features)
            
            # Reconstruction error as anomaly score
            reconstruction_error = torch.mean(
                (batch_features - reconstructed) ** 2, dim=1
            ).cpu().numpy()
            
            scores.extend(reconstruction_error)
            labels.extend(batch_labels.numpy())
    
    return np.array(scores), np.array(labels)


def compute_anomaly_scores_siamese(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str = "cpu",
    reference_samples: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute anomaly scores using Siamese distance metric.
    
    Args:
        model: Trained Siamese Autoencoder model
        data_loader: Data loader
        device: Device to run on
        reference_samples: Reference normal samples for distance computation.
            If None, uses mean encoding of normal samples.
        
    Returns:
        Tuple of (anomaly_scores, true_labels)
    """
    model.eval()
    scores = []
    labels = []
    
    # Get reference encoding if not provided
    if reference_samples is None:
        # Collect normal samples from data
        normal_samples = []
        for batch_features, batch_labels in data_loader:
            normal_mask = batch_labels == 0
            if normal_mask.any():
                normal_samples.append(batch_features[normal_mask])
        if normal_samples:
            reference_samples = torch.cat(normal_samples, dim=0)[:100].to(device)
        else:
            # Fallback: use first batch
            for batch_features, _ in data_loader:
                reference_samples = batch_features[:10].to(device)
                break
    
    reference_encoding = model.encode(reference_samples)
    reference_mean = reference_encoding.mean(dim=0, keepdim=True)
    
    with torch.no_grad():
        for batch_features, batch_labels in data_loader:
            batch_features = batch_features.to(device)
            batch_encoding = model.encode(batch_features)
            
            # Distance to reference normal encoding
            distances = torch.norm(batch_encoding - reference_mean, dim=1).cpu().numpy()
            
            scores.extend(distances)
            labels.extend(batch_labels.numpy())
    
    return np.array(scores), np.array(labels)


def evaluate_model(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """Evaluate model performance with various metrics.
    
    Args:
        scores: Anomaly scores
        labels: True labels (0=normal, 1=anomaly)
        threshold: Classification threshold. If None, uses optimal threshold.
        
    Returns:
        Dictionary with evaluation metrics
    """
    if threshold is None:
        # Find optimal threshold using ROC curve
        fpr, tpr, thresholds = roc_curve(labels, scores)
        optimal_idx = np.argmax(tpr - fpr)
        threshold = thresholds[optimal_idx]
    
    predictions = (scores >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    # ROC AUC
    try:
        roc_auc = roc_auc_score(labels, scores)
    except ValueError:
        roc_auc = 0.0
    
    # PR AUC
    try:
        pr_auc = average_precision_score(labels, scores)
    except ValueError:
        pr_auc = 0.0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": threshold,
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None) -> None:
    """Plot training history curves.
    
    Args:
        history: Dictionary with training history
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    ax.plot(epochs, history["val_loss"], label="Validation Loss", marker="s")
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training History", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Plot confusion matrix.
    
    Args:
        labels: True labels
        predictions: Predicted labels
        save_path: Optional path to save figure
    """
    cm = confusion_matrix(labels, predictions)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
    )
    
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_roc_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Plot ROC curve.
    
    Args:
        labels: True labels
        scores: Anomaly scores
        save_path: Optional path to save figure
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = roc_auc_score(labels, scores)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier", linewidth=1)
    
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def detect_data_drift(
    reference_data: np.ndarray,
    new_data: np.ndarray,
    threshold: float = 0.05,
) -> Dict[str, float]:
    """Detect data drift using statistical tests.
    
    This function implements drift detection by comparing feature distributions
    between reference (training) data and new (inference) data.
    
    Args:
        reference_data: Reference dataset (training data)
        new_data: New dataset to test for drift
        threshold: Threshold for drift detection (p-value)
        
    Returns:
        Dictionary with drift detection results
    """
    from scipy import stats
    
    drift_results = {
        "drift_detected": False,
        "features_with_drift": [],
        "max_ks_statistic": 0.0,
        "mean_ks_statistic": 0.0,
    }
    
    ks_statistics = []
    
    for feature_idx in range(reference_data.shape[1]):
        ref_feature = reference_data[:, feature_idx]
        new_feature = new_data[:, feature_idx]
        
        # Kolmogorov-Smirnov test
        ks_stat, p_value = stats.ks_2samp(ref_feature, new_feature)
        ks_statistics.append(ks_stat)
        
        if p_value < threshold:
            drift_results["features_with_drift"].append(feature_idx)
            drift_results["drift_detected"] = True
    
    if ks_statistics:
        drift_results["max_ks_statistic"] = max(ks_statistics)
        drift_results["mean_ks_statistic"] = np.mean(ks_statistics)
    
    return drift_results


def create_adversarial_examples(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    attack_type: str = "fgsm",
    eps: float = 0.1,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate adversarial examples using ART.
    
    Args:
        model: Trained model
        X: Input features
        y: True labels
        attack_type: Type of attack ('fgsm' or 'pgd')
        eps: Attack strength (epsilon)
        device: Device to run on
        
    Returns:
        Tuple of (adversarial_examples, predictions_on_adversarial)
    """
    # Ensure input is float32 to match model dtype
    X = X.astype(np.float32)
    
    # Create a wrapper model that outputs anomaly scores for classification
    class AnomalyClassifier(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            
        def forward(self, x):
            x = x.float()  # Ensure float32
            # For autoencoder: compute reconstruction error as anomaly score
            # Check class name to avoid circular imports
            model_class_name = self.base_model.__class__.__name__
            
            if 'Siamese' in model_class_name:
                # For Siamese: compute distance to reference
                encoding = self.base_model.encode(x)
                # Use a dummy reference (mean of encodings)
                if not hasattr(self, 'reference'):
                    with torch.no_grad():
                        self.reference = encoding.mean(dim=0, keepdim=True)
                distance = torch.norm(encoding - self.reference, dim=1, keepdim=True)
                prob = torch.sigmoid(distance)
                return torch.cat([1 - prob, prob], dim=1)  # [normal_prob, anomaly_prob]
            else:
                # Standard autoencoder: compute reconstruction error
                reconstructed = self.base_model(x)
                error = torch.mean((x - reconstructed) ** 2, dim=1, keepdim=True)
                # Convert to probability-like output (higher error = more anomalous)
                # Use sigmoid to get values between 0 and 1
                prob = torch.sigmoid(error)
                return torch.cat([1 - prob, prob], dim=1)  # [normal_prob, anomaly_prob]
    
    # Ensure model is on correct device and in eval mode
    model.eval()
    wrapped_model = AnomalyClassifier(model)
    wrapped_model = wrapped_model.to(device)
    wrapped_model.eval()
    
    # Wrap model for ART
    classifier = PyTorchClassifier(
        model=wrapped_model,
        loss=nn.CrossEntropyLoss(),
        input_shape=(X.shape[1],),
        nb_classes=2,
        device_type=device,
    )
    
    # Create attack
    if attack_type == "fgsm":
        attack = FastGradientMethod(estimator=classifier, eps=eps)
    elif attack_type == "pgd":
        attack = ProjectedGradientDescent(estimator=classifier, eps=eps, max_iter=10)
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")
    
    # Generate adversarial examples
    X_adv = attack.generate(x=X)
    
    # Get predictions on adversarial examples
    predictions = classifier.predict(X_adv)
    
    return X_adv, predictions


def explain_predictions(
    model: nn.Module,
    inputs: torch.Tensor,
    method: str = "integrated_gradients",
    device: str = "cpu",
) -> torch.Tensor:
    """Generate explanations for model predictions using Captum.
    
    Args:
        model: Trained model
        inputs: Input tensor to explain
        method: Explanation method ('integrated_gradients', 'gradient_shap', or 'saliency')
        device: Device to run on
        
    Returns:
        Attribution tensor
    """
    model.eval()
    inputs = inputs.to(device).float()  # Ensure float32
    
    # For autoencoder, we explain reconstruction error (scalar output)
    # Captum needs a single scalar output, not multiple outputs
    def model_wrapper(x):
        x = x.float()  # Ensure float32
        if hasattr(model, '__class__') and 'Autoencoder' in model.__class__.__name__:
            if 'Siamese' in model.__class__.__name__:
                # Siamese autoencoder: compute reconstruction error from single input
                # The forward() returns (reconstruction, encoding) when x2=None
                output = model(x)
                if isinstance(output, tuple):
                    reconstruction, encoding = output[0], output[1]
                else:
                    reconstruction = output
                    encoding = model.encode(x)
                
                # Compute reconstruction error as anomaly score
                error = torch.mean((x - reconstruction) ** 2, dim=1)
                return error  # Return scalar per sample
            else:
                # Standard autoencoder: return reconstruction error
                reconstructed = model(x)
                error = torch.mean((x - reconstructed) ** 2, dim=1)
                return error  # Return scalar per sample
        else:
            # Fallback: try to get encoding and compute distance
            try:
                encoding = model.encode(x)
                # Use mean encoding as reference
                if not hasattr(model_wrapper, 'reference_encoding'):
                    with torch.no_grad():
                        model_wrapper.reference_encoding = encoding.mean(dim=0, keepdim=True)
                distance = torch.norm(encoding - model_wrapper.reference_encoding, dim=1)
                return distance
            except:
                # Last resort: try forward pass and take first output
                output = model(x)
                if isinstance(output, tuple):
                    # Take reconstruction from tuple
                    reconstruction = output[0]
                    error = torch.mean((x - reconstruction) ** 2, dim=1)
                    return error
                else:
                    error = torch.mean((x - output) ** 2, dim=1)
                    return error
    
    if method == "integrated_gradients":
        explainer = IntegratedGradients(model_wrapper)
        # For scalar output, we don't need target
        attributions = explainer.attribute(inputs, n_steps=50)
    elif method == "gradient_shap":
        explainer = GradientShap(model_wrapper)
        baseline = torch.zeros_like(inputs).float()
        attributions = explainer.attribute(inputs, baselines=baseline, n_samples=50)
    elif method == "saliency":
        explainer = Saliency(model_wrapper)
        attributions = explainer.attribute(inputs)
    else:
        raise ValueError(f"Unknown explanation method: {method}")
    
    return attributions


def plot_feature_importance(
    attributions: torch.Tensor,
    feature_names: Optional[List[str]] = None,
    top_k: int = 10,
    save_path: Optional[str] = None,
) -> None:
    """Plot feature importance from model explanations.
    
    Args:
        attributions: Attribution tensor from explainer
        feature_names: Optional list of feature names
        top_k: Number of top features to display
        save_path: Optional path to save figure
    """
    # Average attributions across samples
    if attributions.dim() > 1:
        mean_attributions = torch.abs(attributions).mean(dim=0).cpu().numpy()
    else:
        mean_attributions = torch.abs(attributions).cpu().numpy()
    
    # Get top k features
    top_indices = np.argsort(mean_attributions)[-top_k:][::-1]
    top_attributions = mean_attributions[top_indices]
    
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(mean_attributions))]
    
    top_feature_names = [feature_names[i] for i in top_indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top_attributions)), top_attributions)
    ax.set_yticks(range(len(top_attributions)))
    ax.set_yticklabels(top_feature_names)
    ax.set_xlabel("Attribution Score", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title(f"Top {top_k} Feature Importances", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

