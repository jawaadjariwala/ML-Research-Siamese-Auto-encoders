"""Final Project: Siamese Autoencoder for Network Intrusion Detection."""

from finalproject.models import (
    Autoencoder,
    SiameseAutoencoder,
    contrastive_loss,
)
from finalproject.data import (
    NSLKDDDataset,
    load_nsl_kdd_data,
    create_data_loaders,
)
from finalproject.train import (
    train_siamese_autoencoder,
    train_autoencoder_baseline,
    CurriculumScheduler,
    create_siamese_pairs,
)
from finalproject.utils import (
    compute_anomaly_scores_autoencoder,
    compute_anomaly_scores_siamese,
    evaluate_model,
    plot_training_history,
    plot_confusion_matrix,
    plot_roc_curve,
    detect_data_drift,
    create_adversarial_examples,
    explain_predictions,
    plot_feature_importance,
)

__all__ = [
    "Autoencoder",
    "SiameseAutoencoder",
    "contrastive_loss",
    "NSLKDDDataset",
    "load_nsl_kdd_data",
    "create_data_loaders",
    "train_siamese_autoencoder",
    "train_autoencoder_baseline",
    "CurriculumScheduler",
    "create_siamese_pairs",
    "compute_anomaly_scores_autoencoder",
    "compute_anomaly_scores_siamese",
    "evaluate_model",
    "plot_training_history",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "detect_data_drift",
    "create_adversarial_examples",
    "explain_predictions",
    "plot_feature_importance",
]

