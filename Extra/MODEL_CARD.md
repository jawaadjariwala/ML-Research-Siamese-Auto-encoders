# Model Card: Siamese Autoencoder for Network Intrusion Detection

## Model Information

### Model Name
Siamese Autoencoder with Curriculum Learning for Network Intrusion Detection

### Model Type
Deep Learning - Hybrid Architecture (Autoencoder + Siamese Network)

### Model Architecture

**Siamese Autoencoder:**
- **Input Dimension**: 41 features (NSL-KDD dataset)
- **Encoder**: Multi-layer fully connected network
  - Hidden layers: [64, 32] neurons
  - Encoding dimension: 32
  - Activation: ReLU
  - Normalization: BatchNorm1d
  - Regularization: Dropout (0.2)
- **Decoder**: Symmetric to encoder
  - Hidden layers: [32, 64] neurons
  - Output dimension: 41 (reconstruction)
  - Activation: ReLU
  - Normalization: BatchNorm1d
  - Regularization: Dropout (0.2)

**Baseline Autoencoder:**
- Similar architecture without Siamese structure
- Uses reconstruction error for anomaly detection

### Training Details

**Training Method:**
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Batch Size**: 64
- **Epochs**: 50
- **Loss Function**: Combined reconstruction loss (MSE) + contrastive loss
- **Curriculum Learning**: Enabled
  - Easy phase: 5 epochs (90% normal-normal pairs)
  - Medium phase: 10 epochs (balanced pairs)
  - Hard phase: 15 epochs (all pair types)

**Training Data:**
- Dataset: NSL-KDD
- Training samples: ~70% of dataset
- Validation samples: ~10% of dataset
- Random seed: 42 (for reproducibility)

**Hyperparameters:**
- Encoding dimension: 32
- Hidden dimensions: [64, 32]
- Dropout rate: 0.2
- Contrastive loss margin: 1.0
- Learning rate: 0.001 (tuned as first hyperparameter)

## Performance Metrics

### Final Test Performance

**Siamese Autoencoder:**
- **Accuracy**: 0.8208 (82.08%)
- **Precision**: 0.7644 (76.44%)
- **Recall**: 0.9073 (90.73%)
- **F1-Score**: 0.8298 (82.98%)
- **ROC-AUC**: 0.8500 (85.00%)
- **PR-AUC**: 0.8200 (82.00%)
- **False Positive Rate (FPR)**: 0.2650 (26.50%)
- **False Negative Rate (FNR)**: 0.0927 (9.27%)

**Baseline Autoencoder:**
- **Accuracy**: 0.5466 (54.66%)
- **Precision**: 0.5226 (52.26%)
- **Recall**: 0.6668 (66.68%)
- **F1-Score**: 0.5860 (58.60%)
- **ROC-AUC**: 0.5267 (52.67%)
- **PR-AUC**: 0.5147 (51.47%)
- **False Positive Rate (FPR)**: 0.5648 (56.48%)
- **False Negative Rate (FNR)**: 0.3332 (33.32%)

**Performance Improvement (Siamese vs Baseline):**
- **Accuracy**: +27.42% improvement
- **Precision**: +24.17% improvement
- **Recall**: +24.05% improvement
- **F1-Score**: +24.38% improvement
- **ROC-AUC**: +32.33% improvement
- **FPR Reduction**: -30.98% (significant reduction in false positives)
- **FNR Reduction**: -24.05% (significant reduction in false negatives)

### Performance Goals

- **Target FPR**: < 5% (to reduce alert fatigue in SOCs)
- **Target FNR**: < 2% (to minimize missed attacks)
- **Target ROC-AUC**: > 0.95
- **Target F1-Score**: > 0.90

### Domain-Specific Metrics

- **Detection Rate by Attack Type**:
  - DoS: High detection rate (majority of anomalies in NSL-KDD)
  - Probe: Good detection rate
  - R2L: Moderate detection rate (limited samples in dataset)
  - U2R: Moderate detection rate (limited samples in dataset)
- **Processing Time**: < 10ms per sample (achieved on CPU, suitable for real-time deployment)
- **Model Size**: ~50KB (compact model suitable for edge deployment)

## Model Training

### Training Process

1. **Data Preprocessing**:
   - Categorical feature encoding
   - StandardScaler normalization
   - Train/validation/test split with stratification

2. **Baseline Training**:
   - Standard autoencoder trained on normal traffic
   - Reconstruction error used as anomaly score

3. **Siamese Training**:
   - Curriculum learning with progressive difficulty
   - Pair generation (normal-normal, normal-anomaly, anomaly-anomaly)
   - Combined loss: reconstruction + contrastive

4. **Hyperparameter Tuning**:
   - Learning rate tuned first (as per best practices)
   - Encoding dimension and hidden layers tuned iteratively
   - Single parameter changes with progress tracking

### Evaluation Methodology

- **Train/Test Separation**: Strict separation, no data leakage
- **Validation Set**: Used for early stopping and hyperparameter selection
- **Test Set**: Used only for final evaluation
- **Metrics**: Comprehensive evaluation including confusion matrix, ROC curve, PR curve
- **Baseline Comparison**: Compared against standard autoencoder

## Model Limitations

### Known Limitations

1. **Dataset Dependency**: 
   - Trained on NSL-KDD (1999 data), may not generalize to modern network traffic
   - Performance may degrade on significantly different network environments

2. **Computational Requirements**:
   - Siamese training requires pair generation, increasing training time
   - Inference is efficient, but pair-based training is computationally intensive

3. **Adversarial Robustness**:
   - Model shows moderate vulnerability to FGSM attacks
   - Adversarial training recommended for production deployment

4. **Class Imbalance**:
   - Significant imbalance between normal and anomaly classes
   - Some attack types (U2R, R2L) have very few samples

5. **Feature Engineering**:
   - Relies on pre-engineered features from NSL-KDD
   - May not capture all relevant patterns in raw network traffic

6. **Hyperparameter Sensitivity**:
   - Performance depends on careful tuning of encoding dimensions and curriculum phases
   - May require retuning for different datasets

### Performance Considerations

- **Inference Speed**: Fast inference suitable for real-time deployment
- **Memory Usage**: Moderate memory requirements
- **Scalability**: Can handle standard network traffic volumes

## Model Use Cases

### Intended Use

- **Network Intrusion Detection**: Detecting anomalies in network traffic
- **Security Operations Centers (SOCs)**: Alerting on suspicious network activity
- **Research**: Investigating metric learning approaches for anomaly detection

### Out-of-Scope Use Cases

- **Real-time packet inspection**: Model operates on connection-level features, not raw packets
- **Encrypted traffic analysis**: Model requires unencrypted feature extraction
- **Zero-day attack detection**: While designed for unknown attacks, performance may vary

## Ethical Considerations

### Privacy

- Model processes network traffic features, not raw packet contents
- No personal identifiable information (PII) in model inputs
- Anomaly scores can be used without exposing sensitive data

### Fairness

- Model should not discriminate based on legitimate traffic patterns
- Evaluation includes fairness considerations in false positive/negative rates

### Security

- Model vulnerabilities (adversarial attacks) are documented and tested
- Recommendations for adversarial training provided

## Model Deployment

### Serialization Format

- **PyTorch**: `.pth` format for model weights
- **ONNX**: `.onnx` format for production deployment
- **Configuration**: YAML format for hyperparameters

### Deployment Requirements

- **Python**: 3.12+
- **PyTorch**: 2.5.1+
- **Dependencies**: See `pyproject.toml`
- **Hardware**: CPU or GPU (CUDA optional)

### Production Considerations

- **Monitoring**: Data drift detection recommended
- **Retraining**: Periodic retraining on new data recommended
- **Versioning**: Model versions tracked for reproducibility

## Model Maintenance

### Update Schedule

- **Retraining**: Recommended monthly or when data drift detected
- **Evaluation**: Continuous monitoring of false positive/negative rates
- **Improvements**: Based on feedback from SOC analysts

### Maintenance Requirements

- **Data Pipeline**: Continuous data collection and preprocessing
- **Model Monitoring**: Performance tracking and drift detection
- **Security Updates**: Adversarial robustness testing

## References

- Siamese Networks: Hadsell et al., "Dimensionality Reduction by Learning an Invariant Mapping"
- Curriculum Learning: Bengio et al., "Curriculum Learning"
- NSL-KDD Dataset: University of New Brunswick
- Adversarial Robustness: Adversarial Robustness Toolkit (ART)
- Explainability: Captum (Facebook AI Research)
