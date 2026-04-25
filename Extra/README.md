# Advancing Cybersecurity Anomaly Detection: Siamese Autoencoders with Curriculum Learning

**RUC Practical AI - Fall 2025 Final Project**

A comprehensive research project investigating Siamese Autoencoder architectures trained via Curriculum Learning for network intrusion detection. This project integrates adversarial robustness testing, explainable AI, and data drift detection to address key challenges in deploying deep learning models for cybersecurity applications.

## Purpose

This project addresses the critical challenge of network intrusion detection in modern cybersecurity systems. Traditional signature-based Intrusion Detection Systems (IDS) are fundamentally reactive and insufficient for detecting zero-day attacks and Advanced Persistent Threats (APTs). Anomaly-based detection using machine learning offers promise, but faces challenges including:

- **Sensitivity-Specificity Trade-off**: High sensitivity leads to false positives, while high specificity misses subtle attacks
- **Over-generalization**: Models may learn to reconstruct anomalies as normal
- **Black Box Opacity**: Lack of interpretability hinders adoption in Security Operations Centers (SOCs)
- **Adversarial Vulnerability**: Models may be susceptible to evasion attacks

Our solution combines:
- **Siamese Autoencoders**: Learn explicit distance metrics between normal and anomalous states
- **Curriculum Learning**: Progressive training from easy to hard examples for improved stability
- **Adversarial Robustness Testing**: Evaluate model resilience using the Adversarial Robustness Toolkit (ART)
- **Explainable AI**: Generate feature attributions using Captum for model interpretability

## Usage Instructions

### Prerequisites

- Python 3.12+
- Poetry (for dependency management)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd final-project-jawaadjariwala-main
   ```

2. **Install dependencies using Poetry:**
   ```bash
   poetry install
   ```

3. **Activate the Poetry environment:**
   ```bash
   poetry shell
   ```

### Running the Project

1. **Download the NSL-KDD dataset** (required):
   
   **Option 1: Automatic download (recommended)**
   ```bash
   python download_data.py
   ```
   
   **Option 2: Manual download**
   - Download from [University of New Brunswick](https://www.unb.ca/cic/datasets/nsl.html)
   - Or from [Kaggle](https://www.kaggle.com/datasets/hassan06/nslkdd)
   - Place `KDDTrain+.txt` and `KDDTest+.txt` in the `data/` directory
   
   The dataset will be automatically downloaded when you run the notebook if it's not present.

2. **Configure the project** (optional):
   Edit `config.yaml` to adjust model parameters, training settings, and evaluation options.
   The default `data_path` is set to `"data"` which will auto-download the dataset.

3. **Run the main notebook:**
   ```bash
   jupyter notebook notebooks/finalproject.ipynb
   ```
   
   Or using JupyterLab:
   ```bash
   jupyter lab notebooks/finalproject.ipynb
   ```

3. **The notebook includes:**
   - Data loading and preprocessing
   - Baseline autoencoder training
   - Siamese Autoencoder training with curriculum learning
   - Model evaluation and comparison
   - Adversarial robustness testing
   - Explainable AI analysis
   - Data drift detection
   - ONNX model export

### Using the Modules Directly

You can also import and use the modules programmatically:

```python
from finalproject import (
    SiameseAutoencoder,
    load_nsl_kdd_data,
    create_data_loaders,
    train_siamese_autoencoder,
    evaluate_model,
)

# Load data
X_train, X_val, X_test, y_train, y_val, y_test = load_nsl_kdd_data()

# Create data loaders
train_loader, val_loader, test_loader = create_data_loaders(
    X_train, X_val, X_test, y_train, y_val, y_test
)

# Initialize model
model = SiameseAutoencoder(input_dim=41, encoding_dim=32)

# Train model
history = train_siamese_autoencoder(
    model, train_loader, val_loader, num_epochs=50
)

# Evaluate
scores, labels = compute_anomaly_scores_siamese(model, test_loader)
metrics = evaluate_model(scores, labels)
```

## Project Structure

```
final-project-jawaadjariwala-main/
├── finalproject/          # Main Python package
│   ├── __init__.py       # Package initialization
│   ├── data.py           # Data loading and preprocessing
│   ├── models.py         # Neural network models
│   ├── train.py          # Training functions with curriculum learning
│   └── utils.py          # Evaluation, visualization, robustness testing
├── notebooks/            # Jupyter notebooks
│   └── finalproject.ipynb # Main project notebook
├── config.yaml           # Configuration file
├── README.md             # This file
├── DATA_CARD.md          # Dataset documentation
├── MODEL_CARD.md         # Model documentation
├── pyproject.toml         # Poetry dependencies
└── LICENSE               # License file
```

## Known Issues

1. **Dataset Download**: The dataset is automatically downloaded when you first run the notebook. If automatic download fails, you can manually download from [University of New Brunswick](https://www.unb.ca/cic/datasets/nsl.html) or run `python download_data.py`.

2. **Adversarial Robustness**: Some adversarial attacks may fail if the model architecture is not compatible with ART's requirements. The code includes error handling for these cases.

3. **Memory Usage**: Training on large datasets may require significant memory. Consider reducing batch size or dataset size if memory issues occur.

4. **CUDA Support**: The project defaults to CPU. To use GPU, set `device: "cuda"` in `config.yaml` and ensure CUDA is available.

## Feature Roadmap

### Short-term (Next Release)
- [ ] Support for CIC-IDS2017 dataset
- [ ] Adversarial training integration
- [ ] Real-time inference API
- [ ] Enhanced visualization tools

### Medium-term
- [ ] Ensemble methods for improved coverage
- [ ] Online learning capabilities
- [ ] Graph neural network integration
- [ ] Multi-attack type analysis

### Long-term
- [ ] Production deployment pipeline
- [ ] Distributed training support
- [ ] Federated learning implementation
- [ ] Causal analysis features

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository** and create a feature branch
2. **Follow code style**: Use `black` for formatting (run `poetry run black finalproject/`)
3. **Add tests**: Include unit tests for new functionality
4. **Update documentation**: Update README, docstrings, and relevant documentation
5. **Submit a pull request** with a clear description of changes

### Development Setup

```bash
# Install development dependencies
poetry install --with dev

# Run code formatting
poetry run black finalproject/

# Run type checking
poetry run mypy finalproject/

# Run linting
poetry run flake8 finalproject/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions, issues, or contributions, please contact:
- **Email**: [Your Email]
- **GitHub Issues**: [Repository Issues Page]

## Acknowledgments

- NSL-KDD dataset provided by University of New Brunswick
- Adversarial Robustness Toolkit (ART) by IBM Research
- Captum by Facebook AI Research
- PyTorch team for the deep learning framework

## Citation

If you use this project in your research, please cite:

```bibtex
@software{siamese_autoencoder_ids,
  title={Advancing Cybersecurity Anomaly Detection: Siamese Autoencoders with Curriculum Learning},
  author={[Your Name]},
  year={2024},
  url={[Repository URL]}
}
```
