# Dengue Forecasting with Spatiotemporal Graph Neural Networks

A machine learning project leveraging Graph Neural Networks (GNNs) to predict dengue incidence across Brazilian municipalities using spatial adjacency and temporal patterns.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Performance](#performance)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Overview

Dengue fever is a major public health concern in tropical and subtropical regions. This project develops a spatiotemporal forecasting model that:

- **Captures spatial dependencies** between neighboring municipalities using graph structures
- **Models temporal patterns** through sequential dengue incidence data
- **Predicts future outbreaks** one week ahead at the municipality level
- **Achieves competitive performance** with MASE of 0.92 (beating naive baseline by 8%)

The model combines Graph Convolutional Networks (GCN) for spatial feature learning with Gated Recurrent Units (GRU) for temporal sequence modeling.

## Key Features

- **Spatiotemporal Modeling**: Integrates both geographic adjacency and time-series patterns
- **Multi-scale Features**: Incorporates static features (area, regional codes) and dynamic features (incidence rates)
- **Graph-Based Architecture**: Leverages municipality adjacency for disease spread modeling
- **Scalable**: Handles 5,570+ municipalities across Brazil
- **Production-Ready**: Includes data processing pipelines, model training, and evaluation metrics

## Architecture

```
Input Layer
    ├── Dynamic Features (4-week sequence of incidence rates)
    └── Static Features (area, regional codes)
           ↓
Graph Convolutional Layers (2 layers, 32 hidden units)
    └── Spatial feature extraction via GCN
           ↓
Gated Recurrent Unit (64 hidden units)
    └── Temporal sequence modeling
           ↓
MLP Prediction Head
    └── Final incidence prediction
```

### Model Components

1. **Input Processing**
   - Dynamic: 4 weeks of historical incidence per 100k population
   - Static: Geographic area (km²) + 5 regional categorical features

2. **Graph Convolutional Network**
   - 2 GCN layers with ReLU activation
   - 32-dimensional hidden representations
   - Captures spatial disease spread patterns

3. **Temporal Modeling**
   - GRU with 64 hidden units
   - Processes graph-enhanced features over time
   - Captures weekly outbreak dynamics

4. **Output Layer**
   - 2-layer MLP (64 → 64 → 1)
   - Predicts next week's incidence rate

## 📊 Performance

Evaluation on held-out test set (20% of temporal data):

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **MASE** | 0.92 | Beats naive baseline by 8% |
| **RMSSE** | 0.69 | 31% better than naive at avoiding large errors |
| **MAE** | 44.6 | Average error of 44.6 cases per 100k |
| **RMSE** | 126.4 | Root mean squared error |
| **R²** | 0.54 | Explains 54% of variance |

**Key Findings:**
- ✅ Model outperforms naive baseline (MASE < 1.0)
- ✅ Strong performance on avoiding catastrophic errors (RMSSE = 0.69)
- ✅ Captures meaningful spatial-temporal patterns (R² = 0.54)

## 📁 Dataset

### Sources
- **Dengue Data**: Brazilian Ministry of Health surveillance system
- **Geographic Data**: IBGE (Brazilian Institute of Geography and Statistics)
- **Coverage**: 5,570 municipalities across Brazil
- **Time Period**: Multiple years of weekly epidemiological data

### Features

**Dynamic (Time-Varying)**
- `p_inc100k`: Dengue incidence per 100,000 population
- Weekly measurements by municipality

**Static (Municipality-Level)**
- `AREA_KM2`: Geographic area in square kilometers
- `CD_UF`: State code
- `CD_RGI`: Immediate geographic region
- `CD_RGINT`: Intermediate geographic region
- `CD_REGIAO`: Macro-region code
- `CD_CONCURB`: Metropolitan area code

### Data Structure
```
data/
├── processed/
│   ├── dengue_modeling.parquet    # Processed dengue time series
│   └── edge_index.npy              # Spatial adjacency graph
└── map/
    └── brazil_municipalities.gpkg  # Geographic boundaries
```

## 🚀 Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (optional, for faster training)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/dengue-gnn-forecasting.git
cd dengue-gnn-forecasting
```

2. **Create virtual environment**
```bash
conda create -n dengue python=3.10
conda activate dengue
```

3. **Install dependencies**
```bash
# PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# PyTorch Geometric
pip install torch-geometric

# Other dependencies
pip install -r requirements.txt
```

### Requirements
```
torch>=2.0.0
torch-geometric>=2.3.0
pandas>=2.0.0
numpy>=1.24.0
geopandas>=0.13.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
pyarrow>=12.0.0
```

## 💻 Usage

### 1. Data Preparation

Process raw dengue data and create spatial adjacency graph:

```python
import pandas as pd
import geopandas as gpd
import numpy as np

# Load geographic data
gdf = gpd.read_file("data/map/brazil_municipalities.gpkg")

# Load dengue data
df_dengue = pd.read_parquet("data/raw/dengue_data.parquet")

# Create spatial adjacency (run once)
# See notebooks/exploration.ipynb for full preprocessing
```

### 2. Train the Model

```python
from src.models.gnn import (
    DengueSpatioTemporalDataset,
    SpatioTemporalGNN,
    train_one_epoch,
    evaluate
)
import torch

# Load data
df = pd.read_parquet("data/processed/dengue_modeling.parquet")
edge_index = torch.from_numpy(np.load("data/processed/edge_index.npy"))

# Create datasets
train_ds = DengueSpatioTemporalDataset(df, edge_index, T=4, train=True, train_frac=0.8)
test_ds = DengueSpatioTemporalDataset(df, edge_index, T=4, train=False, train_frac=0.8)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpatioTemporalGNN(
    dyn_in=1,
    static_in=6,
    gcn_hidden=32,
    gcn_layers=2,
    rnn_hidden=64
).to(device)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

for epoch in range(50):
    train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
    test_loss = evaluate(model, test_loader, loss_fn)
    print(f"Epoch {epoch+1} | Train: {train_loss:.4f} | Test: {test_loss:.4f}")
```

### 3. Make Predictions

```python
# Load trained model
model.load_state_dict(torch.load("models/gnn_dengue.pt"))
model.eval()

# Generate predictions
predictions = []
with torch.no_grad():
    for batch in test_loader:
        pred = model(*batch)
        predictions.append(pred.cpu().numpy())
```

### 4. Visualize Results

```python
import matplotlib.pyplot as plt

# Plot predictions vs actual
plt.scatter(targets, predictions, alpha=0.3)
plt.plot([0, max(targets)], [0, max(targets)], 'r--')
plt.xlabel('Actual Incidence')
plt.ylabel('Predicted Incidence')
plt.title('GNN Predictions vs Actual')
plt.show()
```

For complete examples, see `notebooks/exploration.ipynb`.

## Project Structure

```
dengue-gnn-forecasting/
├── data/
│   ├── processed/              # Processed datasets
│   │   ├── dengue_modeling.parquet
│   │   └── edge_index.npy
│   └── map/                    # Geographic data
│       └── brazil_municipalities.gpkg
├── notebooks/
│   └── exploration.ipynb       # Data exploration and model training
├── src/
│   └── models/
│       └── gnn.py             # GNN model implementation
├── models/                     # Saved model checkpoints
│   └── gnn_dengue.pt
├── results/                    # Visualizations and outputs
├── requirements.txt
└── README.md
```

## Methodology

### Graph Construction

Municipal adjacency is determined through:
- **Geographic neighbors**: Municipalities sharing borders
- **Bidirectional edges**: Disease can spread in both directions
- **Edge count**: ~20-30K edges for 5,570 nodes (sparse graph)

### Training Strategy

- **Temporal split**: 80% train, 20% test (chronological)
- **Sequence length**: 4 weeks of historical data
- **Prediction horizon**: 1 week ahead
- **Optimization**: Adam optimizer with learning rate 1e-3
- **Loss function**: Mean Squared Error (MSE)
- **Batch size**: 1 (full graph per batch)

### Evaluation Metrics

1. **MASE (Mean Absolute Scaled Error)**
   - Scales errors by naive forecast performance
   - < 1.0 indicates beating naive baseline

2. **RMSSE (Root Mean Squared Scaled Error)**
   - Similar to MASE but penalizes large errors
   - Useful for detecting catastrophic predictions

3. **R² Score**
   - Proportion of variance explained
   - Indicates model's explanatory power

## Results

### Model Performance

The spatiotemporal GNN demonstrates strong forecasting capability:

- **Spatial Learning**: Successfully captures disease spread patterns between adjacent municipalities
- **Temporal Dynamics**: Models weekly outbreak trends and seasonality
- **Generalization**: Maintains performance on unseen time periods (test set)

### Visualizations

Generated visualizations include:
- Training/validation loss curves
- Prediction scatter plots
- Residual analysis
- Geographic heatmaps of predictions vs actuals
- Time series comparisons for selected municipalities

See `notebooks/exploration.ipynb` for interactive visualizations.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{dengue_gnn_2025,
  title={Dengue Forecasting with Spatiotemporal Graph Neural Networks},
  author={Your Name},
  year={2025},
  publisher={GitHub},
  url={https://github.com/yourusername/dengue-gnn-forecasting}
}
```

## 🙏 Acknowledgments

- Brazilian Ministry of Health for dengue surveillance data
- IBGE for geographic boundary data
- PyTorch Geometric team for the GNN framework
- Open-source contributors to the scientific Python ecosystem
