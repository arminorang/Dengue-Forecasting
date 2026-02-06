You could build a multimodal, explainable “Structural Dengue Vulnerability Forecaster” that predicts short‑term dengue risk while showing how structural inequality and environment contribute to each prediction at the neighborhood/municipality level. 
Core idea

    Train a multimodal model to predict dengue incidence or outbreak probability 1–3 months ahead at district/municipality level, using: past dengue cases, socioeconomic/inequality indicators, climate, and satellite‑derived urban form.

    Wrap the model in an explainability layer that decomposes each prediction into contributions from structural inequality vs climate vs urban environment, providing global (across Brazil) and local (per district) explanations.

    Data modalities and prediction target

    Target
     
        Monthly dengue cases or a binary “outbreak” label per district/municipality, e.g., as done in GeoSeeq Dengue and Rio‑focused forecasting.

    Modalities
    Structural inequality: census variables (income, overcrowding, sanitation, race, education, access to health services), leveraging Brazilian inequality and health surveys.
    Historical epidemiology: past dengue (and possibly zika/chikungunya) case counts and rates.
    Climate: temperature, rainfall, humidity indices at monthly resolution.

    Satellite imagery: Sentinel‑2 or a prepared dengue imagery dataset to capture urban density, vegetation, standing water proxies, as in DengueNet‑style work.

    Model and XAI design
    Multimodal architecture
        A tabular branch (gradient boosting or MLP) for structural and time‑series features, plus a CNN/ViT encoder for satellite tiles; fuse embeddings before the prediction head.

    Benchmark against a strong tabular model (e.g., CatBoost) like previous explainable dengue forecasters used for Rio de Janeiro.

    Explainability components
    Global feature importance (SHAP/TreeSHAP or permutation importance) to show, on average, how much structural inequality vs climate vs history drives forecasts.
    Local explanations per district and month (SHAP values, ICE plots) to reveal why the model flags a particular vulnerable area.
    For imagery, Grad‑CAM or attention maps over satellite tiles to highlight where in the urban landscape the model “looks” when associating risk (e.g., dense informal settlements vs green zones).

    A simple decomposition: variance explained or performance drop when you ablate entire blocks of features (inequality vs climate vs imagery) to quantify their relative importance.

    How it centers structural inequality
    Make structural features first‑class citizens: explicitly group features into “structural inequality”, “climate”, “vector/environment”, and “history” and plot their grouped contributions for each prediction.
    Use partial‑dependence and SHAP interaction plots to examine how inequality and climate interact: e.g., whether heavy rainfall only becomes high‑risk in areas with poor sanitation and high crowding.

    Tie explanations back to the vulnerability framing of the British Academy project, which argues that dengue risk is produced by socio‑spatial inequalities in Brazil rather than just weather–mosquito dynamics.

    Suggested repo deliverables
    Clean, reproducible pipeline: data download/preprocessing scripts for each modality, model training script, evaluation script, and a small demo notebook that:
        trains a minimal multimodal model on 1–2 states or a single metropolis (e.g., Rio de Janeiro),

    generates maps of predicted risk vs observed cases, and

    renders a few case‑study districts with SHAP bar plots + satellite activation maps that a public‑health reader can interpret.

    If you tell me your preferred framework (PyTorch Lightning vs pure PyTorch vs scikit‑learn + LightGBM) I can sketch a concrete architecture and XAI workflow that’s realistic to implement in a few weeks

Help me set up this project and get started
3:21 p.m.

I'll help you set up this Structural Dengue Vulnerability Forecaster project. This is an excellent research direction that combines public health impact with technical innovation in explainable AI.

Let me create a comprehensive project structure and starter code to get you going:
Dengue Vulnerability Forecaster - Project Setup
Code 

Now let me create the foundational code files to get you started:
requirements.txt
Code 
Data Download Script - src/data/download.py
Code 
Multimodal Model Architecture - src/models/multimodal.py
Code 


Claude is AI and can make mistakes. Please double-check responses.
# Structural Dengue Vulnerability Forecaster

## Project Overview
A multimodal explainable AI system that predicts dengue risk 1-3 months ahead while decomposing contributions from structural inequality, climate, and urban environment at the neighborhood/municipality level in Brazil.

## Project Structure

```
dengue-forecaster/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── config.yaml              # Main configuration
│   └── data_sources.yaml        # Data source URLs and paths
├── data/
│   ├── raw/                     # Raw downloaded data
│   │   ├── epidemiology/
│   │   ├── socioeconomic/
│   │   ├── climate/
│   │   └── satellite/
│   ├── processed/               # Cleaned, aligned data
│   └── features/                # Engineered features
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_explainability_demo.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py          # Data acquisition scripts
│   │   ├── preprocessing.py     # Data cleaning
│   │   └── loaders.py           # Dataset classes
│   ├── features/
│   │   ├── __init__.py
│   │   ├── structural.py        # Inequality features
│   │   ├── climate.py           # Climate features
│   │   ├── epidemiology.py      # Historical case features
│   │   └── satellite.py         # Image processing
│   ├── models/
│   │   ├── __init__.py
│   │   ├── multimodal.py        # Main multimodal architecture
│   │   ├── tabular.py           # Tabular branch (CatBoost/MLP)
│   │   ├── image.py             # CNN/ViT for satellite
│   │   └── baseline.py          # Baseline models
│   ├── explainability/
│   │   ├── __init__.py
│   │   ├── shap_analysis.py     # SHAP explanations
│   │   ├── gradcam.py           # Satellite attention maps
│   │   ├── ablation.py          # Feature group ablation
│   │   └── visualization.py     # Plots and maps
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py           # Performance metrics
│   │   └── validation.py        # Cross-validation schemes
│   └── utils/
│       ├── __init__.py
│       ├── spatial.py           # Spatial operations
│       └── config.py            # Config loading
├── scripts/
│   ├── download_data.sh         # Automated data download
│   ├── train_model.py           # Training script
│   ├── evaluate.py              # Evaluation script
│   └── generate_explanations.py # XAI generation
├── tests/
│   └── test_*.py
└── outputs/
    ├── models/                  # Saved model checkpoints
    ├── predictions/             # Forecast outputs
    ├── explanations/            # SHAP values, etc.
    └── figures/                 # Visualizations
```

## Technology Stack Recommendation

**Framework: PyTorch Lightning + CatBoost Hybrid**

### Why this combination?
- **PyTorch Lightning**: Clean, scalable for the image branch and multimodal fusion
- **CatBoost**: Excellent baseline for tabular data, native categorical handling, fast SHAP
- **Hybrid approach**: Use CatBoost for pure tabular baseline, PyTorch for multimodal

### Core Dependencies
```
# Deep Learning
pytorch>=2.0.0
pytorch-lightning>=2.0.0
torchvision>=0.15.0
timm>=0.9.0  # Pre-trained vision models

# Tabular ML
catboost>=1.2.0
lightgbm>=4.0.0
xgboost>=2.0.0

# Explainability
shap>=0.42.0
captum>=0.6.0  # PyTorch interpretability
scikit-learn>=1.3.0

# Geospatial
geopandas>=0.13.0
rasterio>=1.3.0
earthengine-api>=0.1.360  # Google Earth Engine
sentinelhub>=3.9.0  # Sentinel-2 data

# Data & Viz
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0

# Utils
pyyaml>=6.0
tqdm>=4.65.0
wandb>=0.15.0  # Experiment tracking
```

## Key Data Sources for Brazil

### 1. Epidemiological Data
- **SINAN (Sistema de Informação de Agravos de Notificação)**
  - Source: Brazilian Ministry of Health
  - URL: http://tabnet.datasus.gov.br/
  - Format: Tabular (CSV/DBF)
  - Coverage: Monthly dengue cases by municipality
  
- **InfoDengue**
  - Source: Fiocruz
  - API: https://info.dengue.mat.br/api/
  - Format: JSON/CSV
  - Coverage: Weekly epidemiological data, alarms

### 2. Structural Inequality Data
- **IBGE Census 2022**
  - Source: Brazilian Institute of Geography and Statistics
  - URL: https://www.ibge.gov.br/
  - Variables: Income, education, race, household density, sanitation
  
- **IPEA Social Indicators**
  - Source: Institute of Applied Economic Research
  - Inequality indices, Gini coefficients

### 3. Climate Data
- **INMET (National Meteorology Institute)**
  - URL: https://portal.inmet.gov.br/
  - Variables: Temperature, rainfall, humidity
  - Resolution: Daily, station-based
  
- **ERA5-Land** (Alternative)
  - Source: Copernicus Climate Data Store
  - Global gridded reanalysis at 9km

### 4. Satellite Imagery
- **Sentinel-2** (10m resolution)
  - Source: ESA/Copernicus
  - Access: Google Earth Engine or SentinelHub
  - Bands: RGB + NIR for vegetation indices
  
- **DengueNet Dataset** (if available)
  - Pre-processed satellite tiles for dengue-prone areas

## Initial Implementation Phases

### Phase 1: Data Pipeline (Week 1-2)
1. Download sample data for Rio de Janeiro state
2. Align temporal resolution (monthly aggregation)
3. Spatial alignment of municipalities
4. Create train/val/test splits (temporal)

### Phase 2: Baseline Models (Week 2-3)
1. Tabular-only CatBoost model
2. Feature importance analysis
3. Establish performance benchmarks

### Phase 3: Multimodal Architecture (Week 3-4)
1. Image encoder (ResNet/EfficientNet)
2. Tabular encoder (MLP)
3. Fusion module
4. End-to-end training

### Phase 4: Explainability (Week 4-5)
1. SHAP for tabular features
2. Grad-CAM for satellite images
3. Feature group ablation
4. Interactive visualizations

### Phase 5: Case Studies (Week 5-6)
1. Select 5-10 representative municipalities
2. Generate local explanations
3. Create interpretable reports
4. Validate with domain experts

## Starter Configuration

```yaml
# config/config.yaml
project:
  name: dengue_vulnerability_forecaster
  experiment_name: rio_baseline_v1
  
data:
  region: rio_de_janeiro  # Start with single state
  temporal_resolution: monthly
  forecast_horizon: [1, 2, 3]  # months ahead
  train_years: [2015, 2016, 2017, 2018, 2019]
  val_years: [2020]
  test_years: [2021, 2022]
  
features:
  structural:
    - income_per_capita
    - gini_coefficient
    - overcrowding_rate
    - inadequate_sanitation_pct
    - black_brown_population_pct
    - education_years_avg
    - health_access_index
  
  climate:
    - temperature_mean
    - temperature_max
    - rainfall_total
    - humidity_mean
    - temperature_range
  
  epidemiology:
    - dengue_cases_lag_1m
    - dengue_cases_lag_2m
    - dengue_cases_lag_3m
    - dengue_rate_lag_1m
    - cases_moving_avg_3m
    - cases_trend
  
  satellite:
    - image_size: [224, 224]
    - bands: [red, green, blue, nir]
    - derived: [ndvi, ndwi]

model:
  architecture: multimodal_fusion
  tabular_branch:
    type: catboost  # or mlp
    params:
      iterations: 1000
      learning_rate: 0.03
      depth: 6
  
  image_branch:
    backbone: resnet50  # or efficientnet_b0
    pretrained: true
    freeze_backbone: false
  
  fusion:
    method: concatenate  # or attention
    hidden_dims: [256, 128]
    dropout: 0.3
  
training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  optimizer: adam
  scheduler: reduce_on_plateau
  early_stopping_patience: 10
  
explainability:
  shap:
    feature_groups:
      structural: [income_per_capita, gini_coefficient, ...]
      climate: [temperature_mean, rainfall_total, ...]
      epidemiology: [dengue_cases_lag_1m, ...]
    n_samples: 100
  
  gradcam:
    target_layer: layer4
  
  ablation:
    groups: [structural, climate, epidemiology, satellite]
```

## Quick Start Commands

```bash
# 1. Setup environment
conda create -n dengue python=3.10
conda activate dengue
pip install -r requirements.txt

# 2. Download sample data (Rio de Janeiro)
python scripts/download_data.sh --region rio_de_janeiro --years 2015-2022

# 3. Preprocess data
python -m src.data.preprocessing --config config/config.yaml

# 4. Train baseline (tabular only)
python scripts/train_model.py --model baseline_catboost --config config/config.yaml

# 5. Train multimodal model
python scripts/train_model.py --model multimodal --config config/config.yaml

# 6. Generate explanations
python scripts/generate_explanations.py --model_path outputs/models/best.ckpt

# 7. Launch demo notebook
jupyter notebook notebooks/04_explainability_demo.ipynb
```

## Expected Outputs

### 1. Performance Metrics
- MAE, RMSE for case count prediction
- F1, Precision, Recall for outbreak detection
- Temporal validation plots
- Spatial error maps

### 2. Global Explanations
- Feature importance bar charts (grouped by modality)
- Partial dependence plots
- Feature interaction heatmaps
- Variance decomposition pie chart

### 3. Local Explanations
- Per-municipality SHAP waterfall plots
- Satellite attention maps overlaid on urban geography
- Time-series decomposition of risk factors
- Comparative case studies (high vs low vulnerability)

### 4. Interactive Dashboard (Optional)
- Map-based risk visualization
- Click municipality → see explanation
- Toggle feature groups on/off
- Export reports

## Next Steps

1. **Choose your starting region**: Rio de Janeiro (rich data) or São Paulo (larger scale)?
2. **Data access**: Do you have institutional access to SINAN/IBGE APIs?
3. **Compute resources**: Local GPU or cloud (Colab/Kaggle)?
4. **Timeline**: Research prototype (6 weeks) or production system (3+ months)?

I can provide detailed starter code for any component once you confirm your preferences!

# Dengue-Forecasting
