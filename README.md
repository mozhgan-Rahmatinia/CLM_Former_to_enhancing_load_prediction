CLM-Former: 
Enhancing Multi-Horizon Time Series Forecasting in Smart Microgrids

This repository is the official implementation of the paper: "CLM-Former for Enhancing Multi-Horizon Time Series Forecasting and Load Prediction in Smart Microgrids Using a Robust Transformer-Based Model".

---------------------------------------------------------------------------------

📜 Abstract

Accurate multi-horizon load forecasting is essential for the stability and efficiency of smart grid operations. While Transformer-based models (e.g., Autoformer) excel at capturing long-term periodic trends, they often struggle with rapidly changing, localized patterns prevalent in residential data.

To address this, we propose CLM-Former, a novel hybrid architecture that integrates:
1- Series Decomposition to separate trend and seasonal components.
2- Auto-Correlation Mechanism for global periodic dependency discovery.
3- CLM-subNet (CNN+LSTM): A specialized bottleneck subnetwork designed to capture high-frequency local variations and sequential dependencies .

Experimental results on real-world datasets (Electricity, Traffic, ETT, Weather) demonstrate that CLM-Former consistently outperforms state-of-the-art baselines while maintaining high computational efficiency.

---------------------------------------------------------------------------------

🚀 Key Features & Novelty

- Hybrid "Divide and Conquer" Strategy: Simultaneously models global seasonality (via Autoformer backbone) and local dynamics (via CLM-subNet).
- Parameter Efficiency: Utilizes a "bottleneck" LSTM design that reduces parameter count (33.96M) compared to the official Autoformer baseline (34.62M).
- High Inference Speed: Achieves inference latency virtually identical to Autoformer (+1.4% avg), suitable for real-time smart grid deployment.
- Faster Convergence: Demonstrates up to ~31.8% faster training convergence on complex long-horizon tasks.

CLM-Former: Enhancing Multi-Horizon Time Series Forecasting in Smart Microgrids
This repository is the official implementation of the paper: "CLM-Former for Enhancing Multi-Horizon Time Series Forecasting and Load Prediction in Smart Microgrids Using a Robust Transformer-Based Model".

---------------------------------------------------------------------------------

🏗️ Model Architecture

1. Conceptual Framework
The model follows a hierarchical processing strategy. The input is decomposed, and the seasonal component undergoes dual refinement: global patterns via Auto-Correlation and local patterns via CLM-subNet.

2. Overall Architecture
The detailed Encoder-Decoder structure showing the integration of CLM-subNet within the decomposition blocks.

<img width="1035" height="326" alt="image" src="https://github.com/user-attachments/assets/c7083207-4694-47c9-b6d3-5a08ed0a4b53" />
Figure 2: The overall architecture of CLM-Former, highlighting data flow for Trend (Dark Blue) and Seasonal (Purple) components.



4. CLM-subNet Internals
The internal structure of the proposed subnetwork combining 1D-CNN layers for feature extraction and LSTM for sequential modeling.


<img width="535" height="230" alt="image" src="https://github.com/user-attachments/assets/772a48ff-886c-4c07-895d-94aec5742df9" />

Figure 3: The internal architecture of the proposed CLM-subNet.

---------------------------------------------------------------------------------

📊 Results

Performance Comparison (Electricity Dataset)
CLM-Former achieves the lowest MSE and MAE across all prediction horizons (96, 192, 336), outperforming both Transformer-based and Deep Learning baselines.
Impact of architectural components on forecasting accuracy (Ablation Study).

Computational Efficiency
Despite the addition of recurrent layers, the model maintains high efficiency. The figure below illustrates the favorable trade-off between Inference Time and Parameter Count.
Computational efficiency analysis: Comparison of Inference Time vs. Parameters.

---------------------------------------------------------------------------------

🔧 Getting Started

Requirements

The model was implemented using Python 3.11 and PyTorch 2.9.0. Install dependencies:


pip install -r requirements.txt
Main dependencies: torch, numpy, pandas, matplotlib.

Data Preparation
The primary dataset used is the ElectricityLoadDiagrams20112014 from the UCI Machine Learning Repository.

Download the dataset.
Place it in the ./data/electricity/ directory.
The code automatically handles Z-score normalization and 70/10/20 train/val/test splitting.


