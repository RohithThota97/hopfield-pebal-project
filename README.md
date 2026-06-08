# Modern Hopfield Networks for Out-of-Distribution Detection in Semantic Segmentation

A deep learning project implementing Modern Hopfield Networks for detecting out-of-distribution (OOD) samples in semantic segmentation tasks.

## Project Overview

This project combines Hopfield memory mechanisms with modern neural networks to improve semantic segmentation performance and provide robust out-of-distribution detection. The model learns to identify pixels that are outside the training distribution, which is crucial for safe deployment in real-world scenarios.

## Repository Structure

```
├── src/                          # Source code organized by functionality
│   ├── core/                    # Main programs and core components
│   │   ├── main.py
│   │   ├── run_training.py      # Main training script
│   │   ├── run_training_ID.py   # In-distribution training
│   │   ├── feature_extractor.py
│   │   └── hopfield_memory_builder.py
│   │
│   ├── models/                  # Model architectures
│   │   ├── resnet.py
│   │   ├── network.py
│   │   ├── wide_network.py
│   │   └── wider_resnet.py
│   │
│   ├── training/                # Training and evaluation
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   ├── ood_metrics.py
│   │   └── ood_evaluator.py
│   │
│   ├── utils/                   # Utility functions
│   │   ├── logger.py
│   │   ├── metric.py
│   │   ├── img_utils.py
│   │   └── wandb_upload.py
│   │
│   ├── analysis/                # Analysis and testing tools
│   │   ├── ablation.py
│   │   ├── visualizations.py
│   │   ├── energy_analyzer.py
│   │   └── memory_analyser.py
│   │
│   ├── datasets/                # Dataset utilities
│   └── legacy/                  # Legacy code
│
├── test_images/                 # Test images and visualization outputs
│   ├── ood_results/
│   ├── results/
│   ├── semantic_results/
│   └── visualizations/
│
├── datasets/                    # Dataset directory
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
└── README.md                    # This file
```

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (optional, for GPU support)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/RohithThota97/Modern-Hopfield-Networks-for-Out-of-Distribution-Detection-in-Semantic-Segmentation.git
cd Modern-Hopfield-Networks-for-Out-of-Distribution-Detection-in-Semantic-Segmentation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install in development mode:
```bash
pip install -e .
```

## Usage

### Training

#### In-Distribution Training
```bash
python src/core/run_training_ID.py --config config/training.yml
```

#### OOD-Aware Training
```bash
python src/core/run_training.py --config config/training.yml
```

### Evaluation

```bash
python src/training/evaluator.py --model-path checkpoints/model.pth --data-path datasets/
```

### Analysis

```bash
python src/analysis/ablation.py
python src/analysis/visualizations.py
```

## Key Components

- **Modern Hopfield Networks**: Memory-augmented architecture for improved feature learning
- **Feature Extractor**: Extracts semantic features from input images
- **OOD Metrics**: Comprehensive metrics for OOD detection evaluation
- **Visualization Tools**: Analysis and visualization of network behavior

## Configuration

Configuration files are located in `src/core/config/`. Key parameters:
- `batch_size`: Training batch size
- `learning_rate`: Initial learning rate
- `epochs`: Number of training epochs
- `model_type`: Architecture selection

## Results

Results and visualizations are stored in `test_images/`:
- OOD detection results: `test_images/ood_results/`
- Semantic segmentation results: `test_images/semantic_results/`
- Training visualizations: `test_images/visualizations/`

## Citation

If you use this project, please cite:

```bibtex
@article{HopfieldNetworks2024,
  title={Modern Hopfield Networks for Out-of-Distribution Detection in Semantic Segmentation},
  author={Author Name},
  journal={Journal Name},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on the GitHub repository.
