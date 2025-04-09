Hopfield-PEBAL

Hopfield-PEBAL is an implementation of an out-of-distribution (OOD) detection and anomaly segmentation framework that integrates a modern Hopfield network with pixel-wise energy-biased abstention learning (PEBAL). The method is designed to improve anomaly detection in complex scenes—such as those encountered in autonomous driving—by combining conventional segmentation with a memory-assisted mechanism for enhanced OOD performance.

Overview

This repository contains code for:
	•	Base Segmentation Network: A deep segmentation backbone (e.g., DeepWV3Plus) modified to extract intermediate features.
	•	Hopfield Memory Module: A memory bank that stores representative feature vectors (prototypes) to help sharpen the decision boundary between in-distribution and OOD pixels.
	•	Energy-Based OOD Detection: An energy head and loss formulation that directly discourages high energy (i.e. uncertainty) on known classes while encouraging it on anomalies.
	•	Prototype-Based Contrastive Learning (Optional): Class-prototype buffers and corresponding update strategies can be used to further refine the OOD detection.

The architecture is tailored for challenging datasets like Cityscapes for inlier segmentation and COCO for auxiliary (outlier) exposure.

hopfield-pebal/
├── code/
│    └── model/
│         ├── __init__.py
│         ├── wide_network.py        # Contains DeepWV3Plus architecture and submodules.
│         ├── mynn.py
│         ├── wide_resnet_base.py
│         └── wider_resnet.py
├── datasets/
│    ├── __init__.py               # Exports SegmentationDataset, SimpleImageDataset, convert_label
│    └── dataset.py
├── utils.py                      # Utility functions (configuration, logging, seeding, etc.)
├── Hopfield_PEBAL.py             # Contains the HopfieldPEBALSegmentation model and inference function.
├── trainer.py                    # Training and validation loops, memory updating functions.
├── main.py                       # Main training script.
└── README.md                     # This file.


Installation

Prerequisites
	•	Python 3.9 or higher
	•	CUDA-enabled GPU (recommended)
	•	PyTorch 2.5.1 with GPU support
	•	Other required packages (install via pip or conda)


Installation Steps
	1.	Clone the repository:

git clone https://github.com/YourUsername/hopfield-pebal-project.git
cd hopfield-pebal-project


Create and activate a conda environment (recommended):

conda create -n hop-pebal python=3.9
conda activate hop-pebal

Install PyTorch with CUDA support:
(Example for CUDA 11.7; adjust based on your GPU and CUDA version.)

conda install pytorch torchvision torchaudio cudatoolkit=11.7 -c pytorch -c nvidia

	Install other dependencies:

pip install tqdm numpy


