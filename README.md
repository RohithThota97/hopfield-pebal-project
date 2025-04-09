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

git clone 
cd hopfield-pebal-project


Create and activate a conda environment (recommended):

conda create -n hop-pebal python=3.9
conda activate hop-pebal

Install PyTorch with CUDA support:
(Example for CUDA 11.7; adjust based on your GPU and CUDA version.)

conda install pytorch torchvision torchaudio cudatoolkit=11.7 -c pytorch -c nvidia

	Install other dependencies:

pip install tqdm numpy

Dataset Setup

This project follows the dataset conventions as described in the original PEBAL repository. For proper evaluation and training, prepare the datasets as follows:

Cityscapes (Inlier Data)
	•	Images & Annotations:
Please download the Cityscapes dataset from the official Cityscapes website.
	•	Use the “fine” annotations for training and validation.
	•	The repository’s structure assumes that the Cityscapes images and annotations are organized into separate folders (one for images and one for labels).
	•	The annotation files should include the *_gtFine_labelIds.png images, which are required for training the segmentation model.

COCO (Auxiliary Outlier Data)
	•	Auxiliary Images:
The auxiliary outlier data comes from the COCO dataset. Please refer to the official COCO website and download the training images (e.g., from the 2017 release).
	•	In our experiments (and as in the original PEBAL repository), the auxiliary dataset is used without annotations.
	•	The images should be extracted into a directory that your dataset loader can read.

Cityscapes/
├── images/
│    ├── train/
│    ├── val/
│    └── test/  (if available)
└── annotations/
     ├── train/       (contains gtFine_labelIds)
     ├── val/         (contains gtFine_labelIds)
     └── test/        (if available)

COCO/
├── train2017/        # All training images used as auxiliary outlier data.
└── annotations/      # (Optional) COCO annotation files if needed.

Training

To train the model, run:

python main.py 

Inference

You can use the inference() function provided in Hopfield_PEBAL.py to run a prediction on a single image. For examplz
python -c "from Hopfield_PEBAL import inference; import torch; \
             img = torch.randn(3, 512, 1024); \
             model = torch.load('checkpoints/best_model.pth'); \
             out = inference(model, img, torch.device('cuda')); print(out['prediction'])"

Configuration

Configuration options such as paths, batch size, learning rates, margins, and network hyperparameters are set via command-line arguments using the helper in utils.py. They have default values in the code which you can override.

Memory & Performance

If you encounter CUDA out-of-memory errors, try reducing the batch size or image resolution. Also, you can set the environment variable to allow expandable segments:
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

Acknowledgements

This project builds upon recent research in out-of-distribution detection using energy-based models and modern Hopfield networks. Many thanks to the original authors of the PEBAL and Hopfield Boosting approaches.
Contact

For questions or contributions, please open an issue or contact Rohith Thota at rohiththota79@gmail.com

