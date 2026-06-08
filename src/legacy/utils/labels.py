# === File: utils/labels.py ===

import torch

# --- Canonical Label Definitions ---
# This ensures all parts of your project use the same values.
ID_CLASSES = list(range(19))
OOD_CLASS = 19
VOID_CLASS = 255

class LabelRemapper:
    """
    A utility to handle different raw label schemes and convert them to the
    project's single, canonical format.
    """
    @staticmethod
    def to_canonical(labels_raw: torch.Tensor, dataset_type: str) -> (torch.Tensor, torch.Tensor):
        """
        Remaps raw labels from a given dataset to the project's canonical format.

        Args:
            labels_raw (torch.Tensor): The raw label tensor from the dataloader.
            dataset_type (str): The source of the labels ('fishyscapes' or 'cityscapes_coco_mix').

        Returns:
            A tuple containing:
            - canonical_labels (torch.Tensor): Labels in the unified format (0-18=ID, 19=OOD, 255=Void).
            - valid_mask (torch.Tensor): A boolean mask of pixels to be included in processing.
        """
        labels = labels_raw.clone()

        if dataset_type == 'fishyscapes':
            # Fishyscapes native labels: 0=ID, 1=OOD, 2=Void
            valid_mask = (labels_raw < 2)
            
            # Remap OOD from 1 to the canonical 19
            labels[labels_raw == 1] = OOD_CLASS
            # Remap void from 2 to the canonical 255
            labels[labels_raw == 2] = VOID_CLASS
            
            return labels, valid_mask
        
        elif dataset_type == 'cityscapes_coco_mix':
            # Training mix native labels: 0-18=ID, 254=OOD, 255=Void
            valid_mask = (labels_raw != 255)

            # Remap the OOD label from 254 to the canonical 19
            labels[labels_raw == 254] = OOD_CLASS
            
            # Any other potential out-of-range labels that are not void are also mapped to OOD
            general_ood_mask = (labels_raw >= len(ID_CLASSES)) & (labels_raw != 254) & (labels_raw != 255)
            labels[general_ood_mask] = OOD_CLASS
            
            return labels, valid_mask

        else:
            raise ValueError(f"Unknown dataset_type for label remapping: {dataset_type}")