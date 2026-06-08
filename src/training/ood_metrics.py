#!/usr/bin/env python3
# File: integrated_evaluation.py

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np
import sklearn.metrics
import wandb
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, List



class PixelMetric(ABC):
    @abstractmethod
    def __call__(self, in_scores, out_scores):
        pass

class AUROCMetric(PixelMetric):
    def __call__(self, in_scores, out_scores):
        targets = torch.cat([
            torch.ones_like(in_scores, dtype=torch.int),
            torch.zeros_like(out_scores, dtype=torch.int)
        ])
        scores = torch.cat([in_scores, out_scores])
        
        targets_np = targets.cpu().numpy()
        scores_np = scores.cpu().numpy()
        
        return sklearn.metrics.roc_auc_score(targets_np, scores_np)

class FPR95Metric(PixelMetric):
    def __call__(self, in_scores, out_scores):
        targets = torch.cat([
            torch.ones_like(in_scores, dtype=torch.int),
            torch.zeros_like(out_scores, dtype=torch.int)
        ])
        scores = torch.cat([in_scores, out_scores])
        
        targets_np = targets.cpu().numpy()
        scores_np = scores.cpu().numpy()
        
        return self._fpr_at_tpr(targets_np, scores_np, tpr_level=0.95)
    
    def _fpr_at_tpr(self, y_true, y_score, tpr_level=0.95):
        y_true = (y_true == 1)
        
        desc_indices = np.argsort(y_score)[::-1]
        y_score = y_score[desc_indices]
        y_true = y_true[desc_indices]
        
        distinct_indices = np.where(np.diff(y_score))[0]
        threshold_indices = np.r_[distinct_indices, y_true.size - 1]
        
        tps = np.cumsum(y_true)[threshold_indices]
        fps = 1 + threshold_indices - tps
        
        tpr = tps / tps[-1] if tps[-1] > 0 else np.zeros_like(tps)
        
        if len(tpr) == 0 or tpr[-1] == 0:
            return 1.0
        
        cutoff = np.argmin(np.abs(tpr - tpr_level))
        n_negatives = np.sum(~y_true)
        
        if n_negatives == 0:
            return 0.0
        
        return fps[cutoff] / n_negatives

class AUPRSMetric(PixelMetric):
    def __call__(self, in_scores, out_scores):
        targets = torch.cat([
            torch.ones_like(in_scores, dtype=torch.int),
            torch.zeros_like(out_scores, dtype=torch.int)
        ])
        scores = torch.cat([in_scores, out_scores])
        
        targets_np = targets.cpu().numpy()
        scores_np = scores.cpu().numpy()
        
        return sklearn.metrics.average_precision_score(targets_np, scores_np)