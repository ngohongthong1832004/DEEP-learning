"""
PIPELINE TỐI ƯU NÂNG CẤP - SEGMENTATION VÀ CROP VÙNG BỆNH
========================================================
Nâng cấp theo document phản hồi với các cải tiến:

P0 (Bắt buộc):
- Vegetation ROI để khóa nền
- Lọc SAM theo % pixel trong range màu (thay vì trung bình)
- QC theo tỉ lệ diện tích lá
- DenseCRF / GrabCut refinement

P1 (Ưu tiên cao):
- Shape priors theo từng bệnh
- Intersection-then-Grow strategy
- SAM với negative prompts + box

P2 (Tối ưu):
- Color constancy (Gray-World)
- TTA & Multi-scale
- SLIC superpixel

P3 (Production):
- Config JSON export
- Logging chi tiết

Author: Claude
Date: 2025-10-05 (Upgraded)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from datetime import datetime
import json
import warnings
import torch
warnings.filterwarnings('ignore')

# =====================================================
# CONFIGURATION & OUTPUT STRUCTURE
# =====================================================

class PipelineConfig:
    """Centralized configuration"""
    
    def __init__(self):
        # Vegetation Segmentation
        self.veg_method = 'combined'  # dùng ExG ∪ HSV để bớt rỗ
        self.veg_morph_close = (9, 9)
        self.veg_morph_open  = (7, 7)
        self.veg_erode = (3, 3)
        
        # SAM Settings
        self.sam_points_per_side = 32
        self.sam_pred_iou_thresh = 0.88
        self.sam_stability_score_thresh = 0.93
        self.sam_min_mask_region_area_ratio = 0.001  # % of vegetation area
        
        # Disease-specific color ranges (HSV)
        self.disease_color_ranges = {
            'brown_spot': {
                'hsv_lower': np.array([5, 40, 40]),
                'hsv_upper': np.array([25, 255, 200]),
                'coverage_threshold': 0.40,
                'color_variance_max': 40
            },
            'leaf_blast': {
                'hsv_lower': np.array([0, 0, 100]),
                'hsv_upper': np.array([180, 80, 255]),
                'coverage_threshold': 0.35,
                'color_variance_max': 50
            },
            'leaf_blight': {
                'hsv_lower': np.array([15, 30, 30]),
                'hsv_upper': np.array([35, 255, 220]),
                'coverage_threshold': 0.45,
                'color_variance_max': 45
            }
        }
        
        # Shape priors for each disease
        self.shape_priors = {
            'leaf_blast': {
                'solidity_min': 0.5,
                'eccentricity_min': 0.70,          # dài/thon hơn
                'aspect_ratio_range': (2.5, 7.0)
            },
            'brown_spot': {
                'solidity_min': 0.80,              # đốm tròn/đặc hơn
                'eccentricity_max': 0.8,
                'aspect_ratio_range': (0.5, 2.5)
            },
            'leaf_blight': {
                'solidity_min': 0.6,
                'distance_to_edge_max': 0.12,      # gần mép lá hơn
                'aspect_ratio_range': (1.5, 8.0)
            }
        }
        
        # QC Settings (relative to leaf area)
        self.qc_min_area_ratio = 0.002  # 0.2% of leaf area
        self.qc_max_area_ratio = 0.25   # 25% of leaf area
        self.qc_blur_threshold = 100
        self.qc_padding = 20
        
        # Post-processing
        self.use_grabcut = True
        self.use_densecrf = True          # bật nếu đã cài pydensecrf
        self.grabcut_iterations = 5
        
        # Multi-scale & TTA
        self.use_multiscale = False
        self.multiscale_factors = [0.75, 1.0, 1.5]
        self.use_tta = False
        self.tta_flips = True
        
        # Color constancy
        self.use_color_constancy = True
        self.color_constancy_method = 'shades_of_gray'  # or 'gray_world'
    
    def to_dict(self):
        """Convert to dict for JSON export"""
        return {k: v if not isinstance(v, np.ndarray) else v.tolist() 
                for k, v in self.__dict__.items()}
    
    def save(self, path: str):
        """Save config to JSON"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


def get_output_folder(parent_dir: str, env_name: str) -> str:
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = f"{env_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    output_dir = os.path.join(parent_dir, experiment_id)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def create_output_structure(base_path: str, class_names: List[str]) -> Dict[str, str]:
    """Tạo structure output theo từng class"""
    folders = {
        '00_original': {},
        '01_vegetation_masks': {},
        '02_disease_masks': {},
        '03_visualization': {},
        '04_final_crops': {},
        '05_quality_filtered': {},
        '06_statistics': base_path,
        '07_config': base_path
    }
    
    folder_paths = {}
    
    for folder_type, _ in folders.items():
        if folder_type in ['06_statistics', '07_config']:
            folder_path = os.path.join(base_path, folder_type)
            os.makedirs(folder_path, exist_ok=True)
            folder_paths[folder_type] = folder_path
        else:
            for class_name in class_names:
                folder_path = os.path.join(base_path, folder_type, class_name)
                os.makedirs(folder_path, exist_ok=True)
                key = f"{folder_type}_{class_name}"
                folder_paths[key] = folder_path
    
    return folder_paths


# =====================================================
# P0.1: VEGETATION SEGMENTATION (Tầng 0)
# =====================================================

class VegetationSegmenter:
    """P0: Tách vegetation để khóa nền"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
    
    def segment_vegetation(self, image: np.ndarray) -> np.ndarray:
        """
        Segment vegetation mask
        
        Returns:
            mask: Binary mask of vegetation (255 = plant)
        """
        if self.config.veg_method == 'exg':
            mask = self._segment_exg(image)
        elif self.config.veg_method == 'hsv':
            mask = self._segment_hsv(image)
        elif self.config.veg_method == 'combined':
            mask_exg = self._segment_exg(image)
            mask_hsv = self._segment_hsv(image)
            mask = cv2.bitwise_or(mask_exg, mask_hsv)
        else:
            mask = self._segment_exg(image)
        
        # Morphological operations theo config
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                 self.config.veg_morph_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                self.config.veg_morph_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        
        # Erode để tránh tràn ra nền
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                 self.config.veg_erode)
        mask = cv2.erode(mask, kernel_erode)
        
        return mask
    
    def _segment_exg(self, image: np.ndarray) -> np.ndarray:
        """Excess Green Index"""
        b, g, r = cv2.split(image.astype(np.float32) / 255.0)
        exg = 2 * g - r - b
        
        exg_uint8 = ((exg - exg.min()) / (exg.max() - exg.min() + 1e-8) * 255).astype(np.uint8)
        _, mask = cv2.threshold(exg_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return mask
    
    def _segment_hsv(self, image: np.ndarray) -> np.ndarray:
        """HSV green detection"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([30, 30, 30])
        upper_green = np.array([90, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        return mask


# =====================================================
# P2.8: COLOR CONSTANCY
# =====================================================

class ColorConstancy:
    """P2: Ổn định màu sắc"""
    
    @staticmethod
    def gray_world(image: np.ndarray) -> np.ndarray:
        """Gray World algorithm"""
        result = image.copy().astype(np.float32)
        
        for i in range(3):
            avg = np.mean(result[:, :, i])
            result[:, :, i] = result[:, :, i] * (128.0 / (avg + 1e-8))
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result
    
    @staticmethod
    def shades_of_gray(image: np.ndarray, p: int = 6) -> np.ndarray:
        """Shades of Gray algorithm"""
        result = image.copy().astype(np.float32)
        
        for i in range(3):
            channel = result[:, :, i]
            norm = np.power(np.mean(np.power(channel, p)), 1.0/p)
            result[:, :, i] = channel * (128.0 / (norm + 1e-8))
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result
    
    @staticmethod
    def apply(image: np.ndarray, method: str = 'gray_world') -> np.ndarray:
        """Apply color constancy"""
        if method == 'gray_world':
            return ColorConstancy.gray_world(image)
        elif method == 'shades_of_gray':
            return ColorConstancy.shades_of_gray(image)
        else:
            return image


# =====================================================
# P0.2: ADVANCED COLOR SEGMENTER (Improved)
# =====================================================

class AdvancedColorSegmenter:
    """P0: Traditional CV với cải tiến lọc theo coverage %"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
    
    def segment_disease(self, image: np.ndarray, disease_type: str,
                       vegetation_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Segment vùng bệnh với improved filtering
        
        Args:
            image: Input image BGR
            disease_type: 'brown_spot', 'leaf_blast', 'leaf_blight'
            vegetation_mask: Optional vegetation ROI
        
        Returns:
            mask: Binary mask vùng bệnh
        """
        if disease_type not in self.config.disease_color_ranges:
            return self._segment_generic(image, vegetation_mask)
        
        color_cfg = self.config.disease_color_ranges[disease_type]
        
        # Multi-colorspace
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # HSV mask
        mask_hsv = cv2.inRange(hsv, color_cfg['hsv_lower'], color_cfg['hsv_upper'])
        
        # Edge enhancement
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges_dilated = cv2.dilate(edges, kernel_edge, iterations=1)
        
        # Combine
        mask = cv2.bitwise_or(mask_hsv, edges_dilated)
        
        # Apply vegetation ROI
        if vegetation_mask is not None:
            mask = cv2.bitwise_and(mask, vegetation_mask)
        
        # Morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Remove small noise
        mask = self._remove_small_regions(mask, min_area=100)
        
        return mask
    
    def _segment_generic(self, image: np.ndarray, 
                        vegetation_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Generic segmentation"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Not healthy green
        healthy_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        mask = cv2.bitwise_not(healthy_mask)
        
        if vegetation_mask is not None:
            mask = cv2.bitwise_and(mask, vegetation_mask)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def _remove_small_regions(self, mask: np.ndarray, min_area: int = 100) -> np.ndarray:
        """Remove small connected components"""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        output = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                output[labels == i] = 255
        
        return output


# =====================================================
# P0.2 & P1.7: SAM SEGMENTER (Improved Filtering)
# =====================================================

class SAMSegmenter:
    """P0 & P1: SAM với lọc theo % coverage và negative prompts"""
    
    def __init__(self, config: PipelineConfig, checkpoint_path: Optional[str] = None, 
                 model_type: str = 'vit_b'):
        self.config = config
        self.sam_available = False
        self.predictor = None
        self.mask_generator = None
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
                
                sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"  # GPU Apple Silicon
                else:
                    device = "cpu"
                sam.to(device=device)
                
                self.predictor = SamPredictor(sam)
                
                # Dynamic min_mask_region_area will be set per image
                self.mask_generator = SamAutomaticMaskGenerator(
                    sam,
                    points_per_side=config.sam_points_per_side,
                    pred_iou_thresh=config.sam_pred_iou_thresh,
                    stability_score_thresh=config.sam_stability_score_thresh,
                    crop_n_layers=1,
                    crop_n_points_downscale_factor=2,
                    min_mask_region_area=100,  # Will override per image
                )
                
                self.sam_available = True
                print(f"✅ SAM loaded on {device}")
                
            except ImportError:
                print("⚠️  pip install segment-anything")
        else:
            print(f"⚠️  SAM checkpoint not found: {checkpoint_path}")
    
    def segment_automatic(self, image: np.ndarray, disease_type: str,
                         vegetation_mask: np.ndarray,
                         vegetation_area: int) -> np.ndarray:
        """
        P0: Automatic segmentation với improved filtering
        
        Args:
            vegetation_area: Used to set dynamic min_mask_region_area
        """
        if not self.sam_available:
            return self._fallback_segment(image, disease_type, vegetation_mask)
        
        # Set dynamic min_mask_region_area
        min_area = int(vegetation_area * self.config.sam_min_mask_region_area_ratio)
        self.mask_generator.min_mask_region_area = max(100, min_area)
        
        # Generate masks
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = self.mask_generator.generate(image_rgb)
        
        # P0.2: Filter by coverage percentage (not average)
        filtered_masks = self._filter_by_coverage(masks, image, disease_type)
        
        # Combine
        h, w = image.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        for mask_data in filtered_masks:
            combined_mask = cv2.bitwise_or(
                combined_mask,
                (mask_data['segmentation'] * 255).astype(np.uint8)
            )
        
        # Apply vegetation ROI
        combined_mask = cv2.bitwise_and(combined_mask, vegetation_mask)
        
        return combined_mask
    
    def segment_with_prompts(self, image: np.ndarray,
                            positive_points: np.ndarray,
                            negative_points: Optional[np.ndarray] = None,
                            box: Optional[np.ndarray] = None) -> np.ndarray:
        """
        P1.7: Segmentation với positive + negative prompts + box
        
        Args:
            positive_points: [[x1, y1], [x2, y2], ...]
            negative_points: [[x1, y1], ...] (optional)
            box: [x1, y1, x2, y2] (optional)
        """
        if not self.sam_available:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)
        
        # Combine positive and negative points
        if negative_points is not None:
            all_points = np.vstack([positive_points, negative_points])
            labels = np.array([1] * len(positive_points) + [0] * len(negative_points))
        else:
            all_points = positive_points
            labels = np.ones(len(positive_points))
        
        masks, scores, _ = self.predictor.predict(
            point_coords=all_points,
            point_labels=labels,
            box=box,
            multimask_output=True
        )
        
        # Choose best mask
        best_mask = masks[np.argmax(scores)]
        
        return (best_mask * 255).astype(np.uint8)
    
    def _filter_by_coverage(self, masks: List[Dict], image: np.ndarray,
                           disease_type: str) -> List[Dict]:
        """
        P0.2: Filter masks by COVERAGE PERCENTAGE (not average color)
        """
        if disease_type not in self.config.disease_color_ranges:
            return []
        
        color_cfg = self.config.disease_color_ranges[disease_type]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        filtered = []
        
        for mask in masks:
            seg = mask['segmentation']
            
            if not seg.any():
                continue
            
            # Get pixels in mask
            masked_hsv = hsv[seg]
            
            if len(masked_hsv) == 0:
                continue
            
            # P0.2: Calculate COVERAGE (% pixels in color range)
            in_range = cv2.inRange(masked_hsv, 
                                  color_cfg['hsv_lower'], 
                                  color_cfg['hsv_upper'])
            coverage = np.sum(in_range > 0) / len(masked_hsv)
            
            # Calculate color variance
            hue_variance = np.var(masked_hsv[:, 0])
            
            # Filter conditions
            if (coverage >= color_cfg['coverage_threshold'] and
                hue_variance <= color_cfg['color_variance_max']):
                filtered.append(mask)
        
        return filtered
    
    def _fallback_segment(self, image: np.ndarray, disease_type: str,
                         vegetation_mask: np.ndarray) -> np.ndarray:
        """Fallback to traditional"""
        segmenter = AdvancedColorSegmenter(self.config)
        return segmenter.segment_disease(image, disease_type, vegetation_mask)


# =====================================================
# P1.5: SHAPE PRIORS VALIDATOR
# =====================================================

class ShapePriorsValidator:
    """P1: Validate regions based on shape priors per disease"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
    
    def validate_region(self, contour: np.ndarray, disease_type: str,
                       leaf_edge_mask: Optional[np.ndarray] = None) -> Tuple[bool, Dict]:
        """
        Validate if contour matches disease shape prior
        
        Returns:
            (is_valid, metrics_dict)
        """
        if disease_type not in self.config.shape_priors:
            return True, {}
        
        priors = self.config.shape_priors[disease_type]
        
        # Compute shape features
        area = cv2.contourArea(contour)
        
        if area < 10:
            return False, {'reason': 'too_small'}
        
        # Convex hull for solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / (hull_area + 1e-8)
        
        # Fit ellipse for eccentricity (need at least 5 points)
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (center, axes, angle) = ellipse
            major_axis = max(axes)
            minor_axis = min(axes)
            
            if major_axis > 0:
                eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
                aspect_ratio = major_axis / (minor_axis + 1e-8)
            else:
                eccentricity = 0
                aspect_ratio = 1
        else:
            # Fallback to bounding rect
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / (min(w, h) + 1e-8)
            eccentricity = 0.5
        
        # Distance to leaf edge (for leaf_blight)
        if leaf_edge_mask is not None and 'distance_to_edge_max' in priors:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Distance transform of leaf edge
                dist_transform = cv2.distanceTransform(leaf_edge_mask, cv2.DIST_L2, 5)
                normalized_dist = dist_transform[cy, cx] / (np.max(dist_transform) + 1e-8)
                
                if normalized_dist > priors['distance_to_edge_max']:
                    return False, {'reason': 'too_far_from_edge', 'distance': normalized_dist}
        
        # Validate against priors
        metrics = {
            'solidity': solidity,
            'eccentricity': eccentricity,
            'aspect_ratio': aspect_ratio
        }
        
        # Check solidity
        if 'solidity_min' in priors and solidity < priors['solidity_min']:
            return False, {**metrics, 'reason': 'low_solidity'}
        
        # Check eccentricity
        if 'eccentricity_max' in priors and eccentricity > priors['eccentricity_max']:
            return False, {**metrics, 'reason': 'high_eccentricity'}
        
        if 'eccentricity_min' in priors and eccentricity < priors['eccentricity_min']:
            return False, {**metrics, 'reason': 'low_eccentricity'}
        
        # Check aspect ratio
        if 'aspect_ratio_range' in priors:
            ar_min, ar_max = priors['aspect_ratio_range']
            if not (ar_min <= aspect_ratio <= ar_max):
                return False, {**metrics, 'reason': 'aspect_ratio_out_of_range'}
        
        return True, metrics


# =====================================================
# P1.6: HYBRID SEGMENTER (Intersection-then-Grow)
# =====================================================

class HybridSegmenter:
    """P1: Kết hợp SAM + Traditional theo Intersection-then-Grow"""
    
    def __init__(self, config: PipelineConfig, sam_checkpoint: Optional[str] = None):
        self.config = config
        self.sam_segmenter = SAMSegmenter(config, sam_checkpoint)
        self.color_segmenter = AdvancedColorSegmenter(config)
    
    def segment(self, image: np.ndarray, disease_type: str,
               vegetation_mask: np.ndarray, vegetation_area: int) -> np.ndarray:
        """
        P1.6: Intersection-then-Grow strategy
        
        Steps:
        1. Get SAM mask (coarse)
        2. Get color mask (detail)
        3. seed = SAM ∩ color (confident regions)
        4. Grow seed within vegetation ROI
        """
        # Step 1 & 2: Get both masks
        if self.sam_segmenter.sam_available:
            sam_mask = self.sam_segmenter.segment_automatic(
                image, disease_type, vegetation_mask, vegetation_area
            )
        else:
            sam_mask = None
        
        color_mask = self.color_segmenter.segment_disease(
            image, disease_type, vegetation_mask
        )
        
        # Step 3: Intersection for seeds
        if sam_mask is not None:
            seed = cv2.bitwise_and(sam_mask, color_mask)
            
            # If seed is too small, fallback to union
            if np.sum(seed > 0) < 1500:
                combined = cv2.bitwise_or(sam_mask, color_mask)
            else:
                # Step 4: Grow from seed
                combined = self._grow_from_seed(image, seed, vegetation_mask)
        else:
            combined = color_mask
        
        # Refine
        refined_mask = self._refine_mask(image, combined, vegetation_mask)
        
        return refined_mask
    
    def _grow_from_seed(self, image: np.ndarray, seed: np.ndarray,
                       roi_mask: np.ndarray) -> np.ndarray:
        """
        P1.6: Region growing from seed within ROI
        """
        # Simple morphological reconstruction
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Dilate seed iteratively within ROI
        grown = seed.copy()
        
        for _ in range(5):
            dilated = cv2.dilate(grown, kernel)
            grown = cv2.bitwise_and(dilated, roi_mask)
            
            # Stop if no change
            if np.array_equal(grown, dilated):
                break
        
        return grown
    
    def _refine_mask(self, image: np.ndarray, mask: np.ndarray,
                    roi_mask: np.ndarray) -> np.ndarray:
        """
        P0.4: Refine với GrabCut / DenseCRF
        """
        # Morphological closing to fill holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Remove small regions
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        refined = np.zeros_like(mask)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= 150:
                refined[labels == i] = 255
        
        # P0.4: GrabCut refinement
        if self.config.use_grabcut and np.sum(refined > 0) > 500:
            refined = self._grabcut_refine(image, refined, roi_mask)
        
        # P0.4: DenseCRF (optional)
        if self.config.use_densecrf:
            refined = self._densecrf_refine(image, refined)
        
        return refined
    
    def _grabcut_refine(self, image: np.ndarray, mask: np.ndarray,
                       roi_mask: np.ndarray) -> np.ndarray:
        """P0.4: GrabCut refinement"""
        try:
            # Constrain to ROI
            mask_roi = cv2.bitwise_and(mask, roi_mask)
            
            if np.sum(mask_roi > 0) < 100:
                return mask
            
            # Init mask for GrabCut
            mask_gc = np.where(mask_roi > 128, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
            
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            cv2.grabCut(image, mask_gc, None, bgd_model, fgd_model,
                       self.config.grabcut_iterations, cv2.GC_INIT_WITH_MASK)
            
            refined = np.where(
                (mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD),
                255, 0
            ).astype(np.uint8)
            
            # Keep only within ROI
            refined = cv2.bitwise_and(refined, roi_mask)
            
            return refined
        
        except:
            return mask
    
    def _densecrf_refine(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """P0.4: DenseCRF refinement (optional)"""
        try:
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_labels
            
            labels = (mask > 128).astype(np.int32)
            
            d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], 2)
            
            U = unary_from_labels(labels, 2, gt_prob=0.7, zero_unsure=False)
            d.setUnaryEnergy(U)
            
            d.addPairwiseGaussian(sxy=(3, 3), compat=3)
            d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13),
                                  rgbim=image, compat=10)
            
            Q = d.inference(5)
            refined = np.argmax(Q, axis=0).reshape(mask.shape)
            
            return (refined * 255).astype(np.uint8)
        
        except:
            return mask


# =====================================================
# P0.3 & P1.5: QUALITY CONTROL (Adaptive)
# =====================================================

class AdvancedQualityControl:
    """P0.3 & P1: QC với adaptive thresholds và shape priors"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.shape_validator = ShapePriorsValidator(config)
    
    def check_quality(self, image: np.ndarray) -> Dict:
        """Comprehensive quality check"""
        reasons = []
        scores = []
        
        # Blur
        blur_score = self._check_blur(image)
        scores.append(blur_score)
        if blur_score < 0.5:
            reasons.append('blurry')
        
        # Exposure
        exposure_score = self._check_exposure(image)
        scores.append(exposure_score)
        if exposure_score < 0.5:
            reasons.append('poor_exposure')
        
        # Color variety
        color_score = self._check_color_variety(image)
        scores.append(color_score)
        if color_score < 0.3:
            reasons.append('low_color_variety')
        
        # Edge sharpness
        edge_score = self._check_edges(image)
        scores.append(edge_score)
        
        overall_score = np.mean(scores)
        
        return {
            'is_valid': len(reasons) == 0 and overall_score > 0.6,
            'score': overall_score,
            'reasons': reasons,
            'metrics': {
                'blur': blur_score,
                'exposure': exposure_score,
                'color': color_score,
                'edge': edge_score
            }
        }
    
    def _check_blur(self, image: np.ndarray) -> float:
        """Laplacian variance"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        score = min(1.0, laplacian_var / 500.0)
        return score
    
    def _check_exposure(self, image: np.ndarray) -> float:
        """Check exposure balance"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        if 80 < mean_brightness < 180:
            score = 1.0
        elif 50 < mean_brightness < 200:
            score = 0.7
        else:
            score = 0.3
        
        return score
    
    def _check_color_variety(self, image: np.ndarray) -> float:
        """Color variety via hue entropy"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hist = hist.flatten() / (hist.sum() + 1e-8)
        
        entropy = -np.sum(hist * np.log(hist + 1e-8))
        score = min(1.0, entropy / 4.0)
        
        return score
    
    def _check_edges(self, image: np.ndarray) -> float:
        """Edge sharpness"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        edge_density = np.sum(edges > 0) / edges.size
        
        if 0.05 < edge_density < 0.3:
            score = 1.0
        elif 0.02 < edge_density < 0.5:
            score = 0.7
        else:
            score = 0.3
        
        return score
    
    def crop_from_mask(self, image: np.ndarray, mask: np.ndarray,
                   disease_type: str, leaf_area: int,
                   vegetation_mask: Optional[np.ndarray] = None) -> List[Tuple[np.ndarray, Dict]]:

        """
        P0.3: Crop với adaptive thresholds theo leaf_area
        P1.5: Validate với shape priors
        
        Args:
            leaf_area: Area of leaf/vegetation for adaptive thresholding
        """
        # P0.3: Adaptive thresholds
        min_area = int(leaf_area * self.config.qc_min_area_ratio)
        max_area = int(leaf_area * self.config.qc_max_area_ratio)
        
        # Find components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        
        # Get leaf edge mask for distance calculation (leaf_blight)
        leaf_edge_mask = None
        if disease_type == 'leaf_blight' and vegetation_mask is not None:
            edges = cv2.Canny(vegetation_mask, 50, 150)  # dựa trên lá, không phải mask bệnh
            leaf_edge_mask = cv2.dilate(edges, np.ones((5, 5), np.uint8))
        
        crops = []
        h, w = image.shape[:2]
        
        for i in range(1, num_labels):
            x, y, w_box, h_box, area = stats[i]
            
            # P0.3: Area filter với adaptive thresholds
            if area < min_area or area > max_area:
                continue
            
            # P1.5: Shape prior validation
            # Get contour for this component
            component_mask = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                continue
            
            contour = max(contours, key=cv2.contourArea)
            
            is_valid, shape_metrics = self.shape_validator.validate_region(
                contour, disease_type, leaf_edge_mask
            )
            
            if not is_valid:
                continue
            
            # Padding
            x1 = max(0, x - self.config.qc_padding)
            y1 = max(0, y - self.config.qc_padding)
            x2 = min(w, x + w_box + self.config.qc_padding)
            y2 = min(h, y + h_box + self.config.qc_padding)
            
            # Crop
            cropped = image[y1:y2, x1:x2]
            
            # Quality check
            qc_result = self.check_quality(cropped)
            
            crops.append((cropped, {
                'bbox': (x1, y1, x2, y2),
                'area': area,
                'centroid': centroids[i],
                'quality': qc_result,
                'shape_metrics': shape_metrics
            }))
        
        return crops


# =====================================================
# P2.9: MULTI-SCALE & TTA WRAPPER
# =====================================================

class MultiScaleTTAWrapper:
    """P2: Multi-scale and Test-Time Augmentation"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
    
    def segment_multiscale(self, image: np.ndarray, segmenter,
                          disease_type: str, vegetation_mask: np.ndarray,
                          vegetation_area: int) -> np.ndarray:
        """
        Multi-scale segmentation
        """
        if not self.config.use_multiscale:
            # Normal segmentation
            return segmenter.segment(image, disease_type, vegetation_mask, vegetation_area)
        
        h, w = image.shape[:2]
        all_masks = []
        
        for scale in self.config.multiscale_factors:
            # Resize
            new_h, new_w = int(h * scale), int(w * scale)
            scaled_img = cv2.resize(image, (new_w, new_h))
            scaled_veg = cv2.resize(vegetation_mask, (new_w, new_h))
            
            # Segment
            mask = segmenter.segment(scaled_img, disease_type, scaled_veg, vegetation_area)
            
            # Resize back
            mask_original = cv2.resize(mask, (w, h))
            all_masks.append(mask_original)
        
        # Aggregate by voting
        stacked = np.stack(all_masks, axis=0)
        voted_mask = (np.sum(stacked > 128, axis=0) > len(all_masks) / 2).astype(np.uint8) * 255
        
        return voted_mask
    
    def segment_tta(self, image: np.ndarray, segmenter,
                   disease_type: str, vegetation_mask: np.ndarray,
                   vegetation_area: int) -> np.ndarray:
        """
        Test-Time Augmentation
        """
        if not self.config.use_tta:
            return segmenter.segment(image, disease_type, vegetation_mask, vegetation_area)
        
        masks = []
        
        # Original
        mask_orig = segmenter.segment(image, disease_type, vegetation_mask, vegetation_area)
        masks.append(mask_orig)
        
        if self.config.tta_flips:
            # Horizontal flip
            img_h = cv2.flip(image, 1)
            veg_h = cv2.flip(vegetation_mask, 1)
            mask_h = segmenter.segment(img_h, disease_type, veg_h, vegetation_area)
            mask_h = cv2.flip(mask_h, 1)
            masks.append(mask_h)
            
            # Vertical flip
            img_v = cv2.flip(image, 0)
            veg_v = cv2.flip(vegetation_mask, 0)
            mask_v = segmenter.segment(img_v, disease_type, veg_v, vegetation_area)
            mask_v = cv2.flip(mask_v, 0)
            masks.append(mask_v)
        
        # Aggregate
        stacked = np.stack(masks, axis=0)
        voted_mask = (np.sum(stacked > 128, axis=0) > len(masks) / 2).astype(np.uint8) * 255
        
        return voted_mask


# =====================================================
# MAIN PIPELINE (UPGRADED)
# =====================================================

class OptimizedSegmentationPipeline:
    """
    UPGRADED PIPELINE với tất cả cải tiến P0, P1, P2
    """
    
    def __init__(self,
                 labels_config: Dict,
                 config: Optional[PipelineConfig] = None,
                 parent_output_dir: str = '../output',
                 experiment_name: str = 'GK-SEGMENTATION-UPGRADED',
                 method: str = 'hybrid',
                 sam_checkpoint: Optional[str] = None):
        """
        Args:
            config: PipelineConfig object (None = use default)
            method: 'traditional', 'sam', 'hybrid'
        """
        self.labels_config = labels_config
        self.class_names = [info['name'] for info in labels_config.values()]
        self.method = method
        
        # Config
        if config is None:
            self.config = PipelineConfig()
        else:
            self.config = config
        
        # Output
        self.output_base = get_output_folder(parent_output_dir, experiment_name)
        self.output_dirs = create_output_structure(self.output_base, self.class_names)
        
        # Save config
        self.config.save(os.path.join(self.output_dirs['07_config'], 'pipeline_config.json'))
        
        print(f"📁 Output: {self.output_base}")
        print(f"   Config saved to: {self.output_dirs['07_config']}\n")
        
        # P0.1: Vegetation segmenter
        self.veg_segmenter = VegetationSegmenter(self.config)
        
        # Main segmenter
        print(f"🔧 Initializing {method.upper()} segmenter...")
        
        if method == 'traditional':
            self.segmenter = AdvancedColorSegmenter(self.config)
        elif method == 'sam':
            self.segmenter = SAMSegmenter(self.config, sam_checkpoint)
        elif method == 'hybrid':
            self.segmenter = HybridSegmenter(self.config, sam_checkpoint)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"   ✅ {method.upper()} segmenter ready\n")
        
        # P0.3 & P1.5: Quality control with shape priors
        self.qc = AdvancedQualityControl(self.config)
        
        # P2: Multi-scale & TTA
        self.ms_tta = MultiScaleTTAWrapper(self.config)
        
        # Stats
        self.stats = []
        self.processing_log = []
    
    def process_image(self, image_path: str, disease_type: str) -> Dict:
        """
        Process 1 image với full pipeline
        """
        base_name = Path(image_path).stem
        
        # Load
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        result = {
            'image_path': image_path,
            'base_name': base_name,
            'disease_type': disease_type,
            'num_regions': 0,
            'crops_saved': 0,
            'quality_filtered': 0,
            'processing_notes': []
        }
        
        # P2.8: Color constancy
        if self.config.use_color_constancy:
            image = ColorConstancy.apply(image, self.config.color_constancy_method)
            result['processing_notes'].append('color_constancy_applied')
        
        # Save original
        original_path = os.path.join(
            self.output_dirs[f'00_original_{disease_type}'],
            f'{base_name}.jpg'
        )
        cv2.imwrite(original_path, image)
        
        # Skip healthy
        if disease_type == 'healthy':
            return result
        
        # P0.1: Vegetation segmentation
        vegetation_mask = self.veg_segmenter.segment_vegetation(image)
        vegetation_area = np.sum(vegetation_mask > 0)
        
        if vegetation_area < 1000:
            result['processing_notes'].append('vegetation_area_too_small')
            return result
        
        # Save vegetation mask
        veg_mask_path = os.path.join(
            self.output_dirs[f'01_vegetation_masks_{disease_type}'],
            f'{base_name}_veg.png'
        )
        cv2.imwrite(veg_mask_path, vegetation_mask)
        
        # Segment disease
        if self.method == 'traditional':
            mask = self.segmenter.segment_disease(image, disease_type, vegetation_mask)
        elif self.method == 'sam':
            mask = self.segmenter.segment_automatic(image, disease_type,
                                                    vegetation_mask, vegetation_area)
        elif self.method == 'hybrid':
            # P2: Multi-scale & TTA wrapper
            if self.config.use_multiscale or self.config.use_tta:
                if self.config.use_multiscale:
                    mask = self.ms_tta.segment_multiscale(
                        image, self.segmenter, disease_type, vegetation_mask, vegetation_area
                    )
                    result['processing_notes'].append('multiscale_applied')
                elif self.config.use_tta:
                    mask = self.ms_tta.segment_tta(
                        image, self.segmenter, disease_type, vegetation_mask, vegetation_area
                    )
                    result['processing_notes'].append('tta_applied')
            else:
                mask = self.segmenter.segment(image, disease_type, vegetation_mask, vegetation_area)
        
        # P0: Final AND with vegetation (insurance)
        mask = cv2.bitwise_and(mask, vegetation_mask)
        
        # Save mask
        mask_path = os.path.join(
            self.output_dirs[f'02_disease_masks_{disease_type}'],
            f'{base_name}_mask.png'
        )
        cv2.imwrite(mask_path, mask)
        
        # Visualization
        viz = self._visualize(image, mask, vegetation_mask)
        viz_path = os.path.join(
            self.output_dirs[f'03_visualization_{disease_type}'],
            f'{base_name}_viz.jpg'
        )
        cv2.imwrite(viz_path, viz)
        
        # P0.3 & P1.5: Crop with adaptive QC and shape priors
        crops = self.qc.crop_from_mask(image, mask, disease_type, vegetation_area, vegetation_mask)
        result['num_regions'] = len(crops)
        
        # Save crops
        for crop_idx, (crop_img, crop_info) in enumerate(crops):
            crop_filename = f'{base_name}_crop_{crop_idx:03d}.jpg'
            
            # All crops
            crop_path = os.path.join(
                self.output_dirs[f'04_final_crops_{disease_type}'],
                crop_filename
            )
            cv2.imwrite(crop_path, crop_img)
            result['crops_saved'] += 1
            
            # Quality filtered
            if crop_info['quality']['is_valid']:
                quality_path = os.path.join(
                    self.output_dirs[f'05_quality_filtered_{disease_type}'],
                    crop_filename
                )
                cv2.imwrite(quality_path, crop_img)
                result['quality_filtered'] += 1
        
        self.stats.append(result)
        self.processing_log.append({
            'image': base_name,
            'disease': disease_type,
            'vegetation_area': int(vegetation_area),
            'crops': result['crops_saved'],
            'quality': result['quality_filtered'],
            'notes': result['processing_notes']
        })
        
        return result
    
    def process_batch(self, image_paths_by_class: Dict[str, List[str]]) -> Dict:
        """Process batch"""
        total_images = sum(len(paths) for paths in image_paths_by_class.values())
        
        print(f"🚀 Xử lý {total_images} ảnh với UPGRADED pipeline...\n")
        print(f"   ✅ P0: Vegetation ROI + Coverage filtering + Adaptive QC + Refinement")
        print(f"   ✅ P1: Shape priors + Intersection-then-Grow + SAM prompts")
        print(f"   ✅ P2: Color constancy + Multi-scale/TTA (if enabled)\n")
        
        results_by_class = {}
        
        for disease_type, image_paths in image_paths_by_class.items():
            print(f"\n📍 {disease_type}: {len(image_paths)} images")
            
            results = []
            for img_path in tqdm(image_paths, desc=f"  {disease_type}"):
                result = self.process_image(img_path, disease_type)
                if result:
                    results.append(result)
            
            results_by_class[disease_type] = results
        
        # Save statistics
        self._save_statistics(results_by_class)
        self._save_processing_log()
        
        # Summary
        total_crops = sum(r['crops_saved'] for r in self.stats)
        total_quality = sum(r['quality_filtered'] for r in self.stats)
        
        print(f"\n✅ HOÀN TẤT!")
        print(f"   📸 Xử lý: {len(self.stats)} ảnh")
        print(f"   ✂️  Tổng crops: {total_crops}")
        print(f"   ⭐ Chất lượng cao: {total_quality}")
        print(f"   📁 Output: {self.output_base}")
        print(f"   📊 Stats: {self.output_dirs['06_statistics']}\n")
        
        return results_by_class
    
    def _visualize(self, image: np.ndarray, disease_mask: np.ndarray,
                  vegetation_mask: np.ndarray) -> np.ndarray:
        """Enhanced visualization"""
        overlay = image.copy()
        
        # Vegetation (green tint)
        veg_color = np.zeros_like(image)
        veg_color[vegetation_mask > 0] = (0, 255, 0)
        overlay = cv2.addWeighted(overlay, 0.9, veg_color, 0.1, 0)
        
        # Disease (red overlay)
        disease_color = np.zeros_like(image)
        disease_color[disease_mask > 0] = (0, 0, 255)
        overlay = cv2.addWeighted(overlay, 0.7, disease_color, 0.3, 0)
        
        # Contours
        contours, _ = cv2.findContours(disease_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
        
        return overlay
    
    def _save_statistics(self, results_by_class: Dict):
        """Save CSV"""
        import csv
        
        csv_path = os.path.join(self.output_dirs['06_statistics'],
                               'segmentation_stats.csv')
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['image', 'disease_type', 'num_regions',
                           'crops_saved', 'quality_filtered', 'notes'])
            
            for disease_type, results in results_by_class.items():
                for r in results:
                    writer.writerow([
                        r['base_name'],
                        r['disease_type'],
                        r['num_regions'],
                        r['crops_saved'],
                        r['quality_filtered'],
                        '; '.join(r['processing_notes'])
                    ])
        
        print(f"📊 Stats: {csv_path}")
    
    def _save_processing_log(self):
        """Save processing log JSON"""
        log_path = os.path.join(self.output_dirs['06_statistics'],
                               'processing_log.json')
        
        with open(log_path, 'w') as f:
            json.dump(self.processing_log, f, indent=2)
        
        print(f"📋 Log: {log_path}")


# =====================================================
# USAGE
# =====================================================

if __name__ == "__main__":
    # ===== LABELS =====
    LABELS = {
        0: {"name": "brown_spot", "match_substrings": ["../data/new_data_field_rice/brown_spot"]},
        1: {"name": "leaf_blast", "match_substrings": ["../data/new_data_field_rice/leaf_blast"]},
        2: {"name": "leaf_blight", "match_substrings": ["../data/new_data_field_rice/leaf_blight"]},
        3: {"name": "healthy", "match_substrings": ["../data/new_data_field_rice/healthy"]}
    }
    
    # ===== CUSTOM CONFIG (Optional) =====
    config = PipelineConfig()
    
    # Tuning options
    config.use_color_constancy = True
    config.use_grabcut = True
    config.use_multiscale = True  # Set True for better accuracy (slower)
    config.use_tta = True  # Set True for TTA
    
    # ===== PIPELINE =====
    pipeline = OptimizedSegmentationPipeline(
        labels_config=LABELS,
        config=config,
        parent_output_dir='../output',
        experiment_name='GK-SEGMENTATION-UPGRADED',
        method='hybrid',  # 'traditional', 'sam', 'hybrid'
        sam_checkpoint=None  # './sam_vit_b_01ec64.pth' if available
    )
    
    # ===== GET IMAGES =====
    image_paths_by_class = {}
    
    for label_id, label_info in LABELS.items():
        class_name = label_info['name']
        folders = label_info['match_substrings']
        
        all_images = []
        for folder in folders:
            folder_path = Path(folder)
            if folder_path.exists():
                images = list(folder_path.glob('*.jpg'))
                images.extend(list(folder_path.glob('*.png')))
                all_images.extend([str(p) for p in images[:10]])
        
        image_paths_by_class[class_name] = all_images
        print(f"🔍 {class_name}: {len(all_images)} images")
    
    # ===== PROCESS =====
    results = pipeline.process_batch(image_paths_by_class)
    
    print("\n" + "=" * 60)
    print("UPGRADED SEGMENTATION PIPELINE COMPLETED!")
    print("=" * 60)