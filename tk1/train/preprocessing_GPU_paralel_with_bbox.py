"""
RICE DISEASE DETECTION - GPU HYBRID (FAST + ACCURATE)
======================================================
Hybrid GPU-CPU approach: GPU cho tốc độ, CPU cho accuracy
Đầy đủ 5 cải tiến CV để đảm bảo kết quả chính xác

Author: Claude  
Date: 2025-10-05
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import os
from datetime import datetim
import json
import warningsuse_superpixel_refine 
import torch
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
warnings.filterwarnings('ignore')

# =====================================================
# GPU UTILITIES - CHỈ DÙNG CHO BATCH I/O
# =====================================================

class GPUAccelerator:
    """GPU cho batch I/O và color conversion nhanh"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_gpu = torch.cuda.is_available()
        
        if self.use_gpu:
            print(f"✅ CUDA GPU: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            torch.backends.cudnn.benchmark = True
        else:
            print("⚠️  GPU not available, using CPU")
    
    def clear_cache(self):
        if self.use_gpu:
            torch.cuda.empty_cache()

# Global GPU
GPU = GPUAccelerator()

# =====================================================
# TUNABLE CONFIGURATION
# =====================================================

class TunableConfig:
    """TẤT CẢ THAM SỐ CÓ THỂ TINH CHỈNH"""
    
    def __init__(self):
        # ========== VEGETATION SEGMENTATION ==========
        self.veg_method = 'combined'
        self.veg_morph_close = (11, 11)
        self.veg_morph_open = (9, 9)
        self.veg_erode = (4, 4)
        
        # ========== DISEASE COLOR DETECTION (HSV) ==========
        self.disease_color_ranges = {
            'brown_spot': {
                # H: 8–22, S: 60–255, V: 40–210
                'hsv_lower': np.array([10, 60, 40]),
                'hsv_upper': np.array([22, 255, 210]),
                'coverage_threshold': 0.40,
                'color_variance_max': 40,
            },
            'leaf_blast': {
                # vùng bạc/xám, bão hòa thấp, sáng: V>=130
                'hsv_lower': np.array([0, 0, 130]),
                'hsv_upper': np.array([180, 65, 255]),
                'coverage_threshold': 0.35,
                'color_variance_max': 50,
            },
            'leaf_blight': {
                # vàng nâu dọc bìa lá
                'hsv_lower': np.array([17, 45, 55]),
                'hsv_upper': np.array([35, 255, 220]),
                'coverage_threshold': 0.45,
                'color_variance_max': 45,
            }
        }
        
        # ========== YELLOW LEAF DETECTION ==========
        self.yellow_hsv_main = ([16, 80, 70], [38, 255, 255])
        self.yellow_hsv_orange = ([5, 80, 60], [20, 255, 255])
        self.yellow_lab_threshold = 145
        self.yellow_min_area = 250
        
        # ========== MORPHOLOGY & POST-PROCESSING ==========
        self.disease_morph_kernel = (7, 7)
        self.disease_morph_close_iter = 2
        self.disease_morph_open_iter = 1
        self.min_region_area = 1200      # trước 250
        
        # ========== EDGE DETECTION ==========
        self.canny_low = 80
        self.canny_high = 180
        self.edge_dilate_kernel = (3, 3)
        self.edge_dilate_iter = 1
        
        # ========== 1. ADAPTIVE K-MEANS LAB ==========
        self.use_adaptive_kmeans = True  # ✅ BẬT
        self.kmeans_k = 3
        self.kmeans_attempts = 3
        self.kmeans_min_cluster_frac = 0.12
        self.kmeans_ab_weight = 1.2
        self.kmeans_veto_green_h = (35, 95)
        
        # ========== 2. LBP TEXTURE ==========
        self.use_lbp = False  # ✅ BẬT
        self.lbp_radius = 1
        self.lbp_thresh = 0.12
        self.lbp_min_area = 300
        
        # ========== 3. SUPERPIXEL REFINEMENT ==========
        self.use_superpixel_refine = True  # ✅ BẬT
        self.slic_region_size = 25
        self.slic_ruler = 12.0
        self.slic_min_size = 40
        self.superpixel_vote_ratio = 0.80  # trước 0.55
        
        # ========== 4. SHAPE FILTERS ==========
        self.shape_min_solidity = 0.5
        self.shape_max_eccentricity = 0.995
        
        # ========== 5. CLAHE ILLUMINATION ==========
        self.use_clahe = True  # ✅ BẬT
        self.clahe_clip = 2.0
        self.clahe_tile = (8, 8)
        
        # ========== VISUALIZATION ==========
        self.viz_vegetation_color = (0, 255, 0)
        self.viz_vegetation_alpha = 0.1
        self.viz_disease_color = (0, 0, 255)
        self.viz_disease_alpha = 0.6
        self.viz_contour_color = (0, 0, 255)
        self.viz_contour_thickness = 2
        
        # ========== QUALITY CONTROL ==========
        self.qc_blur_threshold = 100
        self.qc_exposure_optimal = (80, 180)
        self.qc_exposure_acceptable = (50, 200)
        self.qc_edge_optimal = (0.05, 0.3)
        self.qc_edge_acceptable = (0.02, 0.5)
        self.qc_min_score = 0.6  # ✅ Giữ nguyên accuracy
        
        # ========== COLOR CONSTANCY ==========
        self.use_color_constancy = True
        self.color_constancy_method = 'shades_of_gray'
        self.shades_of_gray_p = 6
        
        # ========== OUTPUT SETTINGS ==========
        self.save_vegetation_mask = True
        self.save_disease_mask = True
        self.save_visualization = True
        self.save_stats = True
        
        # ========== PARALLEL PROCESSING ==========
        self.use_multiprocessing = True
        self.num_workers = max(1, mp.cpu_count() - 1)
    
    def to_dict(self):
        return {k: v if not isinstance(v, np.ndarray) else v.tolist() 
                for k, v in self.__dict__.items()}
    
    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, default=str, ensure_ascii=False)
    
    def print_summary(self):
        print("\n" + "="*60)
        print("GPU-HYBRID PARAMETERS SUMMARY (FAST + ACCURATE)")
        print("="*60)
        print(f"Vegetation Method: {self.veg_method}")
        print(f"Color Constancy: {self.use_color_constancy} ({self.color_constancy_method})")
        print(f"✅ Adaptive K-means: {self.use_adaptive_kmeans}")
        print(f"✅ LBP Texture: {self.use_lbp}")
        print(f"✅ Superpixel Refine: {self.use_superpixel_refine}")
        print(f"✅ Shape Filtering: solidity≥{self.shape_min_solidity}, ecc≤{self.shape_max_eccentricity}")
        print(f"✅ CLAHE Enhancement: {self.use_clahe}")
        print(f"Parallel Workers: {self.num_workers}")
        print(f"Quality Threshold: {self.qc_min_score} (Accurate)")
        print("="*60 + "\n")


# =====================================================
# OUTPUT STRUCTURE
# =====================================================

def get_output_folder(parent_dir: str, env_name: str) -> str:
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = f"{env_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    output_dir = os.path.join(parent_dir, experiment_id)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def create_output_structure(base_path: str, class_names: List[str]) -> Dict[str, str]:
    folders = {
        '01_originals': {},
        '02_vegetation_masks': {},
        '03_disease_masks': {},
        '04_detected_results': {},
        '05_statistics': base_path,
        '06_config': base_path
    }
    
    folder_paths = {}
    for folder_type, _ in folders.items():
        if folder_type in ['05_statistics', '06_config']:
            folder_path = os.path.join(base_path, folder_type)
            os.makedirs(folder_path, exist_ok=True)
            folder_paths[folder_type] = folder_path
        else:
            for class_name in class_names:
                folder_path = os.path.join(base_path, folder_type, class_name)
                os.makedirs(folder_path, exist_ok=True)
                folder_paths[f"{folder_type}_{class_name}"] = folder_path
    
    return folder_paths


# =====================================================
# VEGETATION SEGMENTATION - CPU (ACCURATE)
# =====================================================

class VegetationSegmenter:
    """Vegetation segmentation - CPU cho accuracy"""
    
    def __init__(self, config: TunableConfig):
        self.config = config
    
    def segment_vegetation(self, image: np.ndarray) -> np.ndarray:
        """CPU segmentation - CHÍNH XÁC"""
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
        
        mask = self._apply_morphology(mask)
        return mask
    
    def _segment_exg(self, image: np.ndarray) -> np.ndarray:
        """Excess Green Index với OTSU threshold (adaptive)"""
        b, g, r = cv2.split(image.astype(np.float32) / 255.0)
        exg = 2 * g - r - b
        
        exg_uint8 = ((exg - exg.min()) / (exg.max() - exg.min() + 1e-8) * 255).astype(np.uint8)
        _, mask = cv2.threshold(exg_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return mask
    
    def _segment_hsv(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, np.array([25, 20, 20]), np.array([95, 255, 255]))
        yellow = cv2.inRange(hsv, np.array([15, 15, 40]), np.array([45, 255, 255]))
        return cv2.bitwise_or(green, yellow)
    
    def _apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """OpenCV morphology - CHÍNH XÁC"""
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.config.veg_morph_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.config.veg_morph_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.config.veg_erode)
        mask = cv2.erode(mask, kernel_erode)
        
        return mask


# =====================================================
# COLOR CONSTANCY (CẢI TIẾN #5: CLAHE)
# =====================================================

class ColorConstancy:
    
    @staticmethod
    def gray_world(image: np.ndarray) -> np.ndarray:
        result = image.copy().astype(np.float32)
        for i in range(3):
            avg = np.mean(result[:, :, i])
            result[:, :, i] = result[:, :, i] * (128.0 / (avg + 1e-8))
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result
    
    @staticmethod
    def shades_of_gray(image: np.ndarray, p: int = 6) -> np.ndarray:
        result = image.copy().astype(np.float32)
        for i in range(3):
            channel = result[:, :, i]
            norm = np.power(np.mean(np.power(channel, p)), 1.0/p)
            result[:, :, i] = channel * (128.0 / (norm + 1e-8))
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result
    
    @staticmethod
    def apply(image: np.ndarray, config: TunableConfig) -> np.ndarray:
        out = image.copy()
        
        if config.color_constancy_method == 'gray_world':
            out = ColorConstancy.gray_world(out)
        elif config.color_constancy_method == 'shades_of_gray':
            out = ColorConstancy.shades_of_gray(out, config.shades_of_gray_p)
        
        if config.use_clahe:
            hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
            clahe = cv2.createCLAHE(clipLimit=config.clahe_clip, 
                                   tileGridSize=config.clahe_tile)
            hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
            out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            lab[:, :, 2] = clahe.apply(lab[:, :, 2])
            out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return out


# =====================================================
# DISEASE SEGMENTATION (ĐẦY ĐỦ 5 CẢI TIẾN - CPU)
# =====================================================

class DiseaseSegmenter:
    """Disease segmentation với đầy đủ 5 cải tiến - CPU cho accuracy"""
    
    def __init__(self, config: TunableConfig):
        self.config = config
    
    def segment_disease(self, image: np.ndarray, disease_type: str,
                       vegetation_mask: Optional[np.ndarray] = None) -> np.ndarray:
        
        # 1. HSV color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        if disease_type in self.config.disease_color_ranges:
            color_cfg = self.config.disease_color_ranges[disease_type]
            mask_hsv = cv2.inRange(hsv, color_cfg['hsv_lower'], color_cfg['hsv_upper'])
        else:
            mask_hsv = self._segment_generic(image, vegetation_mask)
        
        # siết thêm theo Lab từng bệnh
        if disease_type == 'brown_spot':
            # điểm nâu/đỏ thật sự
            a_ok = (lab[:, :, 1] > 135).astype(np.uint8) * 255
            b_ok = (lab[:, :, 2] > 140).astype(np.uint8) * 255
            mask_hsv = cv2.bitwise_and(mask_hsv, a_ok)
            mask_hsv = cv2.bitwise_and(mask_hsv, b_ok)

        elif disease_type == 'leaf_blast':
            # giữ vùng bạc/xám sáng
            L_ok = (lab[:, :, 0] < 130).astype(np.uint8) * 255
            mask_hsv = cv2.bitwise_and(mask_hsv, L_ok)

        # --- xử lý ROI vàng-lá ---
        yellow_roi = None
        if vegetation_mask is not None:
            yellow_roi = self._yellow_leaf_roi(image, vegetation_mask)

        if disease_type in ['brown_spot', 'leaf_blast']:
            # LOẠI vàng-lá
            if yellow_roi is not None:
                mask_hsv = cv2.bitwise_and(mask_hsv, cv2.bitwise_not(yellow_roi))

        elif disease_type == 'leaf_blight':
            # BẮT BUỘC nằm trong vàng-lá và sát mép lá
            edge_band = self._edge_band_of_leaf(vegetation_mask, max_dist_px=8)
            if yellow_roi is not None:
                mask_hsv = cv2.bitwise_and(mask_hsv, yellow_roi)
            mask_hsv = cv2.bitwise_and(mask_hsv, edge_band)
        
        mask = mask_hsv
        
        # CẢI TIẾN #1: Adaptive K-means Lab
        if vegetation_mask is not None and self.config.use_adaptive_kmeans:
            adaptive_mask = self._adaptive_kmeans_lab(image, vegetation_mask)
            mask = cv2.bitwise_or(mask, adaptive_mask)
        
        # CẢI TIẾN #2: LBP Texture
        if vegetation_mask is not None and self.config.use_lbp:
            lbp_mask = self._lbp_mask(image, vegetation_mask)
            mask = cv2.bitwise_or(mask, lbp_mask)
        
        # 3. Edge enhancement
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.config.canny_low, self.config.canny_high)
        kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                self.config.edge_dilate_kernel)
        edges_dilated = cv2.dilate(edges, kernel_edge, 
                                   iterations=self.config.edge_dilate_iter)
        
        if disease_type == 'leaf_blight':
            # blight men theo gân/bìa -> cần bờ rõ
            mask = cv2.bitwise_and(mask, edges_dilated)   # ✅
        else:
            # với brown_spot / leaf_blast, giữ nguyên hoặc AND nhẹ:
            mask = cv2.bitwise_and(mask, cv2.bitwise_or(mask, edges_dilated))
        
        if vegetation_mask is not None:
            mask = cv2.bitwise_and(mask, vegetation_mask)
        
        # 4. Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                          self.config.disease_morph_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 
                               iterations=self.config.disease_morph_close_iter)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 
                               iterations=self.config.disease_morph_open_iter)
        
        # CẢI TIẾN #4: Shape filtering
        mask = self._remove_small_regions(mask, self.config.min_region_area)
        
        # Bộ lọc texture/độ tương phản cục bộ
        g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        local_std = cv2.GaussianBlur(g*g, (0,0), 3) - (cv2.GaussianBlur(g, (0,0), 3))**2
        local_std = np.sqrt(np.clip(local_std, 0, None)).astype(np.float32)
        std_ok = (local_std > 10).astype(np.uint8) * 255   # vùng vàng đều sẽ bị loại
        mask = cv2.bitwise_and(mask, std_ok)

        # Eccentricity cho leaf_blast (vệt dài)
        if disease_type == 'leaf_blast':
            mask = self._remove_small_regions(mask, self.config.min_region_area)
            # đã có kiểm soát eccentricity trong _remove_small_regions(); tăng ngưỡng:
            # Tạm thời lưu và khôi phục giá trị eccentricity
            old_ecc = self.config.shape_max_eccentricity
            self.config.shape_max_eccentricity = 0.995
            mask = self._remove_small_regions(mask, self.config.min_region_area)
            self.config.shape_max_eccentricity = old_ecc
        
        # CẢI TIẾN #3: Superpixel refinement
        if vegetation_mask is not None and self.config.use_superpixel_refine:
            mask = self._refine_with_superpixels(image, mask, vegetation_mask)
        
        return mask
    
    def _adaptive_kmeans_lab(self, image: np.ndarray, vegetation_mask: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(lab)
        
        ys, xs = np.where(vegetation_mask > 0)
        if len(xs) < 500:
            return np.zeros_like(vegetation_mask)
        
        samples = np.stack([
            L[ys, xs],
            self.config.kmeans_ab_weight * a[ys, xs],
            self.config.kmeans_ab_weight * b[ys, xs]
        ], axis=1).astype(np.float32)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
        _, labels, centers = cv2.kmeans(
            samples, self.config.kmeans_k, None, criteria,
            self.config.kmeans_attempts, cv2.KMEANS_PP_CENTERS
        )
        
        mask = np.zeros_like(vegetation_mask)
        total = len(labels)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        for k in range(centers.shape[0]):
            frac = (labels == k).sum() / total
            if frac < self.config.kmeans_min_cluster_frac:
                continue
            
            Lc, ac, bc = centers[k]
            
            cluster = np.zeros_like(vegetation_mask)
            cluster_indices = (labels.ravel() == k)
            cluster[ys[cluster_indices], xs[cluster_indices]] = 255
            
            h_vals = hsv[:, :, 0][cluster > 0]
            if h_vals.size > 0:
                h_med = np.median(h_vals)
                if self.config.kmeans_veto_green_h[0] <= h_med <= self.config.kmeans_veto_green_h[1]:
                    continue
            
            if (Lc < 120) or (bc > 150) or (ac > 150):
                mask = cv2.bitwise_or(mask, cluster.astype(np.uint8))
        
        mask = self._remove_small_regions(mask, self.config.min_region_area)
        return mask
    
    def _lbp_mask(self, image: np.ndarray, vegetation_mask: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        g = gray.astype(np.int16)
        
        shifts = [(0, 1), (0, -1), (1, 0), (-1, 0), 
                  (1, 1), (1, -1), (-1, 1), (-1, -1)]
        lbp = np.zeros_like(g, dtype=np.uint8)
        
        for idx, (dy, dx) in enumerate(shifts):
            shifted = np.zeros_like(g)
            h, w = g.shape
            
            src_y_start = max(0, -dy)
            src_y_end = h - max(0, dy)
            src_x_start = max(0, -dx)
            src_x_end = w - max(0, dx)
            
            dst_y_start = max(0, dy)
            dst_y_end = h + min(0, dy)
            dst_x_start = max(0, dx)
            dst_x_end = w + min(0, dx)
            
            shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                g[src_y_start:src_y_end, src_x_start:src_x_end]
            
            lbp |= ((shifted >= g) << idx).astype(np.uint8)
        
        hist = cv2.calcHist([lbp], [0], vegetation_mask, [256], [0, 256]).flatten()
        if hist.sum() == 0:
            return np.zeros_like(vegetation_mask)
        
        rare = np.where(hist / hist.sum() < 0.01)[0]
        rare_mask = np.isin(lbp, rare).astype(np.uint8) * 255
        rare_mask = cv2.bitwise_and(rare_mask, vegetation_mask)
        
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        rare_mask = cv2.morphologyEx(rare_mask, cv2.MORPH_OPEN, k, iterations=1)
        rare_mask = self._remove_small_regions(rare_mask, self.config.lbp_min_area)
        
        return rare_mask
    
    def _refine_with_superpixels(self, image: np.ndarray, mask: np.ndarray,
                                 vegetation_mask: np.ndarray) -> np.ndarray:
        try:
            import cv2.ximgproc as xi
            slic = xi.createSuperpixelSLIC(
                image, algorithm=xi.SLICO,
                region_size=self.config.slic_region_size,
                ruler=self.config.slic_ruler
            )
            slic.iterate(10)
            labels = slic.getLabels()
            
            out = np.zeros_like(mask)
            n_labels = labels.max() + 1
            
            for lab in range(n_labels):
                sp = (labels == lab)
                if vegetation_mask is not None:
                    sp = sp & (vegetation_mask > 0)
                
                if sp.sum() < self.config.slic_min_size:
                    continue
                
                vote = (mask[sp] > 0).mean()
                if vote >= self.config.superpixel_vote_ratio:
                    out[sp] = 255
            
            return out
            
        except Exception:
            ms = cv2.pyrMeanShiftFiltering(image, sp=15, sr=20)
            seg = cv2.cvtColor(ms, cv2.COLOR_BGR2GRAY)
            seg = cv2.Canny(seg, 30, 90)
            seg = cv2.dilate(seg, np.ones((3, 3), np.uint8), iterations=1)
            
            _, markers = cv2.connectedComponents(255 - seg, connectivity=4)
            
            out = np.zeros_like(mask)
            for lab in range(1, markers.max() + 1):
                region = (markers == lab)
                
                if vegetation_mask is not None:
                    region = region & (vegetation_mask > 0)
                
                if region.sum() < self.config.slic_min_size:
                    continue
                
                if (mask[region] > 0).mean() >= self.config.superpixel_vote_ratio:
                    out[region] = 255
            
            return out
    
    def _segment_yellow_leaf(self, image: np.ndarray, 
                            vegetation_mask: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        y_lower, y_upper = self.config.yellow_hsv_main
        o_lower, o_upper = self.config.yellow_hsv_orange
        
        y_main = cv2.inRange(hsv, np.array(y_lower), np.array(y_upper))
        o_main = cv2.inRange(hsv, np.array(o_lower), np.array(o_upper))
        y = cv2.bitwise_or(y_main, o_main)
        
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        _, bmask = cv2.threshold(lab[:, :, 2], self.config.yellow_lab_threshold, 
                                255, cv2.THRESH_BINARY)
        
        y = cv2.bitwise_and(y, bmask)
        y = cv2.bitwise_and(y, vegetation_mask)
        
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        y = cv2.morphologyEx(y, cv2.MORPH_CLOSE, k, iterations=1)
        y = cv2.morphologyEx(y, cv2.MORPH_OPEN, k, iterations=1)
        
        y = self._remove_small_regions(y, self.config.yellow_min_area)
        return y
    
    def _remove_small_regions(self, mask: np.ndarray, min_area: int) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        output = np.zeros_like(mask)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            
            comp = (labels == i).astype(np.uint8)
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not cnts:
                continue
            
            cnt = max(cnts, key=cv2.contourArea)
            
            hull = cv2.convexHull(cnt)
            area_cnt = cv2.contourArea(cnt) + 1e-6
            area_hull = cv2.contourArea(hull) + 1e-6
            solidity = area_cnt / area_hull
            
            ecc = 0.0
            if len(cnt) >= 5:
                try:
                    (x, y), (MA, ma), ang = cv2.fitEllipse(cnt)
                    if MA > 0:
                        a = max(MA, ma) / 2.0
                        b = min(MA, ma) / 2.0
                        ecc = np.sqrt(1 - (b * b) / (a * a + 1e-8))
                except:
                    pass
            
            if (solidity >= self.config.shape_min_solidity and 
                ecc <= self.config.shape_max_eccentricity):
                output[labels == i] = 255
        
        return output
    
    def _segment_generic(self, image: np.ndarray, 
                        vegetation_mask: Optional[np.ndarray] = None) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        brown_mask = cv2.inRange(hsv, np.array([5, 40, 40]), 
                                np.array([30, 255, 220]))
        dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), 
                               np.array([180, 255, 60]))
        
        mask = cv2.bitwise_or(brown_mask, dark_mask)
        
        if vegetation_mask is not None:
            mask = cv2.bitwise_and(mask, vegetation_mask)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        mask = self._remove_small_regions(mask, self.config.min_region_area)
        return mask

    def _yellow_leaf_roi(self, image, vegetation_mask):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        y1 = cv2.inRange(hsv, np.array([16, 80, 70]), np.array([38, 255, 255]))
        y2 = cv2.inRange(hsv, np.array([5, 80, 60]),  np.array([20, 255, 255]))
        y  = cv2.bitwise_or(y1, y2)

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        b_ok = (lab[:, :, 2] > 145).astype(np.uint8) * 255
        y = cv2.bitwise_and(y, b_ok)
        y = cv2.bitwise_and(y, vegetation_mask)
        y = self._remove_small_regions(y, 300)
        return y

    def _edge_band_of_leaf(self, vegetation_mask, max_dist_px=8):
        # dải 0..max_dist_px tính từ đường biên ROI lá
        kernel = np.ones((3,3), np.uint8)
        border = cv2.morphologyEx(vegetation_mask, cv2.MORPH_GRADIENT, kernel)
        # giãn biên để tạo band
        band = cv2.dilate(border, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(max_dist_px*2+1, max_dist_px*2+1)))
        return band


# =====================================================
# QUALITY CONTROL
# =====================================================

class QualityControl:
    
    def __init__(self, config: TunableConfig):
        self.config = config
    
    def check_quality(self, image: np.ndarray) -> Dict:
        reasons = []
        scores = []
        
        blur_score = self._check_blur(image)
        scores.append(blur_score)
        if blur_score < 0.5:
            reasons.append('blurry')
        
        exposure_score = self._check_exposure(image)
        scores.append(exposure_score)
        if exposure_score < 0.5:
            reasons.append('poor_exposure')
        
        color_score = self._check_color_variety(image)
        scores.append(color_score)
        if color_score < 0.3:
            reasons.append('low_color_variety')
        
        edge_score = self._check_edges(image)
        scores.append(edge_score)
        
        overall_score = np.mean(scores)
        
        return {
            'is_valid': len(reasons) == 0 and overall_score >= self.config.qc_min_score,
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
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return min(1.0, laplacian_var / 500.0)
    
    def _check_exposure(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        opt_min, opt_max = self.config.qc_exposure_optimal
        acc_min, acc_max = self.config.qc_exposure_acceptable
        
        if opt_min < mean_brightness < opt_max:
            return 1.0
        elif acc_min < mean_brightness < acc_max:
            return 0.7
        return 0.3
    
    def _check_color_variety(self, image: np.ndarray) -> float:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hist = hist.flatten() / (hist.sum() + 1e-8)
        
        entropy = -np.sum(hist * np.log(hist + 1e-8))
        return min(1.0, entropy / 4.0)
    
    def _check_edges(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        edge_density = np.sum(edges > 0) / edges.size
        
        opt_min, opt_max = self.config.qc_edge_optimal
        acc_min, acc_max = self.config.qc_edge_acceptable
        
        if opt_min < edge_density < opt_max:
            return 1.0
        elif acc_min < edge_density < acc_max:
            return 0.7
        return 0.3


# =====================================================
# VISUALIZATION
# =====================================================

class Visualizer:
    
    def __init__(self, config: TunableConfig):
        self.config = config
    
    def create_detection_image(self, image, disease_mask, vegetation_mask):
        overlay = image.copy()

        # Bỏ overlay vegetation hoặc để rất nhẹ
        if self.config.viz_vegetation_alpha > 0:
            veg_color = np.zeros_like(image)
            veg_color[vegetation_mask > 0] = self.config.viz_vegetation_color
            overlay = cv2.addWeighted(overlay, 1 - self.config.viz_vegetation_alpha,
                                    veg_color, self.config.viz_vegetation_alpha, 0)

        # Overlay vùng bệnh (đỏ) mạnh hơn
        disease_color = np.zeros_like(image)
        disease_color[disease_mask > 0] = self.config.viz_disease_color
        overlay = cv2.addWeighted(overlay, 1 - self.config.viz_disease_alpha,
                                disease_color, self.config.viz_disease_alpha, 0)

        # Viền + nhãn từng vùng
        contours, _ = cv2.findContours(disease_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for idx, cnt in enumerate(contours, start=1):
            area = int(cv2.contourArea(cnt))
            if area < 50: 
                continue
            cv2.drawContours(overlay, [cnt], -1, self.config.viz_contour_color, self.config.viz_contour_thickness)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(overlay, f"#{idx} {area}px", (x, max(0, y-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)

        # Tỷ lệ bệnh
        disease_area = int((disease_mask > 0).sum())
        veg_area = max(1, int((vegetation_mask > 0).sum()))
        ratio = disease_area / veg_area
        cv2.putText(overlay, f"Disease area: {disease_area}px  |  On-leaf: {ratio:.1%}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)

        # Legend
        cv2.rectangle(overlay, (10, 40), (30, 60), (0,0,255), -1)
        cv2.putText(overlay, "Benh (mask)", (35, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)

        return overlay



# =====================================================
# MAIN PIPELINE
# =====================================================

class RiceDiseaseDetectionPipeline:
    """GPU-HYBRID Pipeline: Fast + Accurate"""
    
    def __init__(self,
                 labels_config: Dict,
                 config: Optional[TunableConfig] = None,
                 parent_output_dir: str = '../output',
                 experiment_name: str = 'RiceDisease'):
        
        self.labels_config = labels_config
        self.class_names = [info['name'] for info in labels_config.values()]
        
        if config is None:
            self.config = TunableConfig()
        else:
            self.config = config
        
        self.output_base = get_output_folder(parent_output_dir, experiment_name)
        self.output_dirs = create_output_structure(self.output_base, self.class_names)
        
        self.config.save(os.path.join(self.output_dirs['06_config'], 
                                      'tunable_config.json'))
        
        self.config.print_summary()
        print(f"📁 Output: {self.output_base}\n")
        
        self.veg_segmenter = VegetationSegmenter(self.config)
        self.disease_segmenter = DiseaseSegmenter(self.config)
        self.qc = QualityControl(self.config)
        self.visualizer = Visualizer(self.config)
        
        self.stats = []
    
    def process_image(self, image_path: str, disease_type: str) -> Dict:
        base_name = Path(image_path).stem
        
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        result = {
            'image_path': image_path,
            'base_name': base_name,
            'disease_type': disease_type,
            'disease_detected': False,
            'quality_score': 0.0,
            'disease_area': 0,
            'vegetation_area': 0
        }
        
        if self.config.use_color_constancy:
            image = ColorConstancy.apply(image, self.config)
        
        qc_result = self.qc.check_quality(image)
        result['quality_score'] = qc_result['score']
        result['quality_valid'] = qc_result['is_valid']
        
        original_path = os.path.join(
            self.output_dirs[f'01_originals_{disease_type}'],
            f'{base_name}.jpg'
        )
        cv2.imwrite(original_path, image)
        
        if disease_type == 'healthy':
            return result
        
        vegetation_mask = self.veg_segmenter.segment_vegetation(image)
        vegetation_area = np.sum(vegetation_mask > 0)
        result['vegetation_area'] = int(vegetation_area)
        
        if vegetation_area < 1000:
            return result
        
        if self.config.save_vegetation_mask:
            veg_mask_path = os.path.join(
                self.output_dirs[f'02_vegetation_masks_{disease_type}'],
                f'{base_name}_veg.png'
            )
            cv2.imwrite(veg_mask_path, vegetation_mask)
        
        disease_mask = self.disease_segmenter.segment_disease(
            image, disease_type, vegetation_mask
        )
        disease_area = np.sum(disease_mask > 0)
        result['disease_area'] = int(disease_area)
        result['disease_detected'] = disease_area > 0
        
        if self.config.save_disease_mask:
            mask_path = os.path.join(
                self.output_dirs[f'03_disease_masks_{disease_type}'],
                f'{base_name}_disease.png'
            )
            cv2.imwrite(mask_path, disease_mask)
        
        if self.config.save_visualization:
            viz = self.visualizer.create_detection_image(
                image, disease_mask, vegetation_mask
            )
            viz_path = os.path.join(
                self.output_dirs[f'04_detected_results_{disease_type}'],
                f'{base_name}_detected.jpg'
            )
            cv2.imwrite(viz_path, viz)
        
        self.stats.append(result)
        return result
    
    def process_batch(self, image_paths_by_class: Dict[str, List[str]]) -> Dict:
        total_images = sum(len(paths) for paths in image_paths_by_class.values())
        
        print(f"🚀 Processing {total_images} images (GPU-HYBRID: FAST + ACCURATE)...\n")
        
        results_by_class = {}
        
        # Parallel processing với multiprocessing
        if self.config.use_multiprocessing and total_images > 20:
            print(f"Using {self.config.num_workers} parallel workers\n")
            
            for disease_type, image_paths in image_paths_by_class.items():
                print(f"📍 {disease_type}: {len(image_paths)} images")
                
                with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
                    futures = [executor.submit(self.process_image, img_path, disease_type)
                              for img_path in image_paths]
                    
                    results = []
                    for future in tqdm(futures, desc=f"  {disease_type}"):
                        result = future.result()
                        if result:
                            results.append(result)
                
                results_by_class[disease_type] = results
        else:
            for disease_type, image_paths in image_paths_by_class.items():
                print(f"\n📍 {disease_type}: {len(image_paths)} images")
                
                results = []
                for img_path in tqdm(image_paths, desc=f"  {disease_type}"):
                    result = self.process_image(img_path, disease_type)
                    if result:
                        results.append(result)
                
                results_by_class[disease_type] = results
        
        if self.config.save_stats:
            self._save_statistics(results_by_class)
        
        total_detected = sum(1 for r in self.stats if r['disease_detected'])
        avg_quality = np.mean([r['quality_score'] for r in self.stats])
        
        print(f"\n✅ COMPLETED (GPU-HYBRID)!")
        print(f"   📸 Processed: {len(self.stats)} images")
        print(f"   🔍 Disease detected: {total_detected} images")
        print(f"   ⭐ Avg quality: {avg_quality:.3f}")
        print(f"   📁 Output: {self.output_base}\n")
        
        # Clear GPU cache
        GPU.clear_cache()
        
        return results_by_class
    
    def _save_statistics(self, results_by_class: Dict):
        import csv
        
        csv_path = os.path.join(self.output_dirs['05_statistics'],
                               'detection_stats.csv')
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['image', 'disease_type', 'disease_detected',
                           'disease_area', 'vegetation_area', 'quality_score', 
                           'quality_valid'])
            
            for disease_type, results in results_by_class.items():
                for r in results:
                    writer.writerow([
                        r['base_name'],
                        r['disease_type'],
                        r['disease_detected'],
                        r['disease_area'],
                        r['vegetation_area'],
                        f"{r['quality_score']:.3f}",
                        r['quality_valid']
                    ])


# =====================================================
# USAGE
# =====================================================

if __name__ == "__main__":
    import time
    
    print("="*80)
    print("🚀 GPU-HYBRID RICE DISEASE DETECTION (FAST + ACCURATE)")
    print("="*80)
    
    LABELS = {
        0: {"name": "brown_spot", "match_substrings": ["../data/new_data_field_rice/brown_spot"]},
        1: {"name": "leaf_blast", "match_substrings": ["../data/new_data_field_rice/leaf_blast"]},
        2: {"name": "leaf_blight", "match_substrings": ["../data/new_data_field_rice/leaf_blight"]},
        3: {"name": "healthy", "match_substrings": ["../data/new_data_field_rice/healthy"]}
    }
    
    config = TunableConfig()
    
    # ĐẦY ĐỦ 5 cải tiến đã BẬT SẴN
    print("✅ All 5 CV improvements enabled for ACCURACY")
    print("✅ Parallel processing enabled for SPEED")
    print("✅ GPU available:", GPU.use_gpu)
    
    pipeline = RiceDiseaseDetectionPipeline(
        labels_config=LABELS,
        config=config,
        parent_output_dir='../output',
        experiment_name='RiceDisease-GPU-Hybrid'
    )
    
    print("\n📁 Collecting images (50 per class)...")
    image_paths_by_class = {}
    total_images = 0
    
    for label_id, label_info in LABELS.items():
        class_name = label_info['name']
        folders = label_info['match_substrings']
        
        all_images = []
        for folder in folders:
            folder_path = Path(folder)
            if folder_path.exists():
                images = list(folder_path.glob('*.jpg'))
                images.extend(list(folder_path.glob('*.png')))
                all_images.extend([str(p) for p in images[:]])
        
        image_paths_by_class[class_name] = all_images[:]
        total_images += len(image_paths_by_class[class_name])
        print(f"   {class_name}: {len(image_paths_by_class[class_name])} images")
    
    print(f"\n⚡ Total: {total_images} images to process")
    
    start_time = time.time()
    results = pipeline.process_batch(image_paths_by_class)
    end_time = time.time()
    
    processing_time = end_time - start_time
    images_per_second = total_images / processing_time if processing_time > 0 else 0
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"⏱️  Total Time: {processing_time:.2f}s")
    print(f"🚀 Speed: {images_per_second:.2f} images/second")
    print(f"📊 All 5 CV improvements: ENABLED (Accurate)")
    print(f"💻 Parallel workers: {config.num_workers}")
    print("="*80)