"""
RICE DISEASE DETECTION - ADVANCED WITH 5 CV IMPROVEMENTS
=========================================================
Detect và highlight vùng bệnh trên ảnh lá lúa
Bổ sung 5 cải tiến không cần Deep Learning:
1. K-means adaptive thresholding (Lab)
2. LBP texture detection
3. Superpixel refinement
4. Shape filtering (solidity & eccentricity)
5. CLAHE illumination enhancement

Author: Claude
Date: 2025-10-05
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import os
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# TUNABLE CONFIGURATION - TẤT CẢ PARAMETERS Ở ĐÂY
# =====================================================

class TunableConfig:
    """
    TẤT CẢ THAM SỐ CÓ THỂ TINH CHỈNH - COMMENT CHI TIẾT
    ==================================================
    """
    
    def __init__(self):
        # ========== VEGETATION SEGMENTATION ==========
        # Phương pháp tách lá: 'exg', 'hsv', hoặc 'combined' (kết hợp cả 2)
        self.veg_method = 'combined'  # 'combined' cho kết quả tốt nhất
        
        # Morphological operations cho vegetation mask
        # Tăng size để loại bỏ noise nhiều hơn, giảm để giữ chi tiết
        self.veg_morph_close = (9, 9)  # Đóng lỗ hổng, default: (9,9)
        self.veg_morph_open = (7, 7)   # Loại bỏ noise nhỏ, default: (7,7)
        self.veg_erode = (3, 3)        # Thu nhỏ biên để chắc chắn, default: (3,3)
        
        # ========== DISEASE COLOR DETECTION (HSV) ==========
        # Mỗi bệnh có dải màu HSV riêng trong không gian màu HSV
        # HSV = (Hue: 0-180, Saturation: 0-255, Value: 0-255)
        
        self.disease_color_ranges = {
            'brown_spot': {
                # Màu nâu của brown spot
                'hsv_lower': np.array([5, 40, 40]),    # H_min, S_min, V_min
                'hsv_upper': np.array([25, 255, 200]), # H_max, S_max, V_max
                # Tăng coverage_threshold nếu detect quá nhiều vùng không phải bệnh
                'coverage_threshold': 0.40,  # 40% vùng phải có màu đúng, default: 0.40
                # Giảm color_variance_max nếu muốn màu đồng nhất hơn
                'color_variance_max': 40,    # Độ biến thiên màu tối đa, default: 40
            },
            
            'leaf_blast': {
                # Màu xám-trắng của leaf blast
                'hsv_lower': np.array([0, 0, 100]),    
                'hsv_upper': np.array([180, 80, 255]),
                'coverage_threshold': 0.35,  # default: 0.35
                'color_variance_max': 50,    # default: 50
            },
            
            'leaf_blight': {
                # Màu vàng-nâu của leaf blight
                'hsv_lower': np.array([15, 30, 30]),   
                'hsv_upper': np.array([35, 255, 220]),
                'coverage_threshold': 0.45,  # default: 0.45
                'color_variance_max': 45,    # default: 45
            }
        }
        
        # ========== YELLOW LEAF DETECTION ==========
        # Phát hiện lá vàng/chết (bổ sung cho disease detection)
        # HSV range cho yellow/orange leaves
        self.yellow_hsv_main = ([16, 80, 70], [38, 255, 255])  # Yellow
        self.yellow_hsv_orange = ([5, 80, 60], [20, 255, 255]) # Orange
        # LAB b-channel threshold (giá trị cao = vàng)
        self.yellow_lab_threshold = 145  # default: 145
        # Minimum area cho yellow regions
        self.yellow_min_area = 250  # pixels, default: 250
        
        # ========== MORPHOLOGY & POST-PROCESSING ==========
        # Kernel size cho disease mask morphology
        # Tăng để làm mịn biên, giảm để giữ chi tiết
        self.disease_morph_kernel = (7, 7)  # default: (7,7)
        self.disease_morph_close_iter = 2   # Số lần đóng lỗ hổng, default: 2
        self.disease_morph_open_iter = 1    # Số lần loại noise, default: 1
        
        # Loại bỏ vùng nhỏ (noise)
        self.min_region_area = 100  # pixels, default: 100
        
        # ========== EDGE DETECTION ==========
        # Canny edge detection thresholds
        self.canny_low = 50   # Ngưỡng thấp, default: 50
        self.canny_high = 150 # Ngưỡng cao, default: 150
        self.edge_dilate_kernel = (3, 3)  # Mở rộng edges, default: (3,3)
        self.edge_dilate_iter = 1  # Số lần dilate, default: 1
        
        # ========== 1. ADAPTIVE K-MEANS LAB ==========
        # K-means clustering trong Lab color space để tự động tìm vùng bệnh
        self.use_adaptive_kmeans = True  # Bật/tắt adaptive thresholding
        self.kmeans_k = 3                # Số cụm màu, default: 3
        self.kmeans_attempts = 3         # Số lần chạy K-means, default: 3
        self.kmeans_min_cluster_frac = 0.02  # Bỏ các cụm quá nhỏ (<2%), default: 0.02
        self.kmeans_ab_weight = 1.0      # Trọng số cho kênh a,b (tăng để nhấn màu), default: 1.0
        self.kmeans_veto_green_h = (35, 95)  # Loại cụm xanh lá (Hue range), default: (35,95)
        
        # ========== 2. LBP TEXTURE ==========
        # Local Binary Pattern để phát hiện texture đốm bệnh
        self.use_lbp = True              # Bật/tắt LBP texture detection
        self.lbp_radius = 1              # Bán kính LBP, default: 1
        self.lbp_thresh = 0.12           # Threshold cho rare patterns, default: 0.12
        self.lbp_min_area = 150          # Vùng LBP tối thiểu, default: 150
        
        # ========== 3. SUPERPIXEL REFINEMENT ==========
        # SLIC superpixel để làm mịn biên và loại nhiễu
        self.use_superpixel_refine = True  # Bật/tắt superpixel refinement
        self.slic_region_size = 20       # Kích thước superpixel, default: 20
        self.slic_ruler = 10.0           # Tham số SLIC ruler, default: 10.0
        self.slic_min_size = 40          # Superpixel tối thiểu, default: 40
        self.superpixel_vote_ratio = 0.4  # Tỉ lệ pixel dương để bật cả superpixel, default: 0.4
        
        # ========== 4. SHAPE FILTERS ==========
        # Lọc theo hình dạng để loại nhiễu không phải vết bệnh
        self.shape_min_solidity = 0.6    # Solidity tối thiểu (area/convex_area), default: 0.6
        self.shape_max_eccentricity = 0.98  # Eccentricity tối đa (loại vùng quá dẹt), default: 0.98
        
        # ========== 5. CLAHE ILLUMINATION ==========
        # Contrast Limited Adaptive Histogram Equalization
        self.use_clahe = True            # Bật/tắt CLAHE enhancement
        self.clahe_clip = 2.0            # Clip limit, default: 2.0
        self.clahe_tile = (8, 8)         # Tile grid size, default: (8,8)
        
        # ========== VISUALIZATION ==========
        # Màu sắc cho visualization overlay
        self.viz_vegetation_color = (0, 255, 0)    # Green cho vegetation
        self.viz_vegetation_alpha = 0.1            # Độ trong suốt, default: 0.1
        self.viz_disease_color = (0, 0, 255)       # Red cho disease
        self.viz_disease_alpha = 0.3               # Độ trong suốt, default: 0.3
        self.viz_contour_color = (0, 255, 255)     # Yellow cho contour
        self.viz_contour_thickness = 2             # Độ dày đường viền, default: 2
        
        # ========== QUALITY CONTROL ==========
        # Blur detection (Laplacian variance)
        # Tăng threshold nếu muốn lọc nghiêm ngặt hơn
        self.qc_blur_threshold = 100  # default: 100
        
        # Exposure check
        self.qc_exposure_optimal = (80, 180)   # Range tối ưu, default: (80, 180)
        self.qc_exposure_acceptable = (50, 200) # Range chấp nhận, default: (50, 200)
        
        # Edge density check
        self.qc_edge_optimal = (0.05, 0.3)    # Range tối ưu, default: (0.05, 0.3)
        self.qc_edge_acceptable = (0.02, 0.5)  # Range chấp nhận, default: (0.02, 0.5)
        
        # Overall quality score threshold
        self.qc_min_score = 0.6  # Điểm tối thiểu để pass QC, default: 0.6
        
        # ========== COLOR CONSTANCY ==========
        # Ổn định màu sắc trước khi xử lý
        self.use_color_constancy = True  # True/False
        self.color_constancy_method = 'shades_of_gray'  # 'gray_world' hoặc 'shades_of_gray'
        self.shades_of_gray_p = 6  # Parameter p cho Shades of Gray, default: 6
        
        # ========== OUTPUT SETTINGS ==========
        # Có lưu các intermediate results không
        self.save_vegetation_mask = True   # Lưu vegetation mask
        self.save_disease_mask = True      # Lưu disease mask  
        self.save_visualization = True     # Lưu ảnh visualization
        self.save_stats = True             # Lưu statistics CSV
    
    def to_dict(self):
        """Convert config to dictionary for saving"""
        return {k: v if not isinstance(v, np.ndarray) else v.tolist() 
                for k, v in self.__dict__.items()}
    
    def save(self, path: str):
        """Save config to JSON"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, default=str, ensure_ascii=False)
    
    def print_summary(self):
        """In ra summary của config"""
        print("\n" + "="*60)
        print("ADVANCED TUNABLE PARAMETERS SUMMARY")
        print("="*60)
        print(f"Vegetation Method: {self.veg_method}")
        print(f"Color Constancy: {self.use_color_constancy} ({self.color_constancy_method})")
        print(f"✨ Adaptive K-means: {self.use_adaptive_kmeans}")
        print(f"✨ LBP Texture: {self.use_lbp}")
        print(f"✨ Superpixel Refine: {self.use_superpixel_refine}")
        print(f"✨ Shape Filtering: solidity≥{self.shape_min_solidity}, ecc≤{self.shape_max_eccentricity}")
        print(f"✨ CLAHE Enhancement: {self.use_clahe}")
        print(f"Disease Morphology Kernel: {self.disease_morph_kernel}")
        print(f"Min Region Area: {self.min_region_area} pixels")
        print(f"Quality Score Threshold: {self.qc_min_score}")
        print("="*60 + "\n")


# =====================================================
# OUTPUT STRUCTURE
# =====================================================

def get_output_folder(parent_dir: str, env_name: str) -> str:
    """Tạo output folder với timestamp"""
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = f"{env_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    output_dir = os.path.join(parent_dir, experiment_id)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def create_output_structure(base_path: str, class_names: List[str]) -> Dict[str, str]:
    """Tạo structure output - SIMPLIFIED (không có crops)"""
    folders = {
        '01_originals': {},           # Ảnh gốc
        '02_vegetation_masks': {},    # Vegetation masks
        '03_disease_masks': {},       # Disease masks
        '04_detected_results': {},    # Ảnh với vùng bệnh được highlight
        '05_statistics': base_path,   # Statistics
        '06_config': base_path        # Configuration
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
                key = f"{folder_type}_{class_name}"
                folder_paths[key] = folder_path
    
    return folder_paths


# =====================================================
# VEGETATION SEGMENTATION
# =====================================================

class VegetationSegmenter:
    """Tách vegetation từ background"""
    
    def __init__(self, config: TunableConfig):
        self.config = config
    
    def segment_vegetation(self, image: np.ndarray) -> np.ndarray:
        """Segment vegetation mask"""
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
        
        # Morphological operations
        mask = self._apply_morphology(mask)
        
        return mask
    
    def _segment_exg(self, image: np.ndarray) -> np.ndarray:
        """Excess Green Index"""
        b, g, r = cv2.split(image.astype(np.float32) / 255.0)
        exg = 2 * g - r - b
        
        exg_uint8 = ((exg - exg.min()) / (exg.max() - exg.min() + 1e-8) * 255).astype(np.uint8)
        _, mask = cv2.threshold(exg_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return mask
    
    def _segment_hsv(self, image: np.ndarray) -> np.ndarray:
        """HSV green + yellow detection"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Green vegetation
        green = cv2.inRange(hsv, np.array([25, 20, 20]), np.array([95, 255, 255]))
        
        # Yellow (diseased leaves)
        yellow = cv2.inRange(hsv, np.array([15, 15, 40]), np.array([45, 255, 255]))
        
        # Combine
        mask = cv2.bitwise_or(green, yellow)
        return mask
    
    def _apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological operations"""
        # Close
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                 self.config.veg_morph_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        
        # Open
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                self.config.veg_morph_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        
        # Erode
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                 self.config.veg_erode)
        mask = cv2.erode(mask, kernel_erode)
        
        return mask


# =====================================================
# COLOR CONSTANCY (CẢI TIẾN #5: CLAHE)
# =====================================================

class ColorConstancy:
    """Ổn định màu sắc + CLAHE enhancement"""
    
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
    def apply(image: np.ndarray, config: TunableConfig) -> np.ndarray:
        """Apply color constancy + CLAHE (CẢI TIẾN #5)"""
        out = image.copy()
        
        # Color constancy
        if config.color_constancy_method == 'gray_world':
            out = ColorConstancy.gray_world(out)
        elif config.color_constancy_method == 'shades_of_gray':
            out = ColorConstancy.shades_of_gray(out, config.shades_of_gray_p)
        
        # CẢI TIẾN #5: CLAHE trên V (HSV) và b (Lab) để tăng contrast
        if config.use_clahe:
            # CLAHE trên HSV-V channel
            hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
            clahe = cv2.createCLAHE(clipLimit=config.clahe_clip, 
                                   tileGridSize=config.clahe_tile)
            hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
            out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # CLAHE trên Lab-b channel (nhạy với vàng)
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            lab[:, :, 2] = clahe.apply(lab[:, :, 2])
            out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return out


# =====================================================
# DISEASE SEGMENTATION (CẢI TIẾN #1, #2, #3, #4)
# =====================================================

class DiseaseSegmenter:
    """Segment disease regions với 4 cải tiến CV nâng cao"""
    
    def __init__(self, config: TunableConfig):
        self.config = config
    
    def segment_disease(self, image: np.ndarray, disease_type: str,
                       vegetation_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Segment disease regions với full pipeline"""
        
        # 1. HSV color segmentation (baseline)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if disease_type in self.config.disease_color_ranges:
            color_cfg = self.config.disease_color_ranges[disease_type]
            mask_hsv = cv2.inRange(hsv, color_cfg['hsv_lower'], color_cfg['hsv_upper'])
        else:
            mask_hsv = self._segment_generic(image, vegetation_mask)
        
        # 2. Yellow leaf detection (bổ sung)
        if vegetation_mask is not None:
            yellow_mask = self._segment_yellow_leaf(image, vegetation_mask)
            mask = cv2.bitwise_or(mask_hsv, yellow_mask)
        else:
            mask = mask_hsv
        
        # CẢI TIẾN #1: Adaptive K-means Lab clustering
        if vegetation_mask is not None and self.config.use_adaptive_kmeans:
            adaptive_mask = self._adaptive_kmeans_lab(image, vegetation_mask)
            mask = cv2.bitwise_or(mask, adaptive_mask)
        
        # CẢI TIẾN #2: LBP Texture detection
        if vegetation_mask is not None and self.config.use_lbp:
            lbp_mask = self._lbp_mask(image, vegetation_mask)
            mask = cv2.bitwise_or(mask, lbp_mask)
        
        # 3. Edge enhancement (baseline)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.config.canny_low, self.config.canny_high)
        kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                self.config.edge_dilate_kernel)
        edges_dilated = cv2.dilate(edges, kernel_edge, 
                                   iterations=self.config.edge_dilate_iter)
        mask = cv2.bitwise_or(mask, edges_dilated)
        
        # Apply vegetation ROI
        if vegetation_mask is not None:
            mask = cv2.bitwise_and(mask, vegetation_mask)
        
        # 4. Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                          self.config.disease_morph_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 
                               iterations=self.config.disease_morph_close_iter)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 
                               iterations=self.config.disease_morph_open_iter)
        
        # CẢI TIẾN #4: Shape filtering (solidity & eccentricity)
        mask = self._remove_small_regions(mask, self.config.min_region_area)
        
        # CẢI TIẾN #3: Superpixel refinement
        if vegetation_mask is not None and self.config.use_superpixel_refine:
            mask = self._refine_with_superpixels(image, mask, vegetation_mask)
        
        return mask
    
    # CẢI TIẾN #1: ADAPTIVE K-MEANS LAB
    def _adaptive_kmeans_lab(self, image: np.ndarray, vegetation_mask: np.ndarray) -> np.ndarray:
        """K-means clustering trong Lab color space để tự động tìm vùng bệnh"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(lab)
        
        # Chỉ lấy pixel trong vùng lá
        ys, xs = np.where(vegetation_mask > 0)
        if len(xs) < 500:
            return np.zeros_like(vegetation_mask)
        
        # Chuẩn bị samples: L, a*weight, b*weight
        samples = np.stack([
            L[ys, xs],
            self.config.kmeans_ab_weight * a[ys, xs],
            self.config.kmeans_ab_weight * b[ys, xs]
        ], axis=1).astype(np.float32)
        
        # K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
        _, labels, centers = cv2.kmeans(
            samples, 
            self.config.kmeans_k, 
            None, 
            criteria,
            self.config.kmeans_attempts, 
            cv2.KMEANS_PP_CENTERS
        )
        
        # Đánh giá từng cụm để tìm cụm "bệnh"
        mask = np.zeros_like(vegetation_mask)
        total = len(labels)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        for k in range(centers.shape[0]):
            # Bỏ cụm quá nhỏ
            frac = (labels == k).sum() / total
            if frac < self.config.kmeans_min_cluster_frac:
                continue
            
            Lc, ac, bc = centers[k]
            
            # Tạo mask cho cụm này
            cluster = np.zeros_like(vegetation_mask)
            cluster_indices = (labels.ravel() == k)
            cluster[ys[cluster_indices], xs[cluster_indices]] = 255
            
            # Kiểm tra Hue để loại cụm xanh lá
            h_vals = hsv[:, :, 0][cluster > 0]
            if h_vals.size > 0:
                h_med = np.median(h_vals)
                if self.config.kmeans_veto_green_h[0] <= h_med <= self.config.kmeans_veto_green_h[1]:
                    continue  # Skip cụm xanh lá
            
            # Tiêu chí bệnh: L thấp (tối) hoặc b cao (vàng) hoặc a cao (nâu/đỏ)
            if (Lc < 120) or (bc > 150) or (ac > 150):
                mask = cv2.bitwise_or(mask, cluster.astype(np.uint8))
        
        # Làm sạch
        mask = self._remove_small_regions(mask, self.config.min_region_area)
        
        return mask
    
    # CẢI TIẾN #2: LBP TEXTURE
    def _lbp_mask(self, image: np.ndarray, vegetation_mask: np.ndarray) -> np.ndarray:
        """Local Binary Pattern để phát hiện texture đốm bệnh"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        g = gray.astype(np.int16)
        
        # 8-neighborhood LBP
        shifts = [(0, 1), (0, -1), (1, 0), (-1, 0), 
                  (1, 1), (1, -1), (-1, 1), (-1, -1)]
        lbp = np.zeros_like(g, dtype=np.uint8)
        
        for idx, (dy, dx) in enumerate(shifts):
            shifted = np.zeros_like(g)
            h, w = g.shape
            
            # Compute valid region
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
        
        # Tìm rare patterns (texture không đồng nhất)
        hist = cv2.calcHist([lbp], [0], vegetation_mask, [256], [0, 256]).flatten()
        if hist.sum() == 0:
            return np.zeros_like(vegetation_mask)
        
        # Các mã xuất hiện thấp = texture đốm bất thường
        rare = np.where(hist / hist.sum() < 0.01)[0]
        rare_mask = np.isin(lbp, rare).astype(np.uint8) * 255
        rare_mask = cv2.bitwise_and(rare_mask, vegetation_mask)
        
        # Morphology
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        rare_mask = cv2.morphologyEx(rare_mask, cv2.MORPH_OPEN, k, iterations=1)
        rare_mask = self._remove_small_regions(rare_mask, self.config.lbp_min_area)
        
        return rare_mask
    
    # CẢI TIẾN #3: SUPERPIXEL REFINEMENT
    def _refine_with_superpixels(self, image: np.ndarray, mask: np.ndarray,
                                 vegetation_mask: np.ndarray) -> np.ndarray:
        """SLIC superpixel refinement để làm mịn biên"""
        try:
            # Thử dùng SLIC từ opencv-contrib
            import cv2.ximgproc as xi
            slic = xi.createSuperpixelSLIC(
                image, 
                algorithm=xi.SLICO,
                region_size=self.config.slic_region_size,
                ruler=self.config.slic_ruler
            )
            slic.iterate(10)
            labels = slic.getLabels()
            
            out = np.zeros_like(mask)
            n_labels = labels.max() + 1
            
            for lab in range(n_labels):
                sp = (labels == lab)
                
                # Chỉ xét superpixel trong vegetation
                if vegetation_mask is not None:
                    sp = sp & (vegetation_mask > 0)
                
                if sp.sum() < self.config.slic_min_size:
                    continue
                
                # Vote: nếu >40% pixel trong superpixel là disease → cả superpixel là disease
                vote = (mask[sp] > 0).mean()
                if vote >= self.config.superpixel_vote_ratio:
                    out[sp] = 255
            
            return out
            
        except Exception:
            # Fallback: Mean shift + region voting
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
        """Segment yellow/diseased areas on leaves"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Yellow/orange ranges
        y_lower, y_upper = self.config.yellow_hsv_main
        o_lower, o_upper = self.config.yellow_hsv_orange
        
        y_main = cv2.inRange(hsv, np.array(y_lower), np.array(y_upper))
        o_main = cv2.inRange(hsv, np.array(o_lower), np.array(o_upper))
        y = cv2.bitwise_or(y_main, o_main)
        
        # LAB b-channel (yellow detection)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        _, bmask = cv2.threshold(lab[:, :, 2], self.config.yellow_lab_threshold, 
                                255, cv2.THRESH_BINARY)
        
        # Combine and mask with vegetation
        y = cv2.bitwise_and(y, bmask)
        y = cv2.bitwise_and(y, vegetation_mask)
        
        # Morphology
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        y = cv2.morphologyEx(y, cv2.MORPH_CLOSE, k, iterations=1)
        y = cv2.morphologyEx(y, cv2.MORPH_OPEN, k, iterations=1)
        
        # Remove small regions
        y = self._remove_small_regions(y, self.config.yellow_min_area)
        
        return y
    
    # CẢI TIẾN #4: SHAPE FILTERING (trong _remove_small_regions)
    def _remove_small_regions(self, mask: np.ndarray, min_area: int) -> np.ndarray:
        """Remove small regions + shape filtering (solidity & eccentricity)"""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        output = np.zeros_like(mask)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Filter by area
            if area < min_area:
                continue
            
            # Extract component
            comp = (labels == i).astype(np.uint8)
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not cnts:
                continue
            
            cnt = max(cnts, key=cv2.contourArea)
            
            # CẢI TIẾN #4: Tính solidity (diện tích / diện tích convex hull)
            hull = cv2.convexHull(cnt)
            area_cnt = cv2.contourArea(cnt) + 1e-6
            area_hull = cv2.contourArea(hull) + 1e-6
            solidity = area_cnt / area_hull
            
            # Tính eccentricity từ ellipse fit
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
            
            # Filter by shape: solidity cao (không rỗng) và eccentricity thấp (không quá dẹt)
            if (solidity >= self.config.shape_min_solidity and 
                ecc <= self.config.shape_max_eccentricity):
                output[labels == i] = 255
        
        return output
    
    def _segment_generic(self, image: np.ndarray, 
                        vegetation_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Generic disease segmentation"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # General disease colors
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


# =====================================================
# QUALITY CONTROL
# =====================================================

class QualityControl:
    """Kiểm tra chất lượng ảnh"""
    
    def __init__(self, config: TunableConfig):
        self.config = config
    
    def check_quality(self, image: np.ndarray) -> Dict:
        """Comprehensive quality check"""
        reasons = []
        scores = []
        
        # Blur check
        blur_score = self._check_blur(image)
        scores.append(blur_score)
        if blur_score < 0.5:
            reasons.append('blurry')
        
        # Exposure check
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
        """Laplacian variance for blur detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        score = min(1.0, laplacian_var / 500.0)
        return score
    
    def _check_exposure(self, image: np.ndarray) -> float:
        """Check exposure balance"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        opt_min, opt_max = self.config.qc_exposure_optimal
        acc_min, acc_max = self.config.qc_exposure_acceptable
        
        if opt_min < mean_brightness < opt_max:
            score = 1.0
        elif acc_min < mean_brightness < acc_max:
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
        """Edge sharpness check"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        edge_density = np.sum(edges > 0) / edges.size
        
        opt_min, opt_max = self.config.qc_edge_optimal
        acc_min, acc_max = self.config.qc_edge_acceptable
        
        if opt_min < edge_density < opt_max:
            score = 1.0
        elif acc_min < edge_density < acc_max:
            score = 0.7
        else:
            score = 0.3
        
        return score


# =====================================================
# VISUALIZATION
# =====================================================

class Visualizer:
    """Tạo visualization với vùng bệnh được highlight"""
    
    def __init__(self, config: TunableConfig):
        self.config = config
    
    def create_detection_image(self, image: np.ndarray, 
                              disease_mask: np.ndarray,
                              vegetation_mask: np.ndarray) -> np.ndarray:
        """Tạo ảnh với vùng bệnh được highlight"""
        overlay = image.copy()
        
        # Vegetation overlay (green tint)
        veg_color = np.zeros_like(image)
        veg_color[vegetation_mask > 0] = self.config.viz_vegetation_color
        overlay = cv2.addWeighted(overlay, 1 - self.config.viz_vegetation_alpha, 
                                 veg_color, self.config.viz_vegetation_alpha, 0)
        
        # Disease overlay (red highlight)
        disease_color = np.zeros_like(image)
        disease_color[disease_mask > 0] = self.config.viz_disease_color
        overlay = cv2.addWeighted(overlay, 1 - self.config.viz_disease_alpha, 
                                 disease_color, self.config.viz_disease_alpha, 0)
        
        # Contours around disease regions
        contours, _ = cv2.findContours(disease_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, 
                        self.config.viz_contour_color, 
                        self.config.viz_contour_thickness)
        
        return overlay


# =====================================================
# MAIN PIPELINE
# =====================================================

class RiceDiseaseDetectionPipeline:
    """Pipeline detect vùng bệnh trên lá lúa với 5 cải tiến CV"""
    
    def __init__(self,
                 labels_config: Dict,
                 config: Optional[TunableConfig] = None,
                 parent_output_dir: str = '../output',
                 experiment_name: str = 'RiceDisease'):
        
        self.labels_config = labels_config
        self.class_names = [info['name'] for info in labels_config.values()]
        
        # Config
        if config is None:
            self.config = TunableConfig()
        else:
            self.config = config
        
        # Output
        self.output_base = get_output_folder(parent_output_dir, experiment_name)
        self.output_dirs = create_output_structure(self.output_base, self.class_names)
        
        # Save config
        self.config.save(os.path.join(self.output_dirs['06_config'], 
                                      'tunable_config.json'))
        
        # Print summary
        self.config.print_summary()
        print(f"📁 Output: {self.output_base}\n")
        
        # Initialize components
        self.veg_segmenter = VegetationSegmenter(self.config)
        self.disease_segmenter = DiseaseSegmenter(self.config)
        self.qc = QualityControl(self.config)
        self.visualizer = Visualizer(self.config)
        
        # Stats
        self.stats = []
    
    def process_image(self, image_path: str, disease_type: str) -> Dict:
        """Process một ảnh"""
        base_name = Path(image_path).stem
        
        # Load image
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
        
        # Color constancy + CLAHE (CẢI TIẾN #5)
        if self.config.use_color_constancy:
            image = ColorConstancy.apply(image, self.config)
        
        # Quality check
        qc_result = self.qc.check_quality(image)
        result['quality_score'] = qc_result['score']
        result['quality_valid'] = qc_result['is_valid']
        
        # Save original
        original_path = os.path.join(
            self.output_dirs[f'01_originals_{disease_type}'],
            f'{base_name}.jpg'
        )
        cv2.imwrite(original_path, image)
        
        # Skip healthy images
        if disease_type == 'healthy':
            return result
        
        # Vegetation segmentation
        vegetation_mask = self.veg_segmenter.segment_vegetation(image)
        vegetation_area = np.sum(vegetation_mask > 0)
        result['vegetation_area'] = int(vegetation_area)
        
        if vegetation_area < 1000:
            return result
        
        # Save vegetation mask
        if self.config.save_vegetation_mask:
            veg_mask_path = os.path.join(
                self.output_dirs[f'02_vegetation_masks_{disease_type}'],
                f'{base_name}_veg.png'
            )
            cv2.imwrite(veg_mask_path, vegetation_mask)
        
        # Disease segmentation (với 4 cải tiến #1,#2,#3,#4)
        disease_mask = self.disease_segmenter.segment_disease(
            image, disease_type, vegetation_mask
        )
        disease_area = np.sum(disease_mask > 0)
        result['disease_area'] = int(disease_area)
        result['disease_detected'] = disease_area > 0
        
        # Save disease mask
        if self.config.save_disease_mask:
            mask_path = os.path.join(
                self.output_dirs[f'03_disease_masks_{disease_type}'],
                f'{base_name}_disease.png'
            )
            cv2.imwrite(mask_path, disease_mask)
        
        # Create and save visualization
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
        """Process batch images"""
        total_images = sum(len(paths) for paths in image_paths_by_class.values())
        
        print(f"🚀 Processing {total_images} images with 5 CV improvements...\n")
        
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
        if self.config.save_stats:
            self._save_statistics(results_by_class)
        
        # Print summary
        total_detected = sum(1 for r in self.stats if r['disease_detected'])
        avg_quality = np.mean([r['quality_score'] for r in self.stats])
        
        print(f"\n✅ COMPLETED!")
        print(f"   📸 Processed: {len(self.stats)} images")
        print(f"   🔍 Disease detected: {total_detected} images")
        print(f"   ⭐ Avg quality: {avg_quality:.3f}")
        print(f"   📁 Output: {self.output_base}\n")
        
        return results_by_class
    
    def _save_statistics(self, results_by_class: Dict):
        """Save statistics to CSV"""
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
# USAGE EXAMPLE
# =====================================================

if __name__ == "__main__":
    # Labels configuration
    LABELS = {
        0: {"name": "brown_spot", "match_substrings": ["../data/new_data_field_rice/brown_spot"]},
        1: {"name": "leaf_blast", "match_substrings": ["../data/new_data_field_rice/leaf_blast"]},
        2: {"name": "leaf_blight", "match_substrings": ["../data/new_data_field_rice/leaf_blight"]},
        3: {"name": "healthy", "match_substrings": ["../data/new_data_field_rice/healthy"]}
    }
    
    # Initialize config với 5 cải tiến
    config = TunableConfig()
    
    # ====== PRESETS ĐỀ XUẤT ======
    
    # PRESET 1: Ruộng ánh sáng gắt (accurate mode)
    # config.use_clahe = True
    # config.shades_of_gray_p = 8
    # config.use_adaptive_kmeans = True
    # config.use_lbp = True
    # config.use_superpixel_refine = True
    
    # PRESET 2: Ảnh nền phức tạp (strict filtering)
    # config.min_region_area = 250
    # config.shape_min_solidity = 0.7
    # config.superpixel_vote_ratio = 0.5
    
    # PRESET 3: Bệnh rải rác nhỏ (sensitive mode)
    # config.superpixel_vote_ratio = 0.3
    # config.lbp_min_area = 120
    # config.kmeans_min_cluster_frac = 0.01
    
    # PRESET 4: Tránh "ăn" nền xanh
    # config.kmeans_veto_green_h = (40, 90)
    # config.kmeans_ab_weight = 0.8
    
    # Initialize pipeline
    pipeline = RiceDiseaseDetectionPipeline(
        labels_config=LABELS,
        config=config,
        parent_output_dir='../output',
        experiment_name='RiceDisease-Advanced'
    )
    
    # Collect images (50 per class for testing)
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
                all_images.extend([str(p) for p in images[:50]])
        
        image_paths_by_class[class_name] = all_images[:50]
        print(f"Collected {len(image_paths_by_class[class_name])} images for {class_name}")
    
    # Process batch
    results = pipeline.process_batch(image_paths_by_class)
    
    print("\n" + "="*60)
    print("DETECTION WITH 5 CV IMPROVEMENTS COMPLETED!")
    print("="*60)