import cv2
import numpy as np
# --- 1. PIPELINE TIỀN XỬ LÝ THÍCH ỨNG (ĐÃ CẬP NHẬT GỌT NHẸ VIỀN 5x5) ---
def crop_fundus_outer_black(img, tolerance=7):
    """Tự động cắt sát viền đen xung quanh võng mạc"""
    if img.ndim == 2:
        mask = img > tolerance
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray > tolerance
        if np.any(mask):
            check_rect = img[np.ix_(mask.any(1), mask.any(0))]
            if check_rect.shape[0] > 0 and check_rect.shape[1] > 0:
                return check_rect
    return img

def preprocess_fundus_adaptive(img_rgb, target_size=512, apply_graham=True):
    """
    Pipeline Production-grade: Khử nhiễu nền IDRiD bằng cách tăng ngưỡng lọc 
    và triệt tiêu hoàn toàn quầng sáng biên bằng cơ chế Background Mean Filling.
    """
    # 1. Auto-Crop viền đen (Nâng ngưỡng lên 15 để loại bỏ nhiễu nền IDRiD)
    cropped = crop_fundus_outer_black(img_rgb, tolerance=15)
    gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    
    # CHỈNH SỬA 1: Tăng ngưỡng lên 15 để bóc tách sạch sẽ dải nhiễu nén màu xanh bên rìa phải
    _, raw_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    
    # Làm mịn cấu trúc bề mặt mask trước khi tìm contour
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, close_kernel)
    
    # Trích xuất vùng mô lớn nhất
    contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    smooth_mask = np.zeros_like(raw_mask)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(smooth_mask, [largest_contour], -1, 255, -1)
    else:
        smooth_mask = raw_mask
    
    # 2. Square Padding giữ nguyên tỉ lệ 1:1
    h, w, _ = cropped.shape
    max_dim = max(h, w)
    
    square_img = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    square_mask = np.zeros((max_dim, max_dim), dtype=np.uint8)
    
    y_offset = (max_dim - h) // 2
    x_offset = (max_dim - w) // 2
    
    square_img[y_offset:y_offset+h, x_offset:x_offset+w] = cropped
    square_mask[y_offset:y_offset+h, x_offset:x_offset+w] = smooth_mask
    
    # 3. Resize ảnh và khử răng cưa hình học
    resized = cv2.resize(square_img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    resized_mask_float = cv2.resize(square_mask, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    _, resized_mask = cv2.threshold(resized_mask_float, 127, 255, cv2.THRESH_BINARY)
    
    if apply_graham:
        # --- CHỈNH SỬA 2: ĐỘT PHÁ CHỐNG LÓA BIÊN (ANTI-EDGE FLARE) ---
        # Tính toán màu trung bình của riêng vùng mô võng mạc
        mean_color = cv2.mean(resized, mask=resized_mask)[:3]
        
        # Tạo một bản sao và lấp đầy vùng ngoài bằng màu trung bình này
        img_filled = resized.copy()
        img_filled[resized_mask == 0] = [int(c) for c in mean_color]
        
        # Tiến hành Blur trên ảnh đã bù màu (Không còn bị sụt giảm dải sáng về 0)
        blurred = cv2.GaussianBlur(img_filled, (0, 0), target_size / 30)
        
        # Áp dụng công thức Ben Graham chuẩn
        enhanced = cv2.addWeighted(resized, 4, blurred, -4, 128)
        
        # Gọt cực nhẹ (5x5) để trả lại nền đen thuần khiết sắc nét ở biên
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        eroded_mask = cv2.erode(resized_mask, kernel, iterations=1)
        
        clean_enhanced = cv2.bitwise_and(enhanced, enhanced, mask=eroded_mask)
        return clean_enhanced
        
    return resized