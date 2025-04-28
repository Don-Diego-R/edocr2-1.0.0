import cv2
import numpy as np
import os
import shutil

# --- Find Drawing Box via Contours ---
def find_drawing_contour_box(image, canny_thresh1=50, canny_thresh2=150, debug_folder=None):
    """
    Finds the main drawing area using edge detection and contour analysis.
    Uses the exact logic from the user-provided original script.

    Args:
        image: The input BGR image.
        canny_thresh1: First threshold for the Canny edge detector.
        canny_thresh2: Second threshold for the Canny edge detector.
        debug_folder: If provided, save visualization images to this directory.

    Returns:
        A tuple (x, y, w, h) defining the bounding box of the detected drawing area.
        Returns (0, 0, width, height) if detection fails.
    """
    print("    [Contour Detect] Starting detection...")
    height, width = image.shape[:2]
    save_debug_step = lambda name, img_state: None # Default no-op
    persistent_debug_img = None

    # --- Nested Debug Save Helper --- 
    # Moved inside the main function to avoid polluting module namespace
    # and to only define it when debug_folder is actually provided.
    if debug_folder:
        # Check if exists before clearing - slightly safer
        if os.path.exists(debug_folder):
            shutil.rmtree(debug_folder) # Clear previous debug images
        os.makedirs(debug_folder, exist_ok=True)
        persistent_debug_img = image.copy() # Keep original colors for drawing overlays
        print(f"    [Debug] Saving visualization steps to: {debug_folder}")

        def _save_debug_image(name, img_state):
            # Ensure img_state is BGR before saving if it's grayscale
            img_to_save = img_state
            if len(img_state.shape) == 2:
                img_to_save = cv2.cvtColor(img_state, cv2.COLOR_GRAY2BGR)
            elif len(img_state.shape) == 3 and img_state.shape[2] == 1:
                 img_to_save = cv2.cvtColor(img_state, cv2.COLOR_GRAY2BGR)
            
            path = os.path.join(debug_folder, f"{name}.png")
            try:
                cv2.imwrite(path, img_to_save)
            except Exception as e:
                print(f"    [Debug] Warning: Failed to save debug image {path}: {e}")
        save_debug_step = _save_debug_image
    # --- End Debug Save Helper ---

    # 1. Preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if debug_folder: save_debug_step("debug_1_gray", gray)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    if debug_folder: save_debug_step("debug_2_blurred", blurred)

    # 2. Edge Detection
    edges = cv2.Canny(blurred, canny_thresh1, canny_thresh2)
    if debug_folder: save_debug_step("debug_3_canny_edges", edges)

    # 3. Contour Finding
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Save all contours image only if debug is enabled
    if debug_folder and persistent_debug_img is not None:
        img_with_contours = persistent_debug_img.copy()
        cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 1) # Draw all contours in green
        save_debug_step("debug_4_all_contours", img_with_contours)

    # 4. Contour Filtering (Exact original logic)
    best_box = (0, 0, width, height) # Default to full image
    max_area = 0
    found_box = False
    min_area = width * height * 0.1 # Require contour area to be at least 10% of image area

    if contours:
        # Sort contours by area (descending)
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in sorted_contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                break # No need to check smaller contours

            # Approximate the contour shape
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # Check if it looks roughly rectangular (4 vertices) and is large enough
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Add criteria: Bounding box should not be too thin or too flat
                aspect_ratio = w / float(h) if h > 0 else 0
                if 0.5 < aspect_ratio < 2.0: # Avoid extremely thin boxes
                    # Additional check: ensure the box isn't right at the edge (like the window border)
                    margin = 5 # pixels
                    if x > margin and y > margin and (x+w) < (width-margin) and (y+h) < (height-margin):
                        print(f"    [Contour Detect] Found candidate box: Rect(x={x}, y={y}, w={w}, h={h}), Area: {area:.0f}, Approx Vertices: {len(approx)}")
                        best_box = (x, y, w, h)
                        found_box = True
                        
                        # Save best contour image only if debug is enabled
                        if debug_folder and persistent_debug_img is not None:
                           # Draw the chosen contour and its bounding box
                           img_with_best_contour = persistent_debug_img.copy()
                           cv2.drawContours(img_with_best_contour, [contour], -1, (255, 0, 0), 2) # Blue
                           cv2.rectangle(img_with_best_contour, (x, y), (x + w, y + h), (0, 0, 255), 2) # Red
                           save_debug_step("debug_5_best_contour_box", img_with_best_contour)
                        break # Stop after finding the first suitable large rectangle

    if not found_box:
        print("    [Contour Detect] No suitable large rectangular contour found. Falling back to full image.")
        # Keep best_box as default (full image)
    else:
         print(f"    [Contour Detect] Selected box: x={best_box[0]}, y={best_box[1]}, w={best_box[2]}, h={best_box[3]}")

    # Return the best bounding box found (or the default full image box)
    return best_box
