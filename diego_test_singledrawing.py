import cv2, os
import numpy as np
from edocr2 import tools
from pdf2image import convert_from_path
import time
import glob # Import glob for file searching
import traceback # For detailed error printing
import math # For Euclidean distance
import shutil # For cleaning debug folders

# --- Helper Function: Simplify by Dominant Color Blocks ---
def simplify_by_dominant_color_blocks(image, block_size=32):
    """Simplifies the image by filling blocks with their local dominant color.
    Args:
        image: The input BGR image.
        block_size: The width and height of the square blocks.
    Returns:
        The simplified image.
    """
    print(f"    [Block Simplify] Simplifying image with {block_size}x{block_size} blocks...")
    try:
        height, width = image.shape[:2]
        output_image = np.zeros_like(image)

        for y in range(0, height - block_size + 1, block_size):
            for x in range(0, width - block_size + 1, block_size):
                block = image[y:y+block_size, x:x+block_size]
                hist_b = cv2.calcHist([block], [0], None, [256], [0, 256])
                hist_g = cv2.calcHist([block], [1], None, [256], [0, 256])
                hist_r = cv2.calcHist([block], [2], None, [256], [0, 256])
                dominant_b = np.argmax(hist_b)
                dominant_g = np.argmax(hist_g)
                dominant_r = np.argmax(hist_r)
                dominant_color = (int(dominant_b), int(dominant_g), int(dominant_r))
                output_image[y:y+block_size, x:x+block_size] = dominant_color

        if height % block_size != 0:
            output_image[height - (height % block_size):height, :] = output_image[height - (height % block_size) - 1:height - (height % block_size), :]
        if width % block_size != 0:
             output_image[:, width - (width % block_size):width] = output_image[:, width - (width % block_size) - 1:width - (width % block_size)]

        print("    [Block Simplify] Simplification complete.")
        return output_image

    except Exception as e:
        print(f"    [Block Simplify] Error during block simplification: {e}")
        return image # Return original on error

# --- NEW Helper Function: Find Drawing Box via Contours ---
def find_drawing_contour_box(image, canny_thresh1=50, canny_thresh2=150, debug_folder=None):
    """
    Finds the main drawing area using edge detection and contour analysis.

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

    if debug_folder:
        if os.path.exists(debug_folder):
            shutil.rmtree(debug_folder) # Clear previous debug images
        os.makedirs(debug_folder, exist_ok=True)
        persistent_debug_img = image.copy() # Keep original colors for drawing overlays
        print(f"    [Debug] Saving visualization steps to: {debug_folder}")

        # --- Debug Save Helper ---
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
    save_debug_step("debug_1_gray", gray)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    save_debug_step("debug_2_blurred", blurred)

    # 2. Edge Detection
    edges = cv2.Canny(blurred, canny_thresh1, canny_thresh2)
    save_debug_step("debug_3_canny_edges", edges)

    # 3. Contour Finding
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    if persistent_debug_img is not None:
        img_with_contours = persistent_debug_img.copy()
        cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 1) # Draw all contours in green
        save_debug_step("debug_4_all_contours", img_with_contours)

    # 4. Contour Filtering
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
                        
                        if persistent_debug_img is not None:
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


# --- Configuration ---
input_folder = 'tests/Shaperfy' # Folder containing drawings
#input_folder = 'tests/test_samples' # Or use the original sample folder
original_folder = os.path.join(input_folder, 'original') # Folder containing drawings
processed_folder = os.path.join(input_folder, 'processed') # Save processed files in a subfolder
steps_folder = os.path.join(input_folder, 'steps')       # Save intermediate steps here
os.makedirs(original_folder, exist_ok=True) # Ensure output folder exists
os.makedirs(processed_folder, exist_ok=True) # Ensure output folder exists
os.makedirs(steps_folder, exist_ok=True)       # Ensure steps folder exists

# Define supported file extensions
supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.pdf', '.PNG', '.JPG', '.JPEG', '.BMP', '.TIFF', '.PDF']

# Find all supported files in the input folder
files_to_process = []
for ext in supported_extensions:
    files_to_process.extend(glob.glob(os.path.join(original_folder, f'*{ext}')))

if not files_to_process:
    print(f"No supported image or PDF files found in '{original_folder}'.")
    exit()

print(f"Found {len(files_to_process)} files to process.")

# --- Process Each File ---
for file_path in files_to_process:
    print(f"\n--- Processing: {file_path} ---")
    base_name = os.path.basename(file_path)
    name_part, ext_part = os.path.splitext(base_name)
    cropped_output_filename = f"{name_part}_cropped{ext_part}"
    cropped_output_path = os.path.join(processed_folder, cropped_output_filename)
    bin_output_filename = f"{name_part}_bin{ext_part}" # Filename for binary image
    bin_output_path = os.path.join(steps_folder, bin_output_filename) # Full path for binary image

    # --- Create a specific debug folder for this image's steps ---
    debug_subfolder = os.path.join(steps_folder, f"{name_part}_debug")

    # Skip if cropped file already exists (optional)
    # if os.path.exists(cropped_output_path):
    #     print(f"Skipping, output file already exists: {cropped_output_path}")
    #     continue

    try:
        # --- Step 1: Load the Drawing ---
        img = None
        if file_path.lower().endswith('.pdf'):
            img_list = convert_from_path(file_path, dpi=200) # Added DPI for better PDF quality
            if not img_list:
                print(f"Error: Could not convert PDF page: {file_path}")
                continue # Skip to next file
            img = np.array(img_list[0])
            # Ensure image is in BGR format
            if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif img.shape[2] == 3: img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # PDF usually loads as RGB
        else:
            img = cv2.imread(file_path, cv2.IMREAD_COLOR) # Ensure color loading
            if img is None:
                print(f"Error: Could not read image file: {file_path}")
                continue # Skip to next file
            # Ensure it's 3 channels even if grayscale was read (though IMREAD_COLOR should handle it)
            if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


        print(f"Image loaded successfully, shape: {img.shape}")
        if img.shape[2] != 3:
            print(f"Warning: Image does not have 3 channels (BGR expected), shape is {img.shape}. Skipping file.")
            continue


        # --- Step 2: Find Drawing Box using Contours ---
        start_time = time.time()
        # Using default Canny thresholds for now
        x, y, w, h = find_drawing_contour_box(img, debug_folder=debug_subfolder)
        end_time = time.time()
        print(f"Contour detection finished in {end_time - start_time:.2f} seconds.")

        # --- Step 3: Crop the Image ---
        if w > 0 and h > 0: # Check if valid box found
             cropped_img = img[y:y+h, x:x+w]
             print(f"Cropped image shape: {cropped_img.shape}")

             # --- Step 4: Save Cropped Result ---
             print(f"Saving cropped image to: {cropped_output_path}")
             save_success = cv2.imwrite(cropped_output_path, cropped_img)
             if not save_success:
                 print(f"Error: Failed to save image to {cropped_output_path}")
             else:
                 print("Cropped image saved successfully.")
        else:
            print("Error: Contour detection failed to find a valid drawing box. Saving original image instead.")
            # Optionally save the original image to the output path if border detection failed badly
            save_success = cv2.imwrite(cropped_output_path, img)
            if not save_success:
                 print(f"Error: Failed to save original image to {cropped_output_path}")


        # --- Step X (Optional): Simplify Image by Blocks (can be applied to cropped_img) ---
        # print("Simplifying image by dominant color blocks...")
        # start_time = time.time()
        # simplified_cropped_img = simplify_by_dominant_color_blocks(cropped_img, block_size=32) # Apply to cropped
        # end_time = time.time()
        # print(f"Simplification finished in {end_time - start_time:.2f} seconds.")
        # simplified_output_filename = f"{name_part}_simplified{ext_part}"
        # simplified_output_path = os.path.join(processed_folder, simplified_output_filename)
        # print(f"Saving simplified image to: {simplified_output_path}")
        # save_success = cv2.imwrite(simplified_output_path, simplified_cropped_img)
        # if not save_success:
        #     print(f"Error: Failed to save simplified image to {simplified_output_path}")


    except Exception as e:
        print(f"An error occurred processing {file_path}:")
        traceback.print_exc() # Print detailed traceback

print("\n--- All files processed. ---") 