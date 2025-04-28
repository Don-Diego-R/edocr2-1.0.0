import sys
import os
import shutil
import cv2
import traceback

# Import necessary components from the package
from cad_overlay.screenshot import capture_screenshot
from cad_overlay.detection import find_drawing_contour_box
from cad_overlay.ui import OverlayApplication

# --- Main Execution --- 
def main():
    print("=== CAD Drawing Detector - Floating Overlay Mode ===")
    
    # Take initial screenshot
    try:
        screenshot = capture_screenshot()
    except Exception as e:
        print(f"Error capturing initial screenshot: {e}")
        traceback.print_exc()
        return

    # Create/clear debug folder
    debug_folder = "screenshot_debug"
    try:
        if os.path.exists(debug_folder):
            shutil.rmtree(debug_folder)
        os.makedirs(debug_folder, exist_ok=True)
        
        # Save the initial screenshot for reference
        screenshot_path = os.path.join(debug_folder, "initial_screenshot.png")
        cv2.imwrite(screenshot_path, screenshot)
        print(f"Initial screenshot saved to: {screenshot_path}")
    except Exception as e:
        print(f"Error handling debug folder or saving initial screenshot: {e}")
        # Continue without debug saving if there's an issue
        debug_folder = None 

    # Find the initial drawing contour
    try:
        initial_rect = find_drawing_contour_box(screenshot, debug_folder=debug_folder)
        print(f"Initial detected box: x={initial_rect[0]}, y={initial_rect[1]}, w={initial_rect[2]}, h={initial_rect[3]}")
    except Exception as e:
        print(f"Error during initial contour detection: {e}")
        traceback.print_exc()
        # Use default full screen if detection fails
        height, width = screenshot.shape[:2]
        initial_rect = (0, 0, width, height)
        print("Falling back to full screen rectangle.")

    # Instantiate and run the overlay application
    try:
        # Pass the functions directly to the OverlayApplication
        overlay_app = OverlayApplication(
            initial_rect=initial_rect, 
            screenshot_func=capture_screenshot, 
            detection_func=find_drawing_contour_box
        )
        overlay_app.run() # This starts the Qt event loop
    except Exception as e:
        print(f"Error running the overlay application: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 