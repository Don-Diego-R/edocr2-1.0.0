import numpy as np
import cv2
import pyautogui

# --- Capture Screenshot ---
def capture_screenshot():
    """
    Captures a screenshot of the entire screen.
    
    Returns:
        A BGR image of the screenshot.
    """
    # Don't log every screenshot during live updates
    # Use a default verbose=True if the attribute hasn't been set
    if getattr(capture_screenshot, 'verbose', True): 
        print("Capturing screenshot...")
        
    screenshot = pyautogui.screenshot()
    # Convert PIL Image to OpenCV format (BGR)
    screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
    if getattr(capture_screenshot, 'verbose', True):
        print(f"Screenshot captured: {screenshot_cv.shape}")
        
    return screenshot_cv

# Set default verbose attribute
capture_screenshot.verbose = True
