import cv2, os
import numpy as np
from edocr2 import tools
import time
import glob # Import glob for file searching
import traceback # For detailed error printing
import math # For Euclidean distance
import shutil # For cleaning debug folders
import pyautogui # For taking screenshots
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush, QFont

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

# --- Helper Function: Find Drawing Box via Contours ---
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
        # Check if exists before clearing - slightly safer
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

# --- NEW Function: Capture Screenshot ---
def capture_screenshot():
    """
    Captures a screenshot of the entire screen.
    
    Returns:
        A BGR image of the screenshot.
    """
    # Don't log every screenshot during live updates
    if getattr(capture_screenshot, 'verbose', True):
        print("Capturing screenshot...")
    screenshot = pyautogui.screenshot()
    # Convert PIL Image to OpenCV format (BGR)
    screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    if getattr(capture_screenshot, 'verbose', True):
        print(f"Screenshot captured: {screenshot_cv.shape}")
    return screenshot_cv

# Set verbose attribute
capture_screenshot.verbose = True

# --- NEW Function: Display Overlay with PyQt5 ---
class FloatingRectangle(QWidget):
    """
    A transparent widget that draws just a rectangle and crosshair
    with transparent background to allow clicking through.
    """
    def __init__(self, rect_coords):
        super().__init__()
        
        # Set initial rectangle coordinates
        self.x, self.y, self.w, self.h = rect_coords
        
        # Configure the widget
        self.setWindowFlags(
            Qt.FramelessWindowHint | 
            Qt.WindowStaysOnTopHint |
            Qt.Tool  # Tool windows don't show in taskbar
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)  # Let mouse events pass through
        
        # Set the window to full screen size
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(0, 0, screen.width(), screen.height())
        
        # Set text info
        self.show_info = True
        self.info_text = f"x={self.x}, y={self.y}, w={self.w}, h={self.h}"
        
        # Show the widget
        self.show()
    
    def update_rect(self, rect_coords):
        """Update the rectangle coordinates"""
        self.x, self.y, self.w, self.h = rect_coords
        self.info_text = f"x={self.x}, y={self.y}, w={self.w}, h={self.h}"
        self.update()  # Trigger repaint
    
    def toggle_info(self):
        """Toggle display of information text"""
        self.show_info = not self.show_info
        self.update()
    
    def paintEvent(self, event):
        """Draw the rectangle and crosshair on the widget"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw rectangle with red, thick border
        pen = QPen(QColor(255, 0, 0))  # Red
        pen.setWidth(3)
        painter.setPen(pen)
        painter.drawRect(self.x, self.y, self.w, self.h)
        
        # Draw crosshair at center
        center_x = self.x + self.w // 2
        center_y = self.y + self.h // 2
        crosshair_size = 15
        
        pen = QPen(QColor(255, 255, 0))  # Yellow
        pen.setWidth(2)
        painter.setPen(pen)
        
        # Horizontal line
        painter.drawLine(
            center_x - crosshair_size, center_y,
            center_x + crosshair_size, center_y
        )
        
        # Vertical line
        painter.drawLine(
            center_x, center_y - crosshair_size,
            center_x, center_y + crosshair_size
        )
        
        # Draw info text if enabled
        if self.show_info:
            # Draw info background
            painter.setBrush(QBrush(QColor(0, 0, 0, 180)))  # Semi-transparent black
            painter.setPen(Qt.NoPen)
            painter.drawRect(10, 10, 250, 30)
            
            # Draw text
            painter.setPen(QPen(QColor(255, 255, 255)))  # White
            painter.setFont(QFont('Arial', 10))
            painter.drawText(15, 30, self.info_text)

class ControlPanel(QWidget):
    """Floating control panel for the overlay"""
    refreshRequested = pyqtSignal()
    toggleInfoRequested = pyqtSignal()
    exitRequested = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        
        # Configure the window
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool
        )
        self.setStyleSheet("background-color: #333333;")
        
        # Create layout
        layout = QVBoxLayout()
        
        # Add title bar
        title_bar = QWidget()
        title_bar.setStyleSheet("background-color: #111111;")
        title_bar.setFixedHeight(25)
        title_bar_layout = QHBoxLayout(title_bar)
        title_bar_layout.setContentsMargins(5, 0, 5, 0)
        
        title_label = QLabel("CAD Overlay Controls")
        title_label.setStyleSheet("color: white; font-weight: bold;")
        title_bar_layout.addWidget(title_label)
        
        # Set up the buttons
        self.refresh_btn = QPushButton("Refresh Detection")
        self.refresh_btn.setStyleSheet(
            "background-color: #2196F3; color: white; font-weight: bold; padding: 5px;"
        )
        self.refresh_btn.clicked.connect(self.refreshRequested.emit)
        
        self.toggle_info_btn = QPushButton("Toggle Info Display")
        self.toggle_info_btn.setStyleSheet(
            "background-color: #FFC107; color: black; font-weight: bold; padding: 5px;"
        )
        self.toggle_info_btn.clicked.connect(self.toggleInfoRequested.emit)
        
        self.exit_btn = QPushButton("Exit")
        self.exit_btn.setStyleSheet(
            "background-color: #F44336; color: white; font-weight: bold; padding: 5px;"
        )
        self.exit_btn.clicked.connect(self.exitRequested.emit)
        
        # Add widgets to layout
        layout.addWidget(title_bar)
        layout.addWidget(self.refresh_btn)
        layout.addWidget(self.toggle_info_btn)
        layout.addWidget(self.exit_btn)
        
        # Set layout
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        self.setLayout(layout)
        
        # Size and position
        self.resize(200, 150)
        screen = QApplication.primaryScreen().geometry()
        self.move(screen.width() - 220, 50)
        
        # Variables for dragging
        self.dragging = False
        self.offset = None
        
        # Show the widget
        self.show()
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.offset = event.pos()
            
    def mouseMoveEvent(self, event):
        if self.dragging and event.buttons() & Qt.LeftButton:
            self.move(self.mapToGlobal(event.pos() - self.offset))
            
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False

def display_contour_overlay(initial_rect):
    """
    Display a floating overlay with just the rectangle, crosshair, and control buttons.
    
    Args:
        initial_rect: Tuple (x, y, w, h) of the detected contour
    """
    app = QApplication(sys.argv)
    
    # Create the rectangle overlay (transparent, click-through)
    rect_overlay = FloatingRectangle(initial_rect)
    
    # Create the control panel (solid, draggable)
    control_panel = ControlPanel()
    # Store initial control panel opacity (it might not be 1.0)
    initial_control_opacity = control_panel.windowOpacity() 
    
    # Setup timer for regular updates
    update_timer = QTimer()
    update_interval = 1000  # ms
    
    # --- Refined Refresh Logic ---
    is_refreshing = False # Flag to prevent overlapping refreshes
    
    def perform_refresh(enable_debug=False):
        """The core logic for taking screenshot and updating"""
        nonlocal is_refreshing
        
        # Now take the screenshot
        capture_screenshot.verbose = False # Keep console clean during live updates
        screenshot = capture_screenshot()
        
        # Find contours using the original function (pass debug folder only if needed)
        debug_folder_path = "screenshot_debug" if enable_debug else None
        new_rect = find_drawing_contour_box(screenshot, debug_folder=debug_folder_path)
        
        # Restore opacity BEFORE updating/showing
        rect_overlay.setWindowOpacity(1.0) # Should be fully opaque for drawing
        control_panel.setWindowOpacity(initial_control_opacity) # Restore original
        
        # Update and ensure overlay is visible
        rect_overlay.update_rect(new_rect)
        if not rect_overlay.isVisible():
             rect_overlay.show()
        
        is_refreshing = False # Allow next refresh
    
    def trigger_refresh(enable_debug=False):
        """Makes overlays transparent, waits, then calls perform_refresh"""
        nonlocal is_refreshing
        if is_refreshing:
            return # Don't start a new refresh if one is already in progress
        
        is_refreshing = True
        
        # Make the overlays fully transparent
        rect_overlay.setWindowOpacity(0.0)
        control_panel.setWindowOpacity(0.0)
        
        # Wait a short moment for the screen to update, then refresh
        # Pass enable_debug flag to the final step
        QTimer.singleShot(100, lambda: perform_refresh(enable_debug)) # Slightly shorter delay
    
    # --- End Refined Refresh Logic ---
    
    def exit_app():
        update_timer.stop()
        rect_overlay.close()
        control_panel.close()
        app.quit()
    
    # Connect control panel signals
    # The "Refresh" button now uses the same logic as the timer, but enables debug
    control_panel.refreshRequested.connect(lambda: trigger_refresh(enable_debug=True))
    control_panel.toggleInfoRequested.connect(rect_overlay.toggle_info)
    control_panel.exitRequested.connect(exit_app)
    
    # Start update timer for live updates (debug disabled for timer)
    update_timer.timeout.connect(lambda: trigger_refresh(enable_debug=False))
    update_timer.start(update_interval)
    
    # Start the app
    sys.exit(app.exec_())

# --- Main Script Entry Point ---
def main():
    print("=== CAD Drawing Detector - Floating Overlay Mode ===")
    
    # Take a screenshot
    screenshot = capture_screenshot()
    
    # Create debug folder (clear it on startup)
    debug_folder = "screenshot_debug"
    if os.path.exists(debug_folder):
        shutil.rmtree(debug_folder)
    os.makedirs(debug_folder, exist_ok=True)
    
    # Save the screenshot for reference
    screenshot_path = os.path.join(debug_folder, "screenshot.png")
    cv2.imwrite(screenshot_path, screenshot)
    print(f"Screenshot saved to: {screenshot_path}")
    
    # Find the drawing contour initially with debug enabled
    initial_rect = find_drawing_contour_box(screenshot, debug_folder=debug_folder)
    print(f"Detected contour box: x={initial_rect[0]}, y={initial_rect[1]}, w={initial_rect[2]}, h={initial_rect[3]}")
    
    # Display overlay
    display_contour_overlay(initial_rect)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred:")
        traceback.print_exc() 