import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush, QFont

# --- Floating Rectangle Overlay --- 
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
        # Use primaryScreen() for better compatibility
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

# --- Control Panel --- 
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
    
    # --- Dragging Logic --- 
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Check if the press was on the title bar area
            if event.pos().y() <= 25: 
                self.dragging = True
                self.offset = event.pos()
            else:
                super().mousePressEvent(event) # Pass event to buttons if not on title bar
            
    def mouseMoveEvent(self, event):
        if self.dragging and event.buttons() & Qt.LeftButton:
            self.move(self.mapToGlobal(event.pos() - self.offset))
        else:
            super().mouseMoveEvent(event)
            
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False
        super().mouseReleaseEvent(event)

# --- Application Management --- 
class OverlayApplication:
    """Manages the lifecycle of the overlay and control panel"""
    _instance = None # For singleton QApplication

    @staticmethod
    def get_qapp_instance():
        """Gets or creates the singleton QApplication instance."""
        if QApplication.instance():
            return QApplication.instance()
        else:
            OverlayApplication._instance = QApplication(sys.argv)
            return OverlayApplication._instance

    def __init__(self, initial_rect, screenshot_func, detection_func):
        self.app = OverlayApplication.get_qapp_instance()
        self.initial_rect = initial_rect
        self.capture_screenshot = screenshot_func
        self.find_drawing_contour_box = detection_func

        # Create the UI elements
        self.rect_overlay = FloatingRectangle(self.initial_rect)
        self.control_panel = ControlPanel()
        self.initial_control_opacity = self.control_panel.windowOpacity() 

        # Setup timer and refresh logic
        self.update_timer = QTimer()
        self.update_interval = 1000  # ms
        self.is_refreshing = False

        # Connect signals
        self.control_panel.refreshRequested.connect(lambda: self.trigger_refresh(enable_debug=True))
        self.control_panel.toggleInfoRequested.connect(self.rect_overlay.toggle_info)
        self.control_panel.exitRequested.connect(self.exit_app)
        self.update_timer.timeout.connect(lambda: self.trigger_refresh(enable_debug=False))

    def perform_refresh(self, enable_debug=False):
        """The core logic for taking screenshot and updating"""
        # Now take the screenshot
        self.capture_screenshot.verbose = False
        screenshot = self.capture_screenshot()
        
        # Find contours
        debug_folder_path = "screenshot_debug" if enable_debug else None 
        new_rect = self.find_drawing_contour_box(screenshot, debug_folder=debug_folder_path)
        
        # Restore opacity BEFORE updating/showing
        self.rect_overlay.setWindowOpacity(1.0)
        self.control_panel.setWindowOpacity(self.initial_control_opacity)
        
        # Update and ensure overlay is visible
        self.rect_overlay.update_rect(new_rect)
        if not self.rect_overlay.isVisible():
             self.rect_overlay.show()
        
        self.is_refreshing = False

    def trigger_refresh(self, enable_debug=False):
        """Makes overlays transparent, waits, then calls perform_refresh"""
        if self.is_refreshing:
            return
        self.is_refreshing = True
        
        # Make overlays transparent
        self.rect_overlay.setWindowOpacity(0.0)
        self.control_panel.setWindowOpacity(0.0)
        
        # Wait and refresh
        QTimer.singleShot(100, lambda: self.perform_refresh(enable_debug))

    def run(self):
        """Starts the timer and the Qt application event loop."""
        print("Starting overlay update timer...")
        self.update_timer.start(self.update_interval)
        print("Starting Qt event loop...")
        exit_code = self.app.exec_()
        print("Qt event loop finished.")
        sys.exit(exit_code)

    def exit_app(self):
        """Stops timer and closes windows gracefully."""
        print("Exiting overlay application...")
        self.update_timer.stop()
        try:
            self.rect_overlay.close()
        except RuntimeError: # Window might already be deleted
            pass
        try:
            self.control_panel.close()
        except RuntimeError:
            pass
        # Ensure the main application event loop terminates
        self.app.quit()
