import cv2
import numpy as np
import coremltools as ct
import PIL.Image
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QSlider, QCheckBox, QPushButton)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QFont, QColor, QPen

class DepthCameraApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize variables
        self.threshold = 50
        self.mirror_enabled = True
        
        # Set up the UI
        self.setWindowTitle("Depth Camera")
        self.setGeometry(100, 100, 900, 700)
        
        # Main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Controls layout
        controls_layout = QHBoxLayout()
        
        # Threshold slider
        threshold_label = QLabel("Threshold:")
        threshold_label.setStyleSheet("font-family: Arial; font-size: 14px; font-weight: bold;")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(self.threshold)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        self.threshold_value_label = QLabel(f"{self.threshold/100:.2f}")
        self.threshold_value_label.setStyleSheet("font-family: Arial; font-size: 14px;")
        
        # Mirror toggle checkbox
        self.mirror_toggle = QCheckBox("Mirror Horizontally")
        self.mirror_toggle.setStyleSheet("font-family: Arial; font-size: 14px; font-weight: bold;")
        self.mirror_toggle.setChecked(self.mirror_enabled)
        self.mirror_toggle.stateChanged.connect(self.toggle_mirror)
        
        # Add controls to layout
        controls_layout.addWidget(threshold_label)
        controls_layout.addWidget(self.threshold_slider)
        controls_layout.addWidget(self.threshold_value_label)
        controls_layout.addWidget(self.mirror_toggle)
        
        # Display label for the camera feed
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        
        # Add widgets to main layout
        main_layout.addLayout(controls_layout)
        main_layout.addWidget(self.display_label)
        
        # Load Core ML model
        self.model_path = "models/DepthAnythingV2SmallF16.mlpackage"
        self.model = ct.models.MLModel(self.model_path)
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Set up timer for frame updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)  # Update at approximately 30 FPS
    
    def update_threshold(self, value):
        self.threshold = value
        self.threshold_value_label.setText(f"{value/100:.2f}")
    
    def toggle_mirror(self, state):
        self.mirror_enabled = (state == Qt.Checked)
    
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Apply horizontal mirroring if enabled
        if self.mirror_enabled:
            frame = cv2.flip(frame, 1)  # 1 means horizontal flip
        
        # Get original frame dimensions
        frame_height, frame_width = frame.shape[:2]
        
        # Resize to the model's expected input size
        # Core ML model expects exactly 518×392
        resized_frame = cv2.resize(frame, (518, 392))
        
        # Convert to BGR to RGB (Core ML typically expects RGB)
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image - Core ML expects PIL Image directly
        pil_image = PIL.Image.fromarray(rgb_frame)
        
        # Make prediction with PIL Image directly
        prediction = self.model.predict({"image": pil_image})
        
        # Extract depth map from prediction
        depth_map = prediction["depth"]
        
        # Convert depth map from PIL Image to NumPy array
        depth_array = np.array(depth_map)
        
        # Normalize for visualization
        depth_array_normalized = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min())
        depth_map_visualization = (depth_array_normalized * 255).astype(np.uint8)
        
        # Get current threshold value (convert from percentage to 0-1 range)
        current_threshold = self.threshold / 100.0
        
        # Extract alpha mask with adjustable thresholding
        alpha_mask = (depth_array > current_threshold).astype(np.uint8) * 255
        
        # Resize alpha mask to match original frame dimensions
        alpha_mask_resized = cv2.resize(alpha_mask, (frame_width, frame_height))
        
        # Apply the alpha mask to the original frame
        frame_with_mask = cv2.bitwise_and(frame, frame, mask=alpha_mask_resized)
        
        # Define a common size for all display images
        display_size = (400, 300)
        
        # Resize all images to the common display size
        display_original = cv2.resize(frame, display_size)
        
        # Convert depth map to BGR for display
        depth_color = cv2.cvtColor(depth_map_visualization, cv2.COLOR_GRAY2BGR)
        display_depth = cv2.resize(depth_color, display_size)
        
        # Convert alpha mask to BGR for display
        alpha_color = cv2.cvtColor(alpha_mask, cv2.COLOR_GRAY2BGR)
        display_alpha = cv2.resize(alpha_color, display_size)
        
        # Resize masked frame
        display_masked = cv2.resize(frame_with_mask, display_size)
        
        # Create a 2x2 grid
        top_row = np.hstack((display_original, display_depth))
        bottom_row = np.hstack((display_alpha, display_masked))
        combined_display = np.vstack((top_row, bottom_row))
        
        # Convert OpenCV image to QImage for display in PyQt
        h, w, c = combined_display.shape
        bytes_per_line = c * w
        qt_image = QImage(combined_display.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        
        # Create a pixmap from the QImage
        pixmap = QPixmap.fromImage(qt_image)
        
        # Create a painter to draw on the pixmap
        painter = QPainter(pixmap)
        
        # Set up font and color for text
        font = QFont("Arial", 16, QFont.Bold)
        painter.setFont(font)
        painter.setPen(QPen(QColor(0, 200, 0)))
        
        # Draw labels on each quadrant
        labels = ["Original", "Depth Map", "Alpha Mask", "Masked Result"]
        positions = [(20, 30), (display_size[0] + 20, 30), 
                    (20, display_size[1] + 30), (display_size[0] + 20, display_size[1] + 30)]
        
        for label, pos in zip(labels, positions):
            painter.drawText(pos[0], pos[1], label)
        
        # End painting
        painter.end()
        
        # Display the combined image
        self.display_label.setPixmap(pixmap)
    
    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DepthCameraApp()
    window.show()
    sys.exit(app.exec_())
