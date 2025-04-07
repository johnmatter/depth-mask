import cv2
import numpy as np
import coremltools as ct
import PIL.Image
import sys
import mediapipe as mp
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QSlider, QCheckBox, QPushButton, QSizePolicy,
                            QColorDialog)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QFont, QColor, QPen

# Looks like pyvirtualcam is not working with OBS Studio on macOS for the time being
# https://github.com/letmaik/pyvirtualcam/issues/111#issuecomment-1763398540
# https://github.com/obsproject/obs-studio/issues/9680#issuecomment-1765758754

# Download the model from Hugging Face
# https://huggingface.co/apple/coreml-depth-anything-v2-small
# huggingface-cli download \
#   --local-dir models --local-dir-use-symlinks False \
#   apple/coreml-depth-anything-v2-small

class DepthCameraApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize variables
        self.threshold = 50
        self.mirror_enabled = True
        self.show_all_panels = True
        self.normal_geometry = None
        self.green_screen_color = QColor(0, 0, 0)
        self.green_screen_mode = True
        self.use_pose_refinement = True  # Enable pose refinement
        self.show_pose_overlay = True    # Show pose landmarks
        
        # Set up the UI
        self.setWindowTitle("Depth Camera")
        self.setGeometry(100, 100, 900, 700)
        
        # Main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        self.main_layout = QVBoxLayout(central_widget)
        
        # Controls layout
        self.controls_layout = QHBoxLayout()
        
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
        
        # Green screen mode toggle
        self.green_screen_toggle = QCheckBox("Green Screen")
        self.green_screen_toggle.setStyleSheet("font-family: Arial; font-size: 14px; font-weight: bold;")
        self.green_screen_toggle.setChecked(self.green_screen_mode)
        self.green_screen_toggle.stateChanged.connect(self.toggle_green_screen)
        
        # Pose refinement toggle
        self.pose_toggle = QCheckBox("Pose Refinement")
        self.pose_toggle.setStyleSheet("font-family: Arial; font-size: 14px; font-weight: bold;")
        self.pose_toggle.setChecked(self.use_pose_refinement)
        self.pose_toggle.stateChanged.connect(self.toggle_pose_refinement)
        
        # Pose overlay toggle
        self.pose_overlay_toggle = QCheckBox("Show Pose")
        self.pose_overlay_toggle.setStyleSheet("font-family: Arial; font-size: 14px; font-weight: bold;")
        self.pose_overlay_toggle.setChecked(self.show_pose_overlay)
        self.pose_overlay_toggle.stateChanged.connect(self.toggle_pose_overlay)
        
        # Color picker button
        self.color_button = QPushButton("Select Color")
        self.color_button.setStyleSheet("font-family: Arial; font-size: 14px; font-weight: bold;")
        self.color_button.clicked.connect(self.choose_color)
        
        # Set initial color button background
        self.update_color_button()
        
        # Add controls to layout
        self.controls_layout.addWidget(threshold_label)
        self.controls_layout.addWidget(self.threshold_slider)
        self.controls_layout.addWidget(self.threshold_value_label)
        self.controls_layout.addWidget(self.mirror_toggle)
        self.controls_layout.addWidget(self.green_screen_toggle)
        self.controls_layout.addWidget(self.pose_toggle)
        self.controls_layout.addWidget(self.pose_overlay_toggle)
        self.controls_layout.addWidget(self.color_button)
        
        # Display label for the camera feed
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        
        # Add widgets to main layout
        self.main_layout.addLayout(self.controls_layout)
        self.main_layout.addWidget(self.display_label)
        
        # Load Core ML model
        self.model_path = "models/DepthAnythingV2SmallF16.mlpackage"
        self.model = ct.models.MLModel(self.model_path)
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=True,  # Important: generates segmentation mask
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Set up timer for frame updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)  # Update at approximately 30 FPS
        
        # Store initial window size
        self.clean_view_size = (900, 700)  # Default size for clean view
    
    def update_color_button(self):
        """Update the color button's background to show the current color"""
        color = self.green_screen_color
        style = f"background-color: rgb({color.red()}, {color.green()}, {color.blue()}); "
        
        # Adjust text color for better contrast
        if color.lightness() > 128:
            style += "color: black; "
        else:
            style += "color: white; "
        
        style += "font-family: Arial; font-size: 14px; font-weight: bold;"
        self.color_button.setStyleSheet(style)
    
    def choose_color(self):
        """Open color picker dialog and update the green screen color"""
        color = QColorDialog.getColor(self.green_screen_color, self, "Select Green Screen Color")
        if color.isValid():
            self.green_screen_color = color
            self.update_color_button()
    
    def toggle_green_screen(self, state):
        """Toggle between green screen and transparency modes"""
        self.green_screen_mode = (state == Qt.Checked)
    
    def toggle_pose_refinement(self, state):
        """Toggle pose refinement"""
        self.use_pose_refinement = (state == Qt.Checked)
    
    def toggle_pose_overlay(self, state):
        """Toggle pose overlay display"""
        self.show_pose_overlay = (state == Qt.Checked)
    
    def keyPressEvent(self, event):
        # Toggle view mode when spacebar is pressed
        if event.key() == Qt.Key_Space:
            self.show_all_panels = not self.show_all_panels
            self.toggle_view_mode()
        # Allow Escape key to exit fullscreen mode
        elif event.key() == Qt.Key_Escape and not self.show_all_panels:
            self.show_all_panels = True
            self.toggle_view_mode()
        
        # Pass event to parent class for default handling of other keys
        super().keyPressEvent(event)
    
    def toggle_view_mode(self):
        if self.show_all_panels:
            # Switch to normal view
            # Restore window decorations
            self.setWindowFlags(Qt.Window)
            # Show controls
            for i in range(self.controls_layout.count()):
                item = self.controls_layout.itemAt(i)
                if item.widget():
                    item.widget().show()
            # Set normal layout margins
            self.main_layout.setContentsMargins(11, 11, 11, 11)
            # Restore window title
            self.setWindowTitle("Depth Camera")
            # Restore normal geometry if we have it stored
            if self.normal_geometry:
                self.setGeometry(self.normal_geometry)
            else:
                self.setGeometry(100, 100, 900, 700)
            
            # Allow the display label to resize with contents in normal mode
            self.display_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            
        else:
            # Switch to clean view
            # Store current geometry before switching
            self.normal_geometry = self.geometry()
            # Store fixed size for clean view
            self.clean_view_size = (self.width(), self.height())
            # Remove window decorations
            self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
            # Hide controls
            for i in range(self.controls_layout.count()):
                item = self.controls_layout.itemAt(i)
                if item.widget():
                    item.widget().hide()
            # Remove layout margins
            self.main_layout.setContentsMargins(0, 0, 0, 0)
            # Clear window title
            self.setWindowTitle("")
            
            # Fix the size policy to prevent auto-resizing
            self.display_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            self.display_label.setFixedSize(self.clean_view_size[0], self.clean_view_size[1])
        
        # Need to show window again after changing flags
        self.show()
    
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
        
        # Copy original frame for pose detection
        rgb_frame_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe Pose
        pose_results = None
        pose_mask = None
        if self.use_pose_refinement:
            pose_results = self.pose.process(rgb_frame_full)
            
            # If pose was detected, create a mask from segmentation
            if pose_results.segmentation_mask is not None:
                # Get the segmentation mask from MediaPipe
                pose_mask = pose_results.segmentation_mask
                
                # Resize to match frame dimensions
                pose_mask = cv2.resize(
                    pose_mask, 
                    (frame_width, frame_height)
                )
                
                # Threshold and convert to binary mask
                _, pose_mask = cv2.threshold(
                    (pose_mask * 255).astype(np.uint8),
                    127, 255, cv2.THRESH_BINARY
                )
        
        # Resize to the model's expected input size for depth estimation
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
        
        # Combine depth and pose masks if available
        final_mask = alpha_mask_resized
        
        if self.use_pose_refinement and pose_mask is not None:
            # Combine the masks:
            # 1. Start with depth mask
            # 2. Refine with pose segmentation mask
            
            # Use logical OR to combine the masks
            final_mask = cv2.bitwise_or(alpha_mask_resized, pose_mask)
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
        
        # Create the masked frame
        if self.green_screen_mode:
            # Create a solid color image for green screen
            # Use pure BGR values for OBS compatibility
            color_bgr = (
                self.green_screen_color.blue(),
                self.green_screen_color.green(), 
                self.green_screen_color.red()
            )
            
            # IMPORTANT: Invert the mask - we want the BACKGROUND to be green, not the subject
            # The depth model gives us a mask where subject is white (255) and background is black (0)
            # We need to invert this for proper chroma keying
            inv_mask_resized = 255 - final_mask
            
            # Apply slight blur to mask to reduce edge artifacts
            mask_smoothed = cv2.GaussianBlur(inv_mask_resized, (5, 5), 0)
            
            # Convert to 3-channel for blending (scaled to 0.0-1.0)
            mask_3ch = cv2.cvtColor(mask_smoothed, cv2.COLOR_GRAY2BGR) / 255.0
            
            # Create the green screen backdrop
            green_screen = np.full_like(frame, color_bgr, dtype=np.uint8)
            
            # Blend original frame with green screen based on mask
            # This puts the green color in the BACKGROUND (where mask is 1.0)
            # and keeps the subject's original colors (where mask is 0.0)
            frame_with_mask = cv2.multiply(1.0 - mask_3ch, frame.astype(float)).astype(np.uint8) + \
                               cv2.multiply(mask_3ch, green_screen.astype(float)).astype(np.uint8)
        else:
            # Apply the alpha mask to the original frame (original transparency mode)
            frame_with_mask = cv2.bitwise_and(frame, frame, mask=final_mask)
        
        # Draw pose landmarks if enabled
        if self.show_pose_overlay and pose_results and pose_results.pose_landmarks:
            frame_with_landmarks = frame_with_mask.copy()
            self.mp_drawing.draw_landmarks(
                frame_with_landmarks,
                pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            frame_with_mask = frame_with_landmarks
        
        # Define a common size for all display images
        display_size = (400, 300)
        
        # Display based on the current view mode
        if self.show_all_panels:
            # Resize all images to the common display size
            display_original = cv2.resize(frame, display_size)
            
            # Convert depth map to BGR for display
            depth_color = cv2.cvtColor(depth_map_visualization, cv2.COLOR_GRAY2BGR)
            display_depth = cv2.resize(depth_color, display_size)
            
            # Convert alpha mask to BGR for display
            alpha_color = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
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
            labels = ["Original", "Depth Map", "Combined Mask", "Masked Result"]
            positions = [(20, 30), (display_size[0] + 20, 30), 
                        (20, display_size[1] + 30), (display_size[0] + 20, display_size[1] + 30)]
            
            for label, pos in zip(labels, positions):
                painter.drawText(pos[0], pos[1], label)
            
            # End painting
            painter.end()
        else:
            # Show only the masked output in full window size
            # Use the fixed size we stored for clean view
            display_masked = cv2.resize(frame_with_mask, (self.clean_view_size[0], self.clean_view_size[1]))
            
            # Convert OpenCV image to QImage for display in PyQt
            h, w, c = display_masked.shape
            bytes_per_line = c * w
            qt_image = QImage(display_masked.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            
            # Create a pixmap from the QImage
            pixmap = QPixmap.fromImage(qt_image)
            
            # No text or UI elements in clean mode - just the image
        
        # Display the image
        self.display_label.setPixmap(pixmap)
    
    def closeEvent(self, event):
        self.cap.release()
        self.pose.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DepthCameraApp()
    window.show()
    sys.exit(app.exec_())
