import cv2
import numpy as np
import winsound
import time
import json
import threading
import pygetwindow as gw
import os
from tkinter import messagebox
from datetime import datetime

# Open a connection to the camera (0 represents the default camera)
cap = cv2.VideoCapture(0)

# Check if config.json exists, otherwise show a message box and exit
config_file_path = 'config.json'
if not os.path.exists(config_file_path):
    # Display a message box
    messagebox.showerror("Error", "'config.json' file not found. Please create the configuration file.")
    exit()

# Add a check for the current date
current_date = datetime.now()
expiration_date = datetime(2024, 1, 31)

if current_date > expiration_date:
    # Display a message box and exit if the current date is past the expiration date
    messagebox.showinfo("Info", "This program has expired. Please contact the developer.")
    exit()

# Read parameters from the configuration file
with open(config_file_path, 'r') as config_file:
    config = json.load(config_file)

# Extract parameters from the config
area_threshold = config.get("area_threshold", 1000)
detect_duration = config.get("detect_duration", 3)
last_beep_time_threshold = config.get("last_beep_time", 5)
gaussian_blur_kernel_size = config.get("gaussian_blur_kernel_size", 1)
gaussian_blur_sigma = config.get("gaussian_blur_sigma", 10)
minimize_chrome = config.get("minimize_chrome", False)
minimize_edge = config.get("minimize_edge", False)

# Create a background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, detectShadows=False)

# Initialize ROI variables
roi_defined = False
top_left_pt, bottom_right_pt = (-1, -1), (-1, -1)

# Initialize variables for beeping control
last_beep_time = time.time()
motion_detected = False
continuous_detection_start_time = 0

# Initialize selected area window
roi_window_name = 'Selected Area'
cv2.namedWindow(roi_window_name)
cv2.setWindowProperty(roi_window_name, cv2.WND_PROP_TOPMOST, 1)  # Set window always on top

# Function to set the area of interest using the mouse
def set_roi(event, x, y, flags, param):
    global top_left_pt, bottom_right_pt, roi_defined

    if event == cv2.EVENT_LBUTTONDOWN:
        print("Setting top-left point of ROI:", (x, y))
        top_left_pt = (x, y)
        roi_defined = False

    elif event == cv2.EVENT_LBUTTONUP:
        print("Setting bottom-right point of ROI:", (x, y))
        bottom_right_pt = (x, y)

        # Check for a valid ROI (top-left is to the left and above bottom-right)
        if top_left_pt[0] < bottom_right_pt[0] and top_left_pt[1] < bottom_right_pt[1]:
            roi_defined = True
            cv2.rectangle(frame, top_left_pt, bottom_right_pt, (0, 255, 0), 2)
        else:
            print("Error: Invalid ROI. Please select a valid region.")

# Create a window and set the callback function for mouse events
cv2.namedWindow('Moving Person Detection')
cv2.setMouseCallback('Moving Person Detection', set_roi)

# Function to display the selected area in a new window
def show_selected_area(frame, top_left, bottom_right):
    if roi_defined:
        roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        cv2.imshow(roi_window_name, roi)
    else:
        cv2.destroyWindow(roi_window_name)

# Function to play beep sound in a separate thread
def play_beep_thread(beep_duration):
    threading.Thread(target=winsound.Beep, args=(1000, beep_duration)).start()

# Function to minimize Chrome and Microsoft Edge windows
def minimize_windows():
    if minimize_chrome:
        # Minimize Chrome window
        chrome_windows = gw.getWindowsWithTitle("Google Chrome")
        if chrome_windows:
            chrome_window = chrome_windows[0]  # Assuming the first Chrome window
            chrome_window.minimize()

    if minimize_edge:
        # Minimize Microsoft Edge window
        edge_windows = [win for win in gw.getAllTitles() if "Edge" in win]
        for edge_window in edge_windows:
            try:
                edge_window_obj = gw.getWindowsWithTitle(edge_window)[0]
                edge_window_obj.minimize()
            except IndexError:
                pass

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Apply Gaussian blur
    frame = cv2.GaussianBlur(frame, (gaussian_blur_kernel_size, gaussian_blur_kernel_size), gaussian_blur_sigma)
    # Check if the frame is not empty
    if not ret or frame is None:
        print("Error: Unable to capture frame")
        break

    # Display the frame without background subtraction
    cv2.imshow('Moving Person Detection', frame)

    # Define ROI if not already defined
    if roi_defined:
        # Apply background subtraction only within the ROI
        roi = frame[top_left_pt[1]:bottom_right_pt[1], top_left_pt[0]:bottom_right_pt[0]]

        # Check if the ROI is not empty and has valid dimensions
        if roi is not None and roi.shape[0] > 0 and roi.shape[1] > 0:
            fg_mask_roi = bg_subtractor.apply(roi)

            # Ensure that the mask and ROI have the same dimensions
            fg_mask_roi = cv2.resize(fg_mask_roi, (roi.shape[1], roi.shape[0]))

            # Apply morphological operations to remove noise within the ROI
            fg_mask_roi = cv2.morphologyEx(fg_mask_roi, cv2.MORPH_OPEN, kernel=np.ones((5, 5), np.uint8))

            # Create a mask of the same size as the original frame
            mask = np.zeros_like(frame)

            # Place the foreground mask into the corresponding ROI of the mask
            mask[top_left_pt[1]:bottom_right_pt[1], top_left_pt[0]:bottom_right_pt[0]] = \
                cv2.bitwise_and(roi, roi, mask=fg_mask_roi)

            # Add the mask to the original frame
            frame = cv2.add(frame, mask)
            cv2.rectangle(frame, top_left_pt, bottom_right_pt, (0, 255, 0), 2)

            # Draw bounding boxes around moving objects (persons) in the selected ROI
            contours, _ = cv2.findContours(fg_mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > area_threshold:  # Use the area threshold from the config
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame[top_left_pt[1]:bottom_right_pt[1], top_left_pt[0]:bottom_right_pt[0]],
                                  (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Check motion and play beep in a separate thread if necessary
                    current_time = time.time()

                    if not motion_detected:
                        continuous_detection_start_time = current_time
                        motion_detected = True

                    detection_duration = current_time - continuous_detection_start_time

                    if detection_duration >= detect_duration:
                        if current_time - last_beep_time >= last_beep_time_threshold:
                            play_beep_thread(900)  # Adjust the beep duration (ms) as needed
                            last_beep_time = current_time

                            # Minimize windows
                            minimize_windows()

                    else:
                        if current_time - last_beep_time >= last_beep_time_threshold:
                            play_beep_thread(500)  # Default beep duration (ms)
                            last_beep_time = current_time

            # Display the frame with bounding boxes
            cv2.imshow('Moving Person Detection', frame)
            show_selected_area(frame, top_left_pt, bottom_right_pt)
        else:
            print("Error: Unable to apply background subtraction within ROI")

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the windows
cap.release()
cv2.destroyAllWindows()
