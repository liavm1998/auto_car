import cv2
import numpy as np

def initialize_camera_parameters(resolution=(2592, 1944 )):
    # Compute the focal length using the resolution and field of view
    focal_length = 3.60 

    # Define the camera matrix
    camera_matrix = np.array([[focal_length, 0, resolution[0] / 2],
                              [0, focal_length, resolution[1] / 2],
                              [0, 0, 1]])
    # Initialize distortion coefficients to zero
    distortion_coeffs = np.zeros((4, 1))

    return camera_matrix, distortion_coeffs

# Function to compute the 3D pose (distance, yaw, pitch, roll) from rotation and translation vectors
def compute_3d_pose(rvec, tvec):
    distance = np.linalg.norm(tvec)  # Calculate the distance
    R, _ = cv2.Rodrigues(rvec)  # Convert rotation vector to rotation matrix
    yaw = np.arctan2(R[1, 0], R[0, 0])  # Calculate yaw angle
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))  # Calculate pitch angle
    roll = np.arctan2(R[2, 1], R[2, 2])  # Calculate roll angle

    return distance, yaw, pitch, roll

# Function to detect QR codes in a given frame
def detect_QR(frame):
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    try:
        # TODO PRINT HARUKO WITH SPECIFIC ID AND USE THIS ID HERE (WE WILL SEARCH FOR IT SPECIFICALY)
        # Detect markers using ArUco dictionary
        corners, ids, _ = cv2.aruco.detectMarkers(gray_frame, cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100))
        return corners, ids
    except Exception as e:
        print(f"Error detecting markers: {e}")
        return None, None
    

# Main function to process the video file
def detect_from_video():
    cap = cv2.VideoCapture(0)  # Open the default camera
    if not cap.isOpened():
        print(f"Error: Unable to open camera")
        return

    frame_rate = cap.get(cv2.CAP_PROP_FPS)   # Get the frame rate of the video
    if frame_rate == 0:
        print(f"Error: Frame rate of the camera is zero.")
        return

    delay = int(1000 / frame_rate) if frame_rate > 0 else 1  # Calculate delay between frames

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()  # Read a frame
        if not ret:
            break

        corners, ids = detect_QR(frame)  # Detect QR codes in the frame
        cap.release()  # Release the video capture object
        cv2.destroyAllWindows()  # Close all OpenCV windows
        return corners[0], ids[0], frame,
    
def calculate_vectors(id,corners,frame):
    camera_matrix, distortion_coeffs = initialize_camera_parameters()
    if id is not None:
        rotation_vec, transformation_vec = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, distortion_coeffs)
        return rotation_vec, transformation_vec
