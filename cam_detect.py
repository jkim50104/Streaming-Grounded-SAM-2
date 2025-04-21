import cv2
import time

def get_available_cameras():
    available_cameras = []
    # Check for 5 cameras 
    for i in range(100):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

def visualize_cameras(cameras, duration=5):
    for cam_index in cameras:
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            print(f"Camera {cam_index} couldn't be opened.")
            continue
        
        print(f"Displaying camera {cam_index}. Press any key to skip.")

        start_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read from camera {cam_index}")
                break

            cv2.imshow(f'Camera {cam_index}', frame)

            # Exit after 'duration' seconds or if a key is pressed
            if cv2.waitKey(1) != -1 or (time.time() - start_time > duration):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    cameras = get_available_cameras()
    if cameras:
        print("Available Cameras:", cameras)
        visualize_cameras(cameras)
    else:
        print("No cameras found.")
