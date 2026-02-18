# AI-Based Attendance System

A simple facial recognition attendance system using Python and OpenCV.

## Requirements
- Python 3.x
- OpenCV with contrib modules (`pip install opencv-contrib-python`)
- Pillow (`pip install Pillow`)
- Numpy (`pip install numpy`) (usually comes with OpenCV)

## Project Structure
- `dataset_capture.py`: Captures user faces and stores them in `dataset/`.
- `train_model.py`: Trains the LBPH recognizer on the captured dataset.
- `attendance_system.py`: Runs the webcam, recognizes faces, and marks attendance.
- `dataset/`: Folder containing face images.
- `trainer/`: Folder containing the trained model (`trainer.yml`).
- `attendance.csv`: Output file accumulating attendance records.

## How to Use

1. **Setup**:
   Ensure `haarcascade_frontalface_default.xml` is in the project folder. (It should have been downloaded automatically).

2. **Capture Faces**:
   Run `dataset_capture.py`.
   ```bash
   python dataset_capture.py
   ```
   - Enter a numeric User ID (e.g., 1) and User Name (e.g., John).
   - Look at the camera. It will take 30 snapshots.

3. **Train Model**:
   Run `train_model.py`.
   ```bash
   python train_model.py
   ```
   - This keeps `trainer/trainer.yml` updated with all users in `dataset/`.

4. **Start Attendance System**:
   Run `attendance_system.py`.
   ```bash
   python attendance_system.py
   ```
   - The system will open a window and recognize faces.
   - Attendance is marked in `attendance.csv` automatically for recognized faces.
   - Press 'q' to exit.

## Notes
- To reset the system, you can delete `trainer/trainer.yml` and empty `dataset/`.
- Ensure lighting is good for better recognition accuracy.
