# AI Object Detection & Robot Control

This repository contains tools and notebooks for training, evaluating, and deploying object detection models, specifically tailored for sorting tasks using a robotic arm and platform.

---

## 📂 Core Components

### 1. Model Evaluation & Comparison
* **[`compare_models.py`](file:///d:/School/Python/builderthing/aibuilder/compare_models.py)**: An evaluation script used to compare different object detection architectures and create a baseline to measure my own custom YOLO26 model against standard models (RT-DETR, YOLOv8, and YOLOv11).
  * Automatically calculates metrics like Precision, Recall, mAP@50, mAP@50-95, inference latency, parameter counts, and FLOPs.
  * Generates visual side-by-side grid overlays of model predictions to inspect qualitative differences.
  * Outputs summary data directly into a spreadsheet format.
* **[`model_comparison.csv`](file:///d:/School/Python/builderthing/aibuilder/model_comparison.csv)**: A spreadsheet recording baseline performance comparison metrics.

### 2. Model Training
* **[`trainyolo26.ipynb`](file:///d:/School/Python/builderthing/aibuilder/trainyolo26.ipynb)**: A Notebook for configuring, training, and fine-tuning YOLO26.

### 3. Robotic Deployment
* **[`robot/`](file:///d:/School/Python/builderthing/aibuilder/robot)**: Contains a script to control the robot through a Raspi, and an STL file of the robot.
  * **[`robotcontrol.py`](file:///d:/School/Python/builderthing/aibuilder/robot/robotcontrol.py)**: A robot control script. It utilizes Tkinter for a control GUI, OpenCV for live video streaming, PCA9685/ServoKit for robotic arm control, and YOLO model inference to identify, track, and sort target objects.
  * **[`shitslopbot.stl`](file:///d:/School/Python/builderthing/aibuilder/robot/robothardware/shitslopbot.stl)**: An STL file of the robot.

---

## 🤗 Models & Datasets
You can find the pre-trained model weights and datasets for this project on my Hugging Face profile: https://huggingface.co/milm01

---

## 🛠️ Usage

### Running the Comparison Pipeline
To run a evaluation and generate comparison results:
```powershell
python compare_models.py --train-yolov8 --train-yolov11 --epochs 30 --batch 32
```
Options:
- `--train-yolov8` / `--train-yolov11`: Train baseline YOLOv8/YOLOv11 models on your dataset.
- `--retrain`: Forces training even if weights exist.
- `--epochs <N>`: Set the number of epochs.
- `--batch <N>`: Set the batch size.

### Training the YOLO26 Model
Open `trainyolo26.ipynb` using Jupyter Notebook, VS Code Jupyter Extension, or Google Colab. Run the setup cells, configure your data directory, and start the training process.
