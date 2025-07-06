# 🚘 Parking Sign Detection using YOLOX (MATLAB)

This repository contains MATLAB code and data preprocessing steps for training an object detection model to identify **parking-related signs** such as `"EV"`, `"Charger"`, and `"Accessible"` signs using the **YOLOX Tiny-Coco** backbone.


## 🚀 Features

- ✅ Object detection using **YOLOX (Tiny-Coco)**
- ✅ Handles 3 parking sign classes: `EV`, `Charger`, `Accessible`
- ✅ End-to-end pipeline: training, evaluation, and inference
- ✅ Configurable mini-batch size, input size, and epochs
- ✅ Evaluation with mAP, per-class AP, and detection threshold tuning

## 🛠️ Getting Started

### 1. Prerequisites

- MATLAB R2023b or later
- Computer Vision Toolbox
- Deep Learning Toolbox

### 2. Load and Preprocess Data

```matlab
data = load('parkingTrainGTfinished.mat');
gTruth = data.gTruth;
[imds, blds] = objectDetectorTrainingData(gTruth);
trainingData = combine(imds, blds);

##Train YOLOX Detector
matlab
Copy
Edit
sampleImg = readimage(imds, 1);
inputSize = size(sampleImg);
classes = categories(blds.LabelData{1,2});
model = yoloxObjectDetector("tiny-coco", classes, InputSize=inputSize);

options = trainingOptions("sgdm", ...
    MiniBatchSize=16, MaxEpochs=10, InitialLearnRate=1e-3, ...
    ValidationData=trainingData, ValidationFrequency=25, ...
    Shuffle="every-epoch", Plots="training-progress");

[detector, info] = trainYOLOXObjectDetector(trainingData, model, options);

📈 Evaluation
Evaluate your trained model on the test dataset using:

matlab
Copy
Edit
testData = load('parkingTestGT.mat');
[imdsTest, bldsTest] = objectDetectorTrainingData(testData.gTruth);
detectionResults = detect(detector, imdsTest);
[ap, recall, precision] = evaluateDetectionPrecision(detectionResults, combine(imdsTest, bldsTest), 0.5);
mAP@0.5: ≈ 0.XX (replace with your result)

Best performing class: EV / Charger / Accessible

Detection threshold: ~0.55 recommended for balanced performance

🔍 Visualization
matlab
Copy
Edit
I = readimage(imdsTest, 10);
[bboxes, scores, labels] = detect(detector, I);
annotated = insertObjectAnnotation(I, 'rectangle', bboxes, cellstr(labels));
imshow(annotated)
📋 Dataset Notes
Train set: 400 images (80% train, 20% validation split)

Test set: From parkingTestGT.mat

Classes: EV, Charger, Accessible

📌 Future Improvements
Add real-time webcam inference

Compare YOLOX with YOLOv4 Tiny

Augment dataset with nighttime and occluded signs

🧑‍💻 Author
Bibek Gautam
M.S. in Computer Science – Machine Learning & Vision
Lamar University


