% === Step 1: Load ground truth ===
data = load('parkingTrainGTfinished.mat');  % Loads variable(s)
gTruth = data.gTruth;  % Confirmed variable name from your earlier message

% Convert ground truth to datastore
[imds, blds] = objectDetectorTrainingData(gTruth);  % No 'OutputVariableName'

% Combine image and box label datastores
trainingData = combine(imds, blds);

% === Step 2: Get image size ===
sampleImg = readimage(imds, 1);
inputSize = size(sampleImg);  % [H W 3]

% === Step 3: Define YOLOX model ===
classes = categories(blds.LabelData{1,2});
numClasses = width(blds.read());  % Automatically detects number of classes
model = yoloxObjectDetector("tiny-coco", classes, InputSize=inputSize);

% === Step 4: Training options ===
miniBatchSize = 16;
numEpochs = 10;
valFrequency = floor(numel(imds.Files)/miniBatchSize);

options = trainingOptions("sgdm", ...
    MiniBatchSize=miniBatchSize, ...
    MaxEpochs=numEpochs, ...
    InitialLearnRate=1e-3, ...
    ValidationData=trainingData, ...
    ValidationFrequency=valFrequency, ...
    Shuffle="every-epoch", ...
    VerboseFrequency=10, ...
    Plots="training-progress");

% === Step 5: Train detector ===
[detector, info] = trainYOLOXObjectDetector(trainingData, model, options);
