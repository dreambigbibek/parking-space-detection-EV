% === Load test ground truth ===
testData = load('parkingTestGT.mat');
gTruthTest = testData.gTruth;

% === Convert ground truth to test datastore ===
[imdsTest, bldsTest] = objectDetectorTrainingData(gTruthTest);
testDataCombined = combine(imdsTest, bldsTest);

% === Run detection on test images ===
detectionResults = detect(detector, imdsTest);

% === Evaluate AP per class at IoU threshold = 0.5 ===
[ap, recall, precision] = evaluateDetectionPrecision(detectionResults, testDataCombined, 0.5);

% === Display AP values per class ===
disp('Average Precision (AP) per class:');
disp(ap)

% === Identify class names ===
classNames = detector.ClassNames;

% === Find highest and lowest AP ===
[~, bestIdx] = max(ap);
[~, worstIdx] = min(ap);
fprintf('Class with highest AP: %s (%.2f)\n', classNames{bestIdx}, ap(bestIdx));
fprintf('Class with lowest AP: %s (%.2f)\n', classNames{worstIdx}, ap(worstIdx));

% === Calculate mAP ===
mAP = mean(ap);
fprintf('Mean Average Precision (mAP) = %.2f\n', mAP);

% === Plot AP vs Detection Threshold and suggest best threshold ===
figure;
evaluateDetectionPrecision(detectionResults, testDataCombined, 0.5, 'ShowPlot', true);
title('AP vs Detection Threshold');
