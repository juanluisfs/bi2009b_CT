%% BI2009B Procesamiento de Imágenes Médicas para el Diagnóstico
%% Actividad 1 - Segmentation U-Net
%% A0138088 Juan Luis Flores Sánchez 

%Ejemplo 1
ImageSize_1 = [480 640 3];
NoClasses_1 = 5;
EncoderDepth_1 = 3;
Network_1 = unetLayers(ImageSize_1,NoClasses_1,'EncoderDepth',EncoderDepth_1);
plot(Network_1)

%% Ejercicio 1
clc
clear
close all

NoClasses = [5;9;3];
EncoderDepth = [3;5;7];
ImageSize = [480 640 EncoderDepth(1);640 416 EncoderDepth(2);512 512 EncoderDepth(3)];

t = tiledlayout(1,3,'TileSpacing','Compact','Padding','Compact');
for i = 1:3
    Network = unetLayers(ImageSize(i,:),NoClasses(i),'EncoderDepth',EncoderDepth(i));
    nexttile
    plot(Network)
    title(sprintf("Network #%0.0f - Image Size = %0.0fx%0.0fx%0.0f ; Classes = %0.0f ; Depth = %0.0f ",i,ImageSize(i,1),ImageSize(i,2),ImageSize(i,3),NoClasses(i),EncoderDepth(i)))
end

dataSetDir = fullfile(toolboxdir('vision'),'visiondata','triangleImages');
imageDir = fullfile(dataSetDir,'trainingImages');
labelDir = fullfile(dataSetDir,'trainingLabels');

classNames = ["triangle","background"];
labelIDs   = [255 0];

imds = imageDatastore(imageDir);
pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);

imageSize = [32 32];
numClasses = 2;

out = unetLayers(imageSize, numClasses);

ds = combine(imds,pxds);

% options = trainingOptions('sgdm', ...
%     'InitialLearnRate',1e-3, ...
%     'MaxEpochs',10, ...
%     'VerboseFrequency',10);
% 
% net = trainNetwork(ds,out,options)

options1 = trainingOptions('sgdm', ...
    'InitialLearnRate',1e-3, ...
    'MaxEpochs',20, ...
    'VerboseFrequency',10);

net1 = trainNetwork(ds,out,options1)

% options2 = trainingOptions('sgdm', ...
%     'InitialLearnRate',1e-3, ...
%     'MaxEpochs',100, ...
%     'VerboseFrequency',10);
% 
% net2 = trainNetwork(ds,out,options2)
% 
% options3 = trainingOptions('sgdm', ...
%     'InitialLearnRate',1e-3, ...
%     'MaxEpochs',200, ...
%     'VerboseFrequency',10);
% 
% net3 = trainNetwork(ds,out,options3)
% 
% options4 = trainingOptions('sgdm', ...
%     'InitialLearnRate',1e-3, ...
%     'MaxEpochs',1000, ...
%     'VerboseFrequency',10);
% 
% net4 = trainNetwork(ds,out,options4)

testImagesDir = fullfile(dataSetDir,'testImages');
testimds = imageDatastore(testImagesDir);
testLabelsDir = fullfile(dataSetDir,'testLabels');

pxdsTruth = pixelLabelDatastore(testLabelsDir,classNames,labelIDs);

pxdsResults = semanticseg(testimds,net1,"WriteLocation",tempdir);

metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTruth);

metrics.ClassMetrics

metrics.ConfusionMatrix

figure
cm = confusionchart(metrics.ConfusionMatrix.Variables, ...
classNames, Normalization='row-normalized');
cm.Title = 'Normalized Confusion Matrix (%)';

imageIoU = metrics.ImageMetrics.MeanIoU;
figure
histogram(imageIoU)
title('Image Mean IoU')

% Find the test image with the lowest IoU.
[minIoU, worstImageIndex] = min(imageIoU);
minIoU = minIoU(1);
worstImageIndex = worstImageIndex(1);

% Read the test image with the worst IoU, its ground truth labels, and its predicted labels for comparison.
worstTestImage = readimage(imds,worstImageIndex);
worstTrueLabels = readimage(pxdsTruth,worstImageIndex);
worstPredictedLabels = readimage(pxdsResults,worstImageIndex);

% Convert the label images to images that can be displayed in a figure window.
worstTrueLabelImage = im2uint8(worstTrueLabels == classNames(1));
worstPredictedLabelImage = im2uint8(worstPredictedLabels == classNames(1));

% Display the worst test image, the ground truth, and the prediction.
worstMontage = cat(4,worstTestImage,worstTrueLabelImage,worstPredictedLabelImage);
WorstMontage = imresize(worstMontage,4,"nearest");

figure (4)
montage(worstMontage,'Size',[1 3])
title(['Test Image vs. Truth vs. Prediction. IoU = ' num2str(minIoU)])

% Find the test image with the highest IoU.
[maxIoU, bestImageIndex] = max(imageIoU);
maxIoU = maxIoU(1);
bestImageIndex = bestImageIndex(1);

% Read the test image with the best IoU, its ground truth labels, and its predicted labels for comparison.
bestTestImage = readimage(imds,bestImageIndex);
bestTrueLabels = readimage(pxdsTruth,bestImageIndex);
bestPredictedLabels = readimage(pxdsResults,bestImageIndex);

% Convert the label images to images that can be displayed in a figure window.
bestTrueLabelImage = im2uint8(bestTrueLabels == classNames(1));
bestPredictedLabelImage = im2uint8(bestPredictedLabels == classNames(1));

% Display the best test image, the ground truth, and the prediction.
bestMontage = cat(4,bestTestImage,bestTrueLabelImage,bestPredictedLabelImage);
bestMontage = imresize(bestMontage,4,"nearest");

figure (5)
montage(bestMontage,'Size',[1 3])
title(['Test Image vs. Truth vs. Prediction. IoU = ' num2str(maxIoU)])