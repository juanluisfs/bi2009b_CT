%Alexa de León
%A01382990
%%
imageSize = [480 640 8]; % input image size
numClasses = 5; % number of output classes
EncoderDepth = 3; % depth of encoder-decoder
value = 3;

% create U-Net layers
layers = unetLayers(imageSize, numClasses, 'EncoderDepth', value);

% visualize network architecture
clf
plot(layers)

%% 2)

imageSize = [32 32]; % input image size
numClasses = 2; % number of output classes


% create U-Net layers
layers = unetLayers(imageSize, numClasses);

% visualize network architecture
clf
plot(layers)

% specify the directories containing the training images and pixel labels
dataSetDir = fullfile(toolboxdir('vision'), 'visiondata', 'triangleImages');
imageDir = fullfile(dataSetDir, 'trainingImages');
labelDir = fullfile(dataSetDir, 'trainingLabels');

% define the class names and label IDs
classNames = ["triangle", "background"];
labelIDs = [255, 0];

% create the imageDatastore object to store the training images
imds = imageDatastore(imageDir);

% create the pixelLabelDatastore object to store the ground truth pixel labels
pxds = pixelLabelDatastore(labelDir, classNames, labelIDs);

%% 
% Define input image size and number of classes
imageSize = [32 32];
numClasses = 2;

% Create U-Net layers
unet = unetLayers(imageSize, numClasses);

% Create a combined datastore for training
ds = combine(imds, pxds);

% Define training options
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 20, ...
    'VerboseFrequency', 10);

% Train the network
net = trainNetwork(ds, unet, options);


%%

% Train the network
%net = trainNetwork(ds,lgraph,options)

% Specify test images and labels
testImagesDir = fullfile(dataSetDir,'testImages');
testimds = imageDatastore(testImagesDir);
testLabelsDir = fullfile(dataSetDir,'testLabels');

%create a pixelLabelDatastore object to hold the ground truth pixel labels for the test images.
pxdsTruth = pixelLabelDatastore(testLabelsDir,classNames,labelIDs);

%run your network on the test images (be patient and wait until the 100 images are processed)
pxdsResults = semanticseg(testimds,net,"WriteLocation",tempdir);

%evaluate the quality of your prediction you are going to use the following instruction. It receives two arguments: your predictions, and the ground truth pixels.
metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTruth);
%%
%  Inspect class metrcis
metrics.ClassMetrics

% Display confusion matrix
metrics.ConfusionMatrix

% Visualize the normalized confusion matrix as a confusion chart in a figure window.
figure
cm = confusionchart(metrics.ConfusionMatrix.Variables, ...
classNames, Normalization='row-normalized');
cm.Title = 'Normalized Confusion Matrix (%)';
%%
imageIoU = metrics.ImageMetrics.MeanIoU;
figure (3)
histogram(imageIoU)
title('Image Mean IoU')
%%
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

%%
