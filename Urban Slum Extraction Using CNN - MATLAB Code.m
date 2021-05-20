clc;
clear all;
close all;
layer = tverskyPixelClassificationLayer('tversky',0.7,0.3);
numClasses = 2;
validInputSize = [4 4 numClasses];
checkLayer(layer,validInputSize, 'ObservationDimension',4);
layers = [
    imageInputLayer([100 100 3])
    convolution2dLayer(3,64,'Padding',1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,64,'Padding',1)
    reluLayer
    transposedConv2dLayer(4,64,'Stride',2,'Cropping',1)
    convolution2dLayer(1,2)
    softmaxLayer
    tverskyPixelClassificationLayer('tversky',0.3,0.7)];
dataSetDir = fullfile('D:\DeepLearning');
imageDir = fullfile(dataSetDir,'ori');
labelDir = fullfile(dataSetDir,'lab');

imds = imageDatastore(imageDir);

classNames = ["slums" "background"];
labelIDs = [255 0];
pxds = pixelLabelDatastore(labelDir, classNames, labelIDs);
ds = pixelLabelImageDatastore(imds,pxds);
options = trainingOptions('adam', ...
    'InitialLearnRate',1e-3, ...
    'MaxEpochs',100, ...
    'LearnRateDropFactor',5e-1, ...
    'LearnRateDropPeriod',20, ...
    'LearnRateSchedule','piecewise', ...
    'MiniBatchSize',50);

net = trainNetwork(ds,layers,options);
I = imread('D:\DeepLearning\trial\41.tif');
[C,scores] = semanticseg(I,net);
B = labeloverlay(I,C);
montage({I,B})