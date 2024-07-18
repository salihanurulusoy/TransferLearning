imdsTrain = imageDatastore('C:\Users\inancok\Desktop\Dataset Crop\train', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
imdsTest = imageDatastore('C:\Users\inancok\Desktop\Dataset Crop\test', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
imdsVal = imageDatastore('C:\Users\inancok\Desktop\Dataset Crop\val', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

% VGG-19 modelini yükleme
net = vgg19;

% Giriş boyutunu al
inputSize = net.Layers(1).InputSize;

% Fully connected layer ekleme
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(imdsTrain.Labels));
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'Name','fc8','WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];

% Veri augmentasyonu
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsVal,...
    'DataAugmentation',imageAugmenter);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest, ...
    'DataAugmentation',imageAugmenter); 

% Eğitim seçenekleri
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');


% Transfer learning modelini eğitme
tic; % Süre ölçümü başlat
netTransfer = trainNetwork(augimdsTrain,layers,options);
trainingTime = toc; % Eğitim süresini hesapla

% Eğitim süresini ekrana yazdırma
disp(['Eğitim Süresi: ' num2str(trainingTime) ' s']);

% Doğrulama
[YPred,scores] = classify(netTransfer,augimdsValidation);
YValidation = imdsVal.Labels;
accuracy = mean(YPred == YValidation);
disp(['Doğruluk: ' num2str(accuracy)]);

% Test doğrulama
[YPredTest,scoresTest] = classify(netTransfer,augimdsTest);
YTest = imdsTest.Labels;
accuracyTest = mean(YPredTest == YTest);
disp(['Test Doğruluğu: ' num2str(accuracyTest)]);

% Çalışma alanını kaydet
save('results_workspace.mat', 'YPred', 'scores', 'YValidation', 'accuracy', 'YPredTest', 'scoresTest', 'YTest', 'accuracyTest', 'netTransfer', 'trainingTime');

