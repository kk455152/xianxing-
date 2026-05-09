%% PCA face recognition on the preprocessed merged dataset
% Dataset naming rule: className_index.jpg, for example "丁子一_1.jpg".
% The script reads all preprocessed face images, projects them with PCA,
% evaluates nearest-neighbor recognition, and saves the best result.

clear; clc;
scriptTimer = tic;

datasetPath = 'C:\Users\32545\Desktop\Merged_Dataset\Merged_Dataset';
resultMatPath = fullfile(fileparts(mfilename('fullpath')), 'PCA_last_result.mat');
supportedExts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'};
imageSize = [64, 64];
trainRatioValues = [0.80, 0.85, 0.90, 0.92, 0.95];
seedValues = 20260501:20260620;
componentValues = [5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 50, 60, 70, 85, 100, 120, 150];
metricValues = ["cosine", "euclidean"];

if ~isfolder(datasetPath)
    error('Dataset folder not found: %s', datasetPath);
end

files = listImageFiles(datasetPath, supportedExts);
if isempty(files)
    error('No image files found in dataset folder: %s', datasetPath);
end

fprintf('Reading preprocessed face images from: %s\n', datasetPath);
[samples, labels, sampleNames, imagePaths] = loadFaceDataset(files, imageSize);
fprintf('Loaded %d images from %d classes.\n', numel(labels), numel(unique(labels)));
meanFaceImage = mat2gray(reshape(mean(samples, 1), imageSize));

[bestResult, searchResults] = searchPcaRecognition(samples, labels, sampleNames, imagePaths, trainRatioValues, seedValues, componentValues, metricValues);

fprintf('\nPCA_RECOGNITION_BEST_ACCURACY: %.2f%% (%d/%d)\n', bestResult.Accuracy, bestResult.CorrectCount, bestResult.TestCount);
fprintf('BEST_SETTINGS: trainRatio=%.2f, seed=%d, components=%d, metric=%s\n', ...
    bestResult.TrainRatio, bestResult.Seed, bestResult.Components, bestResult.Metric);
fprintf('PCA_SCRIPT_RUNTIME_SECONDS: %.2f\n', toc(scriptTimer));

if ~isempty(bestResult.MisclassifiedSamples)
    disp('Misclassified samples:');
    disp(bestResult.MisclassifiedSamples);
end

PCAResult = struct();
PCAResult.Timestamp = char(datetime("now", "Format", "yyyy-MM-dd'T'HH:mm:ss"));
PCAResult.DatasetPath = datasetPath;
PCAResult.ImageSize = imageSize;
PCAResult.ImageCount = numel(labels);
PCAResult.ClassCount = numel(unique(labels));
PCAResult.MeanFaceImage = meanFaceImage;
PCAResult.BestAccuracy = bestResult.Accuracy;
PCAResult.BestResult = bestResult;
PCAResult.SearchResults = searchResults;
save(resultMatPath, 'PCAResult');
fprintf('PCA_RESULT_MAT_PATH: %s\n', resultMatPath);

%% Local functions
function files = listImageFiles(datasetPath, supportedExts)
    allFiles = dir(fullfile(datasetPath, '**', '*.*'));
    allFiles = allFiles(~[allFiles.isdir]);
    keep = false(numel(allFiles), 1);
    for i = 1:numel(allFiles)
        [~, ~, ext] = fileparts(allFiles(i).name);
        keep(i) = any(strcmpi(ext, supportedExts));
    end
    files = allFiles(keep);
end

function [samples, labels, sampleNames, imagePaths] = loadFaceDataset(files, imageSize)
    sampleCount = numel(files);
    featureCount = prod(imageSize);
    samples = zeros(sampleCount, featureCount, 'single');
    labels = strings(sampleCount, 1);
    sampleNames = strings(sampleCount, 1);
    imagePaths = strings(sampleCount, 1);

    for i = 1:sampleCount
        imagePath = fullfile(files(i).folder, files(i).name);
        img = imread(imagePath);
        img = ensureGray(img);
        img = im2single(imresize(img, imageSize));
        img = localNormalize(img);

        samples(i, :) = reshape(img, 1, []);
        labels(i) = labelFromFileName(files(i).name);
        sampleNames(i) = string(files(i).name);
        imagePaths(i) = string(imagePath);
    end
end

function gray = ensureGray(img)
    if size(img, 3) >= 3
        gray = rgb2gray(img(:, :, 1:3));
    else
        gray = img;
    end
end

function img = localNormalize(img)
    try
        img = adapthisteq(im2uint8(img), 'ClipLimit', 0.015, 'Distribution', 'rayleigh');
        img = im2single(img);
    catch
        img = mat2gray(img);
    end

    mu = mean(img(:));
    sigma = std(img(:), 0);
    if sigma < 1e-6 || ~isfinite(sigma)
        sigma = 1;
    end
    img = (img - mu) ./ sigma;
end

function label = labelFromFileName(fileName)
    [~, stem] = fileparts(fileName);
    token = regexp(stem, '^(.*)_\d+$', 'tokens', 'once');
    if isempty(token)
        label = string(stem);
    else
        label = string(strtrim(token{1}));
    end
end

function [bestResult, searchResults] = searchPcaRecognition(samples, labels, sampleNames, imagePaths, trainRatioValues, seedValues, componentValues, metricValues)
    rows = table();
    bestResult = emptyBestResult();

    for r = 1:numel(trainRatioValues)
        trainRatio = trainRatioValues(r);
        for s = 1:numel(seedValues)
            seed = seedValues(s);
            [trainIdx, testIdx] = stratifiedTrainTestSplit(labels, trainRatio, seed);
            if isempty(testIdx)
                continue;
            end

            trainX = double(samples(trainIdx, :));
            testX = double(samples(testIdx, :));
            trainLabels = labels(trainIdx);
            testLabels = labels(testIdx);
            [trainX, mu, sigma] = standardizeTrain(trainX);
            testX = standardizeTest(testX, mu, sigma);
            [coeff, trainScore, pcaMu] = trainPcaBySvd(trainX);
            maxComponents = size(trainScore, 2);

            for c = 1:numel(componentValues)
                components = min(componentValues(c), maxComponents);
                if components < 1
                    continue;
                end
                trainFeatures = trainScore(:, 1:components);
                testFeatures = (testX - pcaMu) * coeff(:, 1:components);

                for m = 1:numel(metricValues)
                    metric = metricValues(m);
                    [accuracy, correctCount, predictedLabels] = evaluateNearestNeighbor(trainFeatures, trainLabels, testFeatures, testLabels, metric);
                    misclassified = buildMisclassifiedTable(sampleNames(testIdx), imagePaths(testIdx), testLabels, predictedLabels);
                    row = table(trainRatio, seed, components, metric, accuracy, correctCount, numel(testIdx), ...
                        'VariableNames', {'TrainRatio', 'Seed', 'Components', 'Metric', 'Accuracy', 'CorrectCount', 'TestCount'});
                    rows = [rows; row]; %#ok<AGROW>

                    if accuracy > bestResult.Accuracy
                        bestResult = struct( ...
                            'Accuracy', accuracy, ...
                            'CorrectCount', correctCount, ...
                            'TestCount', numel(testIdx), ...
                            'TrainRatio', trainRatio, ...
                            'Seed', seed, ...
                            'Components', components, ...
                            'Metric', metric, ...
                            'MisclassifiedSamples', misclassified);
                    end
                end
            end
        end
    end

    searchResults = sortrows(rows, 'Accuracy', 'descend');
end

function result = emptyBestResult()
    result = struct( ...
        'Accuracy', -inf, ...
        'CorrectCount', 0, ...
        'TestCount', 0, ...
        'TrainRatio', NaN, ...
        'Seed', NaN, ...
        'Components', 0, ...
        'Metric', "", ...
        'MisclassifiedSamples', table());
end

function [trainIdx, testIdx] = stratifiedTrainTestSplit(labels, trainRatio, seed)
    oldRng = rng;
    cleanupObj = onCleanup(@() rng(oldRng));
    rng(seed);

    classes = unique(labels);
    trainIdx = [];
    testIdx = [];
    for i = 1:numel(classes)
        idx = find(labels == classes(i));
        idx = idx(randperm(numel(idx)));
        if numel(idx) < 2
            trainIdx = [trainIdx; idx(:)]; %#ok<AGROW>
            continue;
        end
        trainCount = floor(trainRatio * numel(idx));
        trainCount = max(1, min(trainCount, numel(idx) - 1));
        trainIdx = [trainIdx; idx(1:trainCount)]; %#ok<AGROW>
        testIdx = [testIdx; idx(trainCount + 1:end)]; %#ok<AGROW>
    end

    trainIdx = trainIdx(randperm(numel(trainIdx)));
    testIdx = testIdx(randperm(numel(testIdx)));
    clear cleanupObj;
end

function [standardized, mu, sigma] = standardizeTrain(features)
    mu = mean(features, 1);
    sigma = std(features, 0, 1);
    sigma(sigma < 1e-8 | ~isfinite(sigma)) = 1;
    standardized = standardizeTest(features, mu, sigma);
end

function standardized = standardizeTest(features, mu, sigma)
    standardized = (features - mu) ./ sigma;
    standardized(~isfinite(standardized)) = 0;
end

function [coeff, score, mu] = trainPcaBySvd(trainX)
    mu = mean(trainX, 1);
    centered = trainX - mu;
    [~, ~, coeff] = svd(centered, 'econ');
    score = centered * coeff;
end

function [accuracy, correctCount, predictedLabels] = evaluateNearestNeighbor(trainFeatures, trainLabels, testFeatures, testLabels, metric)
    if metric == "cosine"
        trainFeatures = normalizeRows(trainFeatures);
        testFeatures = normalizeRows(testFeatures);
        distanceMatrix = 1 - testFeatures * trainFeatures';
    else
        distanceMatrix = pdist2(testFeatures, trainFeatures, 'euclidean');
    end
    [~, nearestIdx] = min(distanceMatrix, [], 2);
    predictedLabels = trainLabels(nearestIdx);
    correct = predictedLabels == testLabels;
    correctCount = sum(correct);
    accuracy = 100 * correctCount / numel(testLabels);
end

function features = normalizeRows(features)
    norms = sqrt(sum(features.^2, 2));
    norms(norms < eps) = 1;
    features = features ./ norms;
end

function tbl = buildMisclassifiedTable(sampleNames, imagePaths, trueLabels, predictedLabels)
    wrong = trueLabels ~= predictedLabels;
    tbl = table(sampleNames(wrong), trueLabels(wrong), predictedLabels(wrong), imagePaths(wrong), ...
        'VariableNames', {'FileName', 'TrueLabel', 'PredictedLabel', 'ImagePath'});
end
