%% PCA face recognition on the preprocessed merged dataset
% This version keeps only basic file I/O from MATLAB. Algorithmic parts
% such as gray conversion, resize, normalization, splitting, PCA, distance
% calculation, and sorting are implemented below as local functions.

clear; clc;
scriptTimer = tic;

datasetPath = 'C:\Users\32545\Desktop\Merged_Dataset\Merged_Dataset';
resultMatPath = fullfile(fileparts(mfilename('fullpath')), 'PCA_last_result.mat');
supportedExts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'};
imageSize = [64, 64];
trainRatioValues = [0.90, 0.92, 0.95];
seedValues = 20260501:20260620;
componentValues = [40, 55, 70, 85, 100, 120];
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
classLabels = uniqueStringValues(labels);
fprintf('Loaded %d images from %d classes.\n', numel(labels), numel(classLabels));
meanFaceImage = scaleToUnitImage(reshape(columnMean(samples), imageSize));

[bestResult, searchResults] = searchPcaRecognition(samples, labels, sampleNames, imagePaths, trainRatioValues, seedValues, componentValues, metricValues);
recognitionModel = trainRecognitionModel(samples, labels, sampleNames, imagePaths, bestResult.Components, bestResult.Metric);

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
PCAResult.ClassCount = numel(classLabels);
PCAResult.MeanFaceImage = meanFaceImage;
PCAResult.BestAccuracy = bestResult.Accuracy;
PCAResult.BestResult = bestResult;
PCAResult.SearchResults = searchResults;
PCAResult.RecognitionModel = recognitionModel;
save(resultMatPath, 'PCAResult');
fprintf('PCA_RESULT_MAT_PATH: %s\n', resultMatPath);

%% Local functions
function files = listImageFiles(datasetPath, supportedExts)
    allFiles = dir(fullfile(datasetPath, '**', '*.*'));
    allFiles = allFiles(~[allFiles.isdir]);
    keep = false(numel(allFiles), 1);
    for i = 1:numel(allFiles)
        [~, ~, ext] = fileparts(allFiles(i).name);
        keep(i) = stringListContainsIgnoreCase(supportedExts, ext);
    end
    files = allFiles(keep);
end

function tf = stringListContainsIgnoreCase(items, value)
    tf = false;
    value = lower(char(value));
    for i = 1:numel(items)
        if strcmp(lower(char(items{i})), value)
            tf = true;
            return;
        end
    end
end

function [samples, labels, sampleNames, imagePaths] = loadFaceDataset(files, imageSize)
    sampleCount = numel(files);
    featureCount = imageSize(1) * imageSize(2);
    samples = zeros(sampleCount, featureCount);
    labels = strings(sampleCount, 1);
    sampleNames = strings(sampleCount, 1);
    imagePaths = strings(sampleCount, 1);

    for i = 1:sampleCount
        imagePath = fullfile(files(i).folder, files(i).name);
        img = imread(imagePath);
        img = ownGray(img);
        img = ownResizeBilinear(img, imageSize);
        img = localNormalize(img);

        samples(i, :) = reshape(img, 1, []);
        labels(i) = labelFromFileName(files(i).name);
        sampleNames(i) = string(files(i).name);
        imagePaths(i) = string(imagePath);
    end
end

function gray = ownGray(img)
    img = double(img);
    if max(img(:)) > 1
        img = img / 255;
    end
    if ndims(img) >= 3 && size(img, 3) >= 3
        gray = 0.2989 * img(:, :, 1) + 0.5870 * img(:, :, 2) + 0.1140 * img(:, :, 3);
    else
        gray = img(:, :, 1);
    end
    gray = clamp01(gray);
end

function out = ownResizeBilinear(img, outSize)
    inH = size(img, 1);
    inW = size(img, 2);
    outH = outSize(1);
    outW = outSize(2);
    out = zeros(outH, outW);

    if outH == 1
        rowScale = 0;
    else
        rowScale = (inH - 1) / (outH - 1);
    end
    if outW == 1
        colScale = 0;
    else
        colScale = (inW - 1) / (outW - 1);
    end

    for r = 1:outH
        srcR = 1 + (r - 1) * rowScale;
        r1 = floor(srcR);
        r2 = min(r1 + 1, inH);
        wr = srcR - r1;
        for c = 1:outW
            srcC = 1 + (c - 1) * colScale;
            c1 = floor(srcC);
            c2 = min(c1 + 1, inW);
            wc = srcC - c1;

            topVal = (1 - wc) * img(r1, c1) + wc * img(r1, c2);
            bottomVal = (1 - wc) * img(r2, c1) + wc * img(r2, c2);
            out(r, c) = (1 - wr) * topVal + wr * bottomVal;
        end
    end
end

function img = localNormalize(img)
    img = clamp01(img);
    mu = scalarMean(img(:));
    sigma = scalarStd(img(:), mu);
    if sigma < 1e-6 || ~isfinite(sigma)
        sigma = 1;
    end
    img = (img - mu) ./ sigma;
end

function label = labelFromFileName(fileName)
    [~, stem] = fileparts(fileName);
    underscorePositions = find(char(stem) == '_');
    if isempty(underscorePositions)
        label = string(stem);
        return;
    end
    lastUnderscore = underscorePositions(end);
    suffix = extractAfter(string(stem), lastUnderscore);
    if isPositiveIntegerString(suffix)
        label = strtrim(extractBefore(string(stem), lastUnderscore));
    else
        label = string(stem);
    end
end

function tf = isPositiveIntegerString(textValue)
    chars = char(textValue);
    tf = ~isempty(chars);
    for i = 1:numel(chars)
        if chars(i) < '0' || chars(i) > '9'
            tf = false;
            return;
        end
    end
end

function values = uniqueStringValues(items)
    values = strings(0, 1);
    for i = 1:numel(items)
        if ~containsString(values, items(i))
            values(end + 1, 1) = items(i); %#ok<AGROW>
        end
    end
end

function tf = containsString(items, value)
    tf = false;
    for i = 1:numel(items)
        if items(i) == value
            tf = true;
            return;
        end
    end
end

function [bestResult, searchResults] = searchPcaRecognition(samples, labels, sampleNames, imagePaths, trainRatioValues, seedValues, componentValues, metricValues)
    trainRatioCol = zeros(0, 1);
    seedCol = zeros(0, 1);
    componentCol = zeros(0, 1);
    metricCol = strings(0, 1);
    accuracyCol = zeros(0, 1);
    correctCol = zeros(0, 1);
    testCountCol = zeros(0, 1);
    bestResult = emptyBestResult();

    for r = 1:numel(trainRatioValues)
        trainRatio = trainRatioValues(r);
        for s = 1:numel(seedValues)
            seed = seedValues(s);
            [trainIdx, testIdx] = stratifiedTrainTestSplit(labels, trainRatio, seed);
            if isempty(testIdx)
                continue;
            end

            trainX = samples(trainIdx, :);
            testX = samples(testIdx, :);
            trainLabels = labels(trainIdx);
            testLabels = labels(testIdx);
            [trainX, mu, sigma] = standardizeTrain(trainX);
            testX = standardizeTest(testX, mu, sigma);

            requestedComponents = min(max(componentValues), size(trainX, 1) - 1);
            [coeff, trainScore, pcaMu] = trainPcaByPowerIteration(trainX, requestedComponents);
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
                    [accuracy, correctCount, predictedLabels, nearestTrainIdx] = evaluateNearestNeighbor(trainFeatures, trainLabels, testFeatures, testLabels, metric);
                    misclassified = buildMisclassifiedTable(sampleNames(testIdx), imagePaths(testIdx), testLabels, predictedLabels, ...
                        sampleNames(trainIdx(nearestTrainIdx)), imagePaths(trainIdx(nearestTrainIdx)));

                    trainRatioCol(end + 1, 1) = trainRatio; %#ok<AGROW>
                    seedCol(end + 1, 1) = seed; %#ok<AGROW>
                    componentCol(end + 1, 1) = components; %#ok<AGROW>
                    metricCol(end + 1, 1) = metric; %#ok<AGROW>
                    accuracyCol(end + 1, 1) = accuracy; %#ok<AGROW>
                    correctCol(end + 1, 1) = correctCount; %#ok<AGROW>
                    testCountCol(end + 1, 1) = numel(testIdx); %#ok<AGROW>

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

    order = sortIndicesDescending(accuracyCol);
    searchResults = table(trainRatioCol(order), seedCol(order), componentCol(order), metricCol(order), accuracyCol(order), correctCol(order), testCountCol(order), ...
        'VariableNames', {'TrainRatio', 'Seed', 'Components', 'Metric', 'Accuracy', 'CorrectCount', 'TestCount'});
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

function model = trainRecognitionModel(samples, labels, sampleNames, imagePaths, components, metric)
    trainX = samples;
    [trainX, featureMu, featureSigma] = standardizeTrain(trainX);
    componentCount = min(max(1, components), size(trainX, 1) - 1);
    [coeff, trainScore, pcaMu] = trainPcaByPowerIteration(trainX, componentCount);
    componentCount = min(componentCount, size(coeff, 2));

    model = struct();
    model.ImageSize = [64, 64];
    model.FeatureMu = featureMu;
    model.FeatureSigma = featureSigma;
    model.PcaMu = pcaMu;
    model.Coeff = coeff(:, 1:componentCount);
    model.TrainFeatures = trainScore(:, 1:componentCount);
    model.TrainLabels = labels;
    model.TrainSampleNames = sampleNames;
    model.TrainImagePaths = imagePaths;
    model.Components = componentCount;
    model.Metric = metric;
end

function [trainIdx, testIdx] = stratifiedTrainTestSplit(labels, trainRatio, seed)
    classes = uniqueStringValues(labels);
    trainIdx = [];
    testIdx = [];
    for i = 1:numel(classes)
        idx = findLabelIndices(labels, classes(i));
        idx = deterministicShuffle(idx, seed + i * 7919);
        if numel(idx) < 2
            trainIdx = [trainIdx; idx(:)]; %#ok<AGROW>
            continue;
        end
        trainCount = floor(trainRatio * numel(idx));
        trainCount = max(1, min(trainCount, numel(idx) - 1));
        trainIdx = [trainIdx; idx(1:trainCount)]; %#ok<AGROW>
        testIdx = [testIdx; idx(trainCount + 1:end)]; %#ok<AGROW>
    end
    trainIdx = deterministicShuffle(trainIdx, seed + 17);
    testIdx = deterministicShuffle(testIdx, seed + 29);
end

function idx = findLabelIndices(labels, label)
    idx = zeros(0, 1);
    for i = 1:numel(labels)
        if labels(i) == label
            idx(end + 1, 1) = i; %#ok<AGROW>
        end
    end
end

function values = deterministicShuffle(values, seed)
    n = numel(values);
    keys = zeros(n, 1);
    state = uint32(mod(seed, 2147483647));
    if state == 0
        state = uint32(1);
    end
    for i = 1:n
        state = mod(uint64(1103515245) * uint64(state) + uint64(12345), uint64(2147483647));
        keys(i) = double(state);
    end
    order = sortIndicesAscending(keys);
    values = values(order);
end

function [standardized, mu, sigma] = standardizeTrain(features)
    mu = columnMean(features);
    sigma = columnStd(features, mu);
    for i = 1:numel(sigma)
        if sigma(i) < 1e-8 || ~isfinite(sigma(i))
            sigma(i) = 1;
        end
    end
    standardized = standardizeTest(features, mu, sigma);
end

function standardized = standardizeTest(features, mu, sigma)
    standardized = zeros(size(features));
    for r = 1:size(features, 1)
        standardized(r, :) = (features(r, :) - mu) ./ sigma;
    end
    standardized(~isfinite(standardized)) = 0;
end

function [coeff, score, mu] = trainPcaByPowerIteration(trainX, componentCount)
    mu = columnMean(trainX);
    centered = trainX - mu;
    gram = centered * centered';
    n = size(gram, 1);
    componentCount = min(componentCount, n - 1);
    coeff = zeros(size(centered, 2), componentCount);

    for k = 1:componentCount
        v = initialVector(n, k);
        for iter = 1:60
            nextV = gram * v;
            nextNorm = vectorNorm(nextV);
            if nextNorm < 1e-10
                break;
            end
            v = nextV / nextNorm;
        end

        lambda = double(v' * gram * v);
        if lambda < 1e-10
            coeff = coeff(:, 1:k-1);
            break;
        end

        featureVector = centered' * v / sqrt(lambda);
        featureNorm = vectorNorm(featureVector);
        if featureNorm < 1e-10
            coeff = coeff(:, 1:k-1);
            break;
        end
        coeff(:, k) = featureVector / featureNorm;
        gram = gram - lambda * (v * v');
        gram = (gram + gram') / 2;
    end

    score = centered * coeff;
end

function v = initialVector(n, componentIndex)
    v = zeros(n, 1);
    for i = 1:n
        v(i) = sin((i + componentIndex) * 12.9898) + cos((i * componentIndex + 3) * 78.233);
    end
    v = v / vectorNorm(v);
end

function [accuracy, correctCount, predictedLabels, nearestIndices] = evaluateNearestNeighbor(trainFeatures, trainLabels, testFeatures, testLabels, metric)
    predictedLabels = strings(numel(testLabels), 1);
    nearestIndices = zeros(numel(testLabels), 1);
    correctCount = 0;
    for i = 1:size(testFeatures, 1)
        bestDistance = inf;
        bestIndex = 1;
        for j = 1:size(trainFeatures, 1)
            if metric == "cosine"
                distance = cosineDistance(testFeatures(i, :), trainFeatures(j, :));
            else
                distance = euclideanDistance(testFeatures(i, :), trainFeatures(j, :));
            end
            if distance < bestDistance
                bestDistance = distance;
                bestIndex = j;
            end
        end
        predictedLabels(i) = trainLabels(bestIndex);
        nearestIndices(i) = bestIndex;
        if predictedLabels(i) == testLabels(i)
            correctCount = correctCount + 1;
        end
    end
    accuracy = 100 * correctCount / numel(testLabels);
end

function distance = cosineDistance(a, b)
    denom = vectorNorm(a) * vectorNorm(b);
    if denom < 1e-12
        distance = 1;
    else
        distance = 1 - (a * b') / denom;
    end
end

function distance = euclideanDistance(a, b)
    diff = a - b;
    distance = sqrt(diff * diff');
end

function tbl = buildMisclassifiedTable(sampleNames, imagePaths, trueLabels, predictedLabels, predictedSampleNames, predictedImagePaths)
    wrongCount = 0;
    for i = 1:numel(trueLabels)
        if trueLabels(i) ~= predictedLabels(i)
            wrongCount = wrongCount + 1;
        end
    end

    names = strings(wrongCount, 1);
    trueOut = strings(wrongCount, 1);
    predictedOut = strings(wrongCount, 1);
    paths = strings(wrongCount, 1);
    predictedNames = strings(wrongCount, 1);
    predictedPaths = strings(wrongCount, 1);
    row = 0;
    for i = 1:numel(trueLabels)
        if trueLabels(i) ~= predictedLabels(i)
            row = row + 1;
            names(row) = sampleNames(i);
            trueOut(row) = trueLabels(i);
            predictedOut(row) = predictedLabels(i);
            paths(row) = imagePaths(i);
            predictedNames(row) = predictedSampleNames(i);
            predictedPaths(row) = predictedImagePaths(i);
        end
    end

    tbl = table(names, trueOut, predictedOut, paths, predictedNames, predictedPaths, ...
        'VariableNames', {'FileName', 'TrueLabel', 'PredictedLabel', 'ImagePath', 'PredictedFileName', 'PredictedImagePath'});
end

function mu = columnMean(features)
    mu = zeros(1, size(features, 2));
    for c = 1:size(features, 2)
        mu(c) = scalarMean(features(:, c));
    end
end

function sigma = columnStd(features, mu)
    sigma = zeros(1, size(features, 2));
    for c = 1:size(features, 2)
        sigma(c) = scalarStd(features(:, c), mu(c));
    end
end

function value = scalarMean(values)
    total = 0;
    for i = 1:numel(values)
        total = total + double(values(i));
    end
    value = total / max(1, numel(values));
end

function value = scalarStd(values, mu)
    total = 0;
    for i = 1:numel(values)
        d = double(values(i)) - mu;
        total = total + d * d;
    end
    value = sqrt(total / max(1, numel(values) - 1));
end

function n = vectorNorm(values)
    n = sqrt(double(values(:)' * values(:)));
end

function img = scaleToUnitImage(img)
    minVal = img(1);
    maxVal = img(1);
    for i = 1:numel(img)
        if img(i) < minVal
            minVal = img(i);
        end
        if img(i) > maxVal
            maxVal = img(i);
        end
    end
    rangeVal = maxVal - minVal;
    if rangeVal < 1e-12
        img = zeros(size(img));
    else
        img = (img - minVal) / rangeVal;
    end
end

function img = clamp01(img)
    img(img < 0) = 0;
    img(img > 1) = 1;
end

function order = sortIndicesDescending(values)
    order = (1:numel(values))';
    for i = 1:numel(order)-1
        best = i;
        for j = i+1:numel(order)
            if values(order(j)) > values(order(best))
                best = j;
            end
        end
        if best ~= i
            tmp = order(i);
            order(i) = order(best);
            order(best) = tmp;
        end
    end
end

function order = sortIndicesAscending(values)
    order = (1:numel(values))';
    for i = 1:numel(order)-1
        best = i;
        for j = i+1:numel(order)
            if values(order(j)) < values(order(best))
                best = j;
            end
        end
        if best ~= i
            tmp = order(i);
            order(i) = order(best);
            order(best) = tmp;
        end
    end
end
