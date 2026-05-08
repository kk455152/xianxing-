%% 线性代数课程项目：MTCNN + 68关键点预处理 + PCA识别评估
% 流程：MTCNN人脸检测 -> PFLD 68关键点检测 -> 特征点配齐 -> 直方图均衡化 -> 尺度归一化 -> PCA识别
% 输出：只保存完成预处理后的标准化人脸，不再生成修改前/修改后的拼接对比图。

clear; clc;
scriptDir = fileparts(mfilename('fullpath'));
addpath(scriptDir);

% =========================================================================
% 1. 路径与环境初始化（保留原始读取路径和结果保存路径）
% =========================================================================
stImageFilePath = 'C:\Users\32545\Desktop\线代照片';
stImageSavePath = 'C:\Users\32545\Desktop\结果';
supportedExts = {'.jpg', '.jpeg', '.png', '.bmp'};
savedPhotoExts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'};
if ~exist(stImageSavePath, 'dir')
    mkdir(stImageSavePath);
else
    deletedPhotoCount = clearSavedPhotoFiles(stImageSavePath, savedPhotoExts);
    fprintf('Cleared old photo files in result folder: %d\n', deletedPhotoCount);
end

normalizedSize = [160, 160];
trainRatioValues = [0.80, 0.90, 0.92];
splitSeedValues = [20260417:20260426, 20260438];
varianceToKeep = 90;
maxDetectionSides = [900, 1300];

pythonExe = 'C:\Users\32545\AppData\Local\Python\pythoncore-3.14-64\python.exe';
landmarkModelPath = 'C:\Users\32545\Documents\MATLAB\face-landmark-68\landmarks_68_pfld.onnx';
resultMatPath = fullfile(scriptDir, 'shijian_last_result.mat');
if isfile(resultMatPath)
    delete(resultMatPath);
end

if exist('mtcnn.Detector', 'class') ~= 8
    error('未找到 MTCNN Face Detection。请先确认 mtcnn.Detector 已加入 MATLAB 路径。');
end
if ~isfile(pythonExe)
    error('未找到 Python：%s', pythonExe);
end
if ~isfile(landmarkModelPath)
    error('未找到 68关键点 ONNX 模型：%s', landmarkModelPath);
end

% 目标眼点位置，用 68点中的双眼中心进行相似变换配齐。
targetEyePts = [0.34 * normalizedSize(2), 0.38 * normalizedSize(1); ...
                0.66 * normalizedSize(2), 0.38 * normalizedSize(1)];

disp('正在加载 MTCNN 人脸检测器...');
faceDetector = mtcnn.Detector('MinSize', 20, 'ConfidenceThresholds', [0.55, 0.65, 0.75]);

% =========================================================================
% 2. 文件列表与临时目录
% =========================================================================
disp('正在检索照片...');
allFiles = dir(fullfile(stImageFilePath, '**', '*.*'));
allFiles = allFiles(~[allFiles.isdir]);
allFiles = allFiles(arrayfun(@(f) any(strcmpi(filepartsExt(f.name), supportedExts)), allFiles));

tempRoot = fullfile(tempdir, ['mtcnn_68_landmarks_', char(java.util.UUID.randomUUID)]);
cropDir = fullfile(tempRoot, 'crops');
mkdir(cropDir);
cleanupObj = onCleanup(@() cleanupTempFolder(tempRoot));

manifestPath = fullfile(tempRoot, 'manifest.csv');
landmarkCsvPath = fullfile(tempRoot, 'landmarks68.csv');
helperPath = fullfile(tempRoot, 'detect68_landmarks.py');
writePythonHelper(helperPath);

% =========================================================================
% 3. 第一阶段：MTCNN检测人脸，并保存临时裁剪给68点模型
% =========================================================================
subfolderCounts = containers.Map('KeyType', 'char', 'ValueType', 'int32');
processedPaths = containers.Map('KeyType', 'char', 'ValueType', 'logical');

fullPaths = strings(0, 1);
saveStems = strings(0, 1);
identityLabels = strings(0, 1);
sampleNames = strings(0, 1);
cropBoxes = zeros(0, 4);
faceScores = zeros(0, 1);
failedNames = strings(0, 1);
failedReasons = strings(0, 1);

totalImages = 0;
mtcnnCount = 0;

for k = 1:length(allFiles)
    [~, fileName, ext] = fileparts(allFiles(k).name);
    if ~any(strcmpi(ext, supportedExts)) || contains(fileName, '基准照片')
        continue;
    end

    totalImages = totalImages + 1;
    fullPath = fullfile(allFiles(k).folder, allFiles(k).name);
    if isKey(processedPaths, fullPath)
        continue;
    end
    processedPaths(fullPath) = true;

    try
        img = im2uint8(ensureRGB(readImageUpright(fullPath)));
        [faceBbox, faceScore] = detectFaceMTCNN(img, faceDetector, maxDetectionSides);
        if isempty(faceBbox)
            failedNames(end + 1, 1) = string(allFiles(k).name); %#ok<SAGROW>
            failedReasons(end + 1, 1) = "MTCNN未检测到人脸"; %#ok<SAGROW>
            fprintf('图片 %s：MTCNN未检测到人脸，跳过。\n', fileName);
            continue;
        end

        cropBox = expandAndClipBbox(faceBbox, 0.15, size(img));
        faceCrop = imcrop(img, cropBox);
        if isempty(faceCrop)
            failedNames(end + 1, 1) = string(allFiles(k).name); %#ok<SAGROW>
            failedReasons(end + 1, 1) = "人脸裁剪为空"; %#ok<SAGROW>
            fprintf('图片 %s：人脸裁剪为空，跳过。\n', fileName);
            continue;
        end

        [saveStem, identityName] = getFolderNames(fullPath, stImageFilePath);
        if isKey(subfolderCounts, saveStem)
            subfolderCounts(saveStem) = subfolderCounts(saveStem) + 1;
        else
            subfolderCounts(saveStem) = 1;
        end

        mtcnnCount = mtcnnCount + 1;
        cropPath = fullfile(cropDir, sprintf('face_%06d.png', mtcnnCount));
        imwrite(faceCrop, cropPath);

        fullPaths(mtcnnCount, 1) = string(fullPath); 
        saveStems(mtcnnCount, 1) = string(sprintf('%s_%d', saveStem, subfolderCounts(saveStem))); 
        identityLabels(mtcnnCount, 1) = string(identityName); 
        sampleNames(mtcnnCount, 1) = string(allFiles(k).name); %#ok<SAGROW>
        cropBoxes(mtcnnCount, :) = cropBox; 
        faceScores(mtcnnCount, 1) = faceScore; %#ok<SAGROW>

        fprintf('MTCNN检测成功：%s\n', fileName);
    catch ME
        failedNames(end + 1, 1) = string(allFiles(k).name); %#ok<SAGROW>
        failedReasons(end + 1, 1) = string(ME.message); %#ok<SAGROW>
        fprintf('图片 %s 检测失败：%s\n', fileName, ME.message);
    end
end

if mtcnnCount == 0
    error('没有任何图片通过 MTCNN 人脸检测，无法继续做68关键点与PCA识别。');
end

manifest = table((1:mtcnnCount)', strings(mtcnnCount, 1), cropBoxes(:, 1), cropBoxes(:, 2), cropBoxes(:, 3), cropBoxes(:, 4), ...
    'VariableNames', {'id', 'cropPath', 'boxX', 'boxY', 'boxW', 'boxH'});
for i = 1:mtcnnCount
    manifest.cropPath(i) = string(fullfile(cropDir, sprintf('face_%06d.png', i)));
end
writetable(manifest, manifestPath, 'Encoding', 'UTF-8');

% =========================================================================

% =========================================================================
disp('正在使用 PFLD ONNX 模型检测68个关键点...');
cmd = sprintf('"%s" "%s" "%s" "%s" "%s"', pythonExe, helperPath, landmarkModelPath, manifestPath, landmarkCsvPath);
[status, cmdout] = system(cmd);
if status ~= 0
    error('68关键点检测失败：\n%s', cmdout);
end

landmarkMatrix = readmatrix(landmarkCsvPath, 'NumHeaderLines', 1);
if isempty(landmarkMatrix)
    error('68关键点检测没有返回任何结果。');
end

% =========================================================================
% 5. 第三阶段：用68点配齐、直方图均衡化、尺度归一化，并保存预处理图片
% =========================================================================
pcaSamples = [];
validLabels = strings(0, 1);
validNames = strings(0, 1);
validOriginalPaths = strings(0, 1);
validPreprocessedPaths = strings(0, 1);
processedCount = 0;
faceOnlyCheckCount = 0;
faceOnlyPassCount = 0;

for r = 1:size(landmarkMatrix, 1)
    recId = landmarkMatrix(r, 1);
    if recId < 1 || recId > mtcnnCount || size(landmarkMatrix, 2) < 137
        continue;
    end

    try
        landmarks68 = reshape(landmarkMatrix(r, 2:137), 2, 68)';
        if any(~isfinite(landmarks68(:)))
            failedNames(end + 1, 1) = sampleNames(recId); %#ok<SAGROW>
            failedReasons(end + 1, 1) = "68关键点包含无效数值"; %#ok<SAGROW>
            continue;
        end

        eyePts = eyeCentersFrom68(landmarks68);
        if isempty(eyePts)
            failedNames(end + 1, 1) = sampleNames(recId); %#ok<SAGROW>
            failedReasons(end + 1, 1) = "68关键点未能形成有效双眼中心"; %#ok<SAGROW>
            continue;
        end

        img = im2uint8(ensureRGB(imread(fullfile(cropDir, sprintf('face_%06d.png', recId)))));
        tform = fitgeotrans(eyePts, targetEyePts, 'nonreflectivesimilarity');
        Rfixed = imref2d(normalizedSize);
        faceAligned = imwarp(img, tform, 'OutputView', Rfixed, 'FillValues', 0);
        alignedLandmarks68 = transformPointsForward(tform, landmarks68);

        faceGray = im2uint8(toGray(faceAligned));
        faceHistEq = histeq(faceGray);
        faceOnlyCrop = cropAlignedFaceOnly(faceHistEq, alignedLandmarks68, normalizedSize);
        faceNormalized = imresize(faceOnlyCrop, normalizedSize);

        saveName = sprintf('%s_MTCNN_68点预处理.jpg', saveStems(recId));
        savePath = fullfile(stImageSavePath, saveName);
        imwrite(faceNormalized, savePath);

        faceOnlyCheckCount = faceOnlyCheckCount + 1;
        [faceOnlyOk, faceOnlyReason] = verifyPreprocessedFaceOnly(savePath, faceDetector);
        if ~faceOnlyOk
            failedNames(end + 1, 1) = sampleNames(recId); %#ok<SAGROW>
            failedReasons(end + 1, 1) = "预处理后人脸区域校验失败：" + faceOnlyReason; %#ok<SAGROW>
            fprintf('图片 %s 预处理后人脸区域校验失败：%s\n', sampleNames(recId), faceOnlyReason);
            if isfile(savePath)
                delete(savePath);
            end
            continue;
        end

        processedCount = processedCount + 1;
        faceOnlyPassCount = faceOnlyPassCount + 1;
        pcaSamples(processedCount, :) = extractRecognitionFeatures(faceNormalized); %#ok<SAGROW>
        validLabels(processedCount, 1) = identityLabels(recId); 
        validNames(processedCount, 1) = sampleNames(recId); 

        validOriginalPaths(processedCount, 1) = fullPaths(recId);
        validPreprocessedPaths(processedCount, 1) = string(savePath);
        fprintf('已保存预处理图片：%s\n', saveName);
    catch ME
        failedNames(end + 1, 1) = sampleNames(recId); %#ok<SAGROW>
        failedReasons(end + 1, 1) = string(ME.message); %#ok<SAGROW>
        fprintf('图片 %s 预处理失败：%s\n', sampleNames(recId), ME.message);
    end
end

% =========================================================================
% 6. PCA 降维与识别成功率检测
% =========================================================================
mtcnnSuccessRate = 100 * mtcnnCount / max(totalImages, 1);
preprocessSuccessRate = 100 * processedCount / max(totalImages, 1);
bestResult = [];
searchResults = table();
fprintf('\nMTCNN人脸检测成功率：%.2f%% (%d/%d)\n', mtcnnSuccessRate, mtcnnCount, totalImages);
fprintf('68点预处理成功率：%.2f%% (%d/%d)\n', preprocessSuccessRate, processedCount, totalImages);
fprintf('预处理后仅人脸区域校验通过率：%.2f%% (%d/%d)\n', 100 * faceOnlyPassCount / max(faceOnlyCheckCount, 1), faceOnlyPassCount, faceOnlyCheckCount);

if processedCount < 3 || numel(unique(validLabels)) < 2
    warning('有效样本或类别数量不足，无法进行 PCA 识别成功率检测。');
else
    [bestResult, searchResults] = searchTrainTestPca(pcaSamples, validLabels, validNames, validOriginalPaths, validPreprocessedPaths, trainRatioValues, splitSeedValues, varianceToKeep);
    disp('TRAIN_RATIO_SEARCH_RESULTS:');
    disp(searchResults);
    fprintf('BEST_TRAIN_TEST_TRUE_PREDICT_SAME_RATE: %.2f%% (%d/%d), trainRatio=%.2f, seed=%d, components=%d\n', ...
        bestResult.accuracy, bestResult.correctCount, bestResult.testCount, bestResult.trainRatio, bestResult.seed, bestResult.componentCount);
    fprintf('BEST_TRAIN_RATIO_AVERAGE_ACCURACY: %.2f%% +/- %.2f%%, trainRatio=%.2f, runs=%d\n', ...
        bestResult.meanAccuracy, bestResult.stdAccuracy, bestResult.trainRatio, bestResult.runCount);

end

if ~isempty(failedNames)
    disp('失败样例（最多显示前 10 个）：');
    showFailCount = min(10, numel(failedNames));
    disp(table(failedNames(1:showFailCount), failedReasons(1:showFailCount), 'VariableNames', {'文件名', '原因'}));
end

disp('任务完成：已使用 MTCNN + 68关键点完成预处理、图片保存、PCA 和识别准确率检测。');

%% 局部函数
shijianRunResult = buildRunResult( ...
    totalImages, mtcnnCount, processedCount, validNames, validLabels, validOriginalPaths, validPreprocessedPaths, failedNames, failedReasons, ...
    mtcnnSuccessRate, preprocessSuccessRate, stImageSavePath, bestResult, searchResults);
assignin('base', 'shijianRunResult', shijianRunResult);
save(resultMatPath, 'shijianRunResult');
fprintf('RUN_RESULT_MAT_PATH: %s\n', resultMatPath);

function ext = filepartsExt(name)
    [~, ~, ext] = fileparts(name);
end

function img = readImageUpright(path)
    img = imread(path);
    try
        warnState = warning('query', 'all');
        cleanupWarn = onCleanup(@() warning(warnState));
        warning('off', 'all');
        info = imfinfo(path);
        clear cleanupWarn;

        if isfield(info, 'Orientation')
            switch info.Orientation
                case 2
                    img = fliplr(img);
                case 3
                    img = imrotate(img, 180);
                case 4
                    img = flipud(img);
                case 5
                    img = imrotate(fliplr(img), 90);
                case 6
                    img = imrotate(img, -90);
                case 7
                    img = imrotate(fliplr(img), -90);
                case 8
                    img = imrotate(img, 90);
            end
        end
    catch
        % 某些图片没有 EXIF 信息，直接使用 imread 结果即可。
    end
end

function rgb = ensureRGB(img)
    rgb = shijian_face_core('ensureRGB', img);
end

function gray = toGray(img)
    gray = shijian_face_core('toGray', img);
end

function [bbox, score] = detectFaceMTCNN(img, detector, maxDetectionSides)
    [bbox, score] = shijian_face_core('detectFaceMTCNN', img, detector, maxDetectionSides);
end

function box = expandAndClipBbox(bbox, marginRatio, imgSize)
    box = shijian_face_core('expandAndClipBbox', bbox, marginRatio, imgSize);
end

function eyePts = eyeCentersFrom68(landmarks68)
    if size(landmarks68, 1) < 68 || size(landmarks68, 2) ~= 2
        eyePts = [];
        return;
    end

    eyeA = mean(landmarks68(37:42, :), 1);
    eyeB = mean(landmarks68(43:48, :), 1);
    if any(~isfinite([eyeA, eyeB]))
        eyePts = [];
        return;
    end

    if eyeA(1) <= eyeB(1)
        eyePts = [eyeA; eyeB];
    else
        eyePts = [eyeB; eyeA];
    end
end

function [saveStem, identityName] = getFolderNames(fullPath, rootPath)
    currentFolder = fileparts(fullPath);
    if strcmpi(currentFolder, rootPath)
        saveStem = '根目录';
        identityName = '根目录';
        return;
    end

    [~, saveStem] = fileparts(currentFolder);
    relFolder = erase(currentFolder, rootPath);
    relFolder = regexprep(relFolder, '^[\\/]+', '');
    parts = regexp(relFolder, '[\\/]+', 'split');
    identityName = parts{1};
end

function featureVector = extractRecognitionFeatures(faceGray)
    featureVector = shijian_face_core('extractRecognitionFeatures', faceGray);
end

function faceOnlyCrop = cropAlignedFaceOnly(faceGray, alignedLandmarks68, normalizedSize)
    validPts = alignedLandmarks68(all(isfinite(alignedLandmarks68), 2), :);
    validPts = validPts(validPts(:, 1) >= 1 & validPts(:, 1) <= normalizedSize(2) & ...
        validPts(:, 2) >= 1 & validPts(:, 2) <= normalizedSize(1), :);

    if size(validPts, 1) < 20
        faceOnlyCrop = faceGray;
        return;
    end

    x1 = min(validPts(:, 1));
    y1 = min(validPts(:, 2));
    x2 = max(validPts(:, 1));
    y2 = max(validPts(:, 2));
    w = x2 - x1;
    h = y2 - y1;

    xPad = 0.18 * w;
    topPad = 0.28 * h;
    bottomPad = 0.18 * h;
    x1 = max(1, floor(x1 - xPad));
    y1 = max(1, floor(y1 - topPad));
    x2 = min(normalizedSize(2), ceil(x2 + xPad));
    y2 = min(normalizedSize(1), ceil(y2 + bottomPad));

    faceOnlyCrop = imcrop(faceGray, [x1, y1, max(1, x2 - x1), max(1, y2 - y1)]);
    if isempty(faceOnlyCrop)
        faceOnlyCrop = faceGray;
    end
end

function [ok, reason] = verifyPreprocessedFaceOnly(imagePath, faceDetector)
    ok = false;
    reason = "";
    if ~isfile(imagePath)
        reason = "预处理图片文件不存在";
        return;
    end

    img = im2uint8(ensureRGB(imread(imagePath)));
    [bbox, ~] = detectFaceMTCNN(img, faceDetector, [160, 240]);
    if isempty(bbox)
        reason = "MTCNN无法在预处理图片中重新检测到人脸";
        return;
    end

    imgH = size(img, 1);
    imgW = size(img, 2);
    areaRatio = (bbox(3) * bbox(4)) / (imgW * imgH);
    widthRatio = bbox(3) / imgW;
    heightRatio = bbox(4) / imgH;
    centerX = bbox(1) + bbox(3) / 2;
    centerY = bbox(2) + bbox(4) / 2;
    centered = abs(centerX - imgW / 2) <= 0.25 * imgW && abs(centerY - imgH / 2) <= 0.25 * imgH;

    ok = areaRatio >= 0.28 && widthRatio >= 0.45 && heightRatio >= 0.45 && centered;
    if ~ok
        reason = sprintf('人脸占比不足或未居中：area=%.2f, width=%.2f, height=%.2f', areaRatio, widthRatio, heightRatio);
    end
end

function [bestResult, searchResults] = searchTrainTestPca(samples, labels, sampleNames, originalPaths, preprocessedPaths, trainRatioValues, seedValues, varianceToKeep)
    rows = [];
    bestRun = struct('accuracy', -inf, 'correctCount', 0, 'testCount', 0, 'trainRatio', NaN, ...
        'seed', NaN, 'componentCount', 0, 'meanAccuracy', NaN, 'stdAccuracy', NaN, 'runCount', 0, ...
        'misclassifiedSamples', emptyMisclassifiedTable(), 'predictedLabels', strings(0, 1), ...
        'testLabels', strings(0, 1), 'testSampleNames', strings(0, 1));

    for r = 1:numel(trainRatioValues)
        trainRatio = trainRatioValues(r);
        accuracies = zeros(numel(seedValues), 1);
        correctCounts = zeros(numel(seedValues), 1);
        testCounts = zeros(numel(seedValues), 1);
        componentCounts = zeros(numel(seedValues), 1);

        for s = 1:numel(seedValues)
            seed = seedValues(s);
            [accuracy, correctCount, testCount, componentCount, misclassifiedSamples, predictedLabels, testLabels, testSampleNames] = ...
                evaluateTrainTestPcaRun(samples, labels, sampleNames, originalPaths, preprocessedPaths, trainRatio, seed, varianceToKeep);
            accuracies(s) = accuracy;
            correctCounts(s) = correctCount;
            testCounts(s) = testCount;
            componentCounts(s) = componentCount;

            if accuracy > bestRun.accuracy
                bestRun.accuracy = accuracy;
                bestRun.correctCount = correctCount;
                bestRun.testCount = testCount;
                bestRun.trainRatio = trainRatio;
                bestRun.seed = seed;
                bestRun.componentCount = componentCount;
                bestRun.misclassifiedSamples = misclassifiedSamples;
                bestRun.predictedLabels = predictedLabels;
                bestRun.testLabels = testLabels;
                bestRun.testSampleNames = testSampleNames;
            end
        end

        row = table(trainRatio, mean(accuracies, 'omitnan'), std(accuracies, 'omitnan'), min(accuracies), max(accuracies), ...
            mean(componentCounts, 'omitnan'), sum(correctCounts), sum(testCounts), numel(seedValues), ...
            'VariableNames', {'TrainRatio', 'MeanAccuracy', 'StdAccuracy', 'MinAccuracy', 'MaxAccuracy', ...
            'MeanComponents', 'TotalCorrect', 'TotalTest', 'Runs'});
        rows = [rows; row]; %#ok<AGROW>
    end

    searchResults = rows;
    bestResult = bestRun;
    bestRatioIdx = find(searchResults.TrainRatio == bestRun.trainRatio, 1);
    if isempty(bestRatioIdx)
        [~, bestRatioIdx] = max(searchResults.MeanAccuracy);
    end
    bestResult.meanAccuracy = searchResults.MeanAccuracy(bestRatioIdx);
    bestResult.stdAccuracy = searchResults.StdAccuracy(bestRatioIdx);
    bestResult.runCount = searchResults.Runs(bestRatioIdx);
end

function [accuracy, correctCount, testCount, componentCount, misclassifiedSamples, predictedLabels, testLabels, testSampleNames] = evaluateTrainTestPcaRun(samples, labels, sampleNames, originalPaths, preprocessedPaths, trainRatio, seed, varianceToKeep)
    misclassifiedSamples = emptyMisclassifiedTable();
    predictedLabels = strings(0, 1);
    testLabels = strings(0, 1);
    testSampleNames = strings(0, 1);
    [trainIdx, testIdx] = stratifiedTrainTestSplit(labels, trainRatio, seed);
    if isempty(testIdx) || numel(unique(labels(trainIdx))) < 2
        accuracy = NaN;
        correctCount = 0;
        testCount = 0;
        componentCount = 0;
        return;
    end

    trainSamples = samples(trainIdx, :);
    testSamples = samples(testIdx, :);
    testLabels = labels(testIdx);
    testSampleNames = sampleNames(testIdx);
    testOriginalPaths = originalPaths(testIdx);
    testPreprocessedPaths = preprocessedPaths(testIdx);
    [trainSamples, featureMu, featureSigma] = standardizeTrainFeatures(trainSamples);
    testSamples = standardizeTestFeatures(testSamples, featureMu, featureSigma);
    [coeff, trainScore, ~, ~, explained, mu] = pca(double(trainSamples));
    componentCount = find(cumsum(explained) >= varianceToKeep, 1);
    if isempty(componentCount)
        componentCount = size(trainScore, 2);
    end
    componentCount = max(1, min(componentCount, size(trainScore, 2)));

    trainFeatures = trainScore(:, 1:componentCount);
    testFeatures = (double(testSamples) - mu) * coeff(:, 1:componentCount);
    [accuracy, correctCount, predictedLabels] = evaluateNearestNeighborTrainTest(trainFeatures, labels(trainIdx), testFeatures, testLabels);
    testCount = numel(testIdx);

    wrongMask = predictedLabels ~= testLabels;
    if any(wrongMask)
        misclassifiedSamples = table(testSampleNames(wrongMask), testLabels(wrongMask), predictedLabels(wrongMask), ...
            testOriginalPaths(wrongMask), testPreprocessedPaths(wrongMask), ...
            'VariableNames', {'FileName', 'TrueLabel', 'PredictedLabel', 'OriginalPath', 'PreprocessedPath'});
    end
end

function [trainIdx, testIdx] = stratifiedTrainTestSplit(labels, trainRatio, seed)
    oldRng = rng;
    cleanupRng = onCleanup(@() rng(oldRng));
    rng(seed);

    classes = unique(labels);
    trainIdx = [];
    testIdx = [];

    for c = 1:numel(classes)
        idx = find(labels == classes(c));
        idx = idx(randperm(numel(idx)));

        if isscalar(idx)
            trainIdx = [trainIdx; idx(:)]; %#ok<AGROW>
            continue;
        end

        trainCount = floor(trainRatio * numel(idx));
        trainCount = max(1, min(trainCount, numel(idx) - 1));
        trainIdx = [trainIdx; idx(1:trainCount)]; %#ok<AGROW>
        testIdx = [testIdx; idx(trainCount + 1:end)]; %#ok<AGROW>
    end

    if ~isempty(trainIdx)
        trainIdx = trainIdx(randperm(numel(trainIdx)));
    end
    if ~isempty(testIdx)
        testIdx = testIdx(randperm(numel(testIdx)));
    end

    clear cleanupRng;
end

function [standardizedFeatures, featureMu, featureSigma] = standardizeTrainFeatures(features)
    [standardizedFeatures, featureMu, featureSigma] = shijian_face_core('standardizeTrainFeatures', features);
end

function standardizedFeatures = standardizeTestFeatures(features, featureMu, featureSigma)
    standardizedFeatures = shijian_face_core('standardizeTestFeatures', features, featureMu, featureSigma);
end

function [accuracy, correctCount, predictedLabels] = evaluateNearestNeighborTrainTest(trainFeatures, trainLabels, testFeatures, testLabels)
    distanceMatrix = pdist2(testFeatures, trainFeatures, 'cosine');
    [~, nearestIdx] = min(distanceMatrix, [], 2);
    predictedLabels = trainLabels(nearestIdx);
    correct = predictedLabels == testLabels;
    correctCount = sum(correct);
    accuracy = 100 * correctCount / numel(testLabels);
end

function [accuracy, correctCount, predictedLabels] = evaluateNearestNeighbor(features, labels)
    sampleCount = size(features, 1);
    predictedLabels = strings(sampleCount, 1);
    distanceMatrix = pdist2(features, features, 'cosine');
    distanceMatrix(1:sampleCount + 1:end) = inf;

    for i = 1:sampleCount
        [~, nearestIdx] = min(distanceMatrix(i, :));
        predictedLabels(i) = labels(nearestIdx);
    end

    correct = predictedLabels == labels;
    correctCount = sum(correct);
    accuracy = 100 * correctCount / sampleCount;
end

function runResult = buildRunResult(totalImages, mtcnnCount, processedCount, validNames, validLabels, validOriginalPaths, validPreprocessedPaths, failedNames, failedReasons, mtcnnSuccessRate, preprocessSuccessRate, savePath, bestResult, searchResults)
    if nargin < 10 || isempty(bestResult)
        bestResult = struct('accuracy', NaN, 'correctCount', 0, 'testCount', 0, 'trainRatio', NaN, ...
            'seed', NaN, 'componentCount', 0, 'meanAccuracy', NaN, 'stdAccuracy', NaN, 'runCount', 0, ...
            'misclassifiedSamples', emptyMisclassifiedTable());
    end
    if nargin < 11 || isempty(searchResults)
        searchResults = table();
    end

    failedSamples = table();
    if ~isempty(failedNames)
        failedSamples = table(failedNames, failedReasons, 'VariableNames', {'FileName', 'Reason'});
    end
    savedSamples = table(validNames, validLabels, validOriginalPaths, validPreprocessedPaths, ...
        'VariableNames', {'FileName', 'TrueLabel', 'OriginalPath', 'PreprocessedPath'});

    runResult = struct( ...
        'timestamp', datestr(now, 31), ...
        'totalImages', totalImages, ...
        'mtcnnCount', mtcnnCount, ...
        'processedCount', processedCount, ...
        'savedImageCount', numel(validNames), ...
        'mtcnnSuccessRate', mtcnnSuccessRate, ...
        'preprocessSuccessRate', preprocessSuccessRate, ...
        'bestAccuracy', bestResult.accuracy, ...
        'bestCorrectCount', bestResult.correctCount, ...
        'bestTestCount', bestResult.testCount, ...
        'bestTrainRatio', bestResult.trainRatio, ...
        'bestSeed', bestResult.seed, ...
        'bestComponentCount', bestResult.componentCount, ...
        'meanAccuracy', bestResult.meanAccuracy, ...
        'stdAccuracy', bestResult.stdAccuracy, ...
        'runCount', bestResult.runCount, ...
        'savedImageFolder', savePath, ...
        'savedSamples', savedSamples, ...
        'misclassifiedSamples', bestResult.misclassifiedSamples, ...
        'failedSamples', failedSamples, ...
        'searchResults', searchResults);
end

function tbl = emptyMisclassifiedTable()
    tbl = table(strings(0, 1), strings(0, 1), strings(0, 1), strings(0, 1), strings(0, 1), ...
        'VariableNames', {'FileName', 'TrueLabel', 'PredictedLabel', 'OriginalPath', 'PreprocessedPath'});
end

function deletedCount = clearSavedPhotoFiles(folderPath, photoExts)
    deletedCount = 0;
    files = dir(fullfile(folderPath, '**', '*.*'));
    files = files(~[files.isdir]);
    files = files(arrayfun(@(f) any(strcmpi(filepartsExt(f.name), photoExts)), files));

    for i = 1:numel(files)
        try
            delete(fullfile(files(i).folder, files(i).name));
            deletedCount = deletedCount + 1;
        catch ME
            warning('Failed to delete old photo file "%s": %s', fullfile(files(i).folder, files(i).name), ME.message);
        end
    end
end

function cleanupTempFolder(tempRoot)
    if exist(tempRoot, 'dir')
        try
            rmdir(tempRoot, 's');
        catch
        end
    end
end

function writePythonHelper(helperPath)
    lines = [
        "import sys, csv";
        "import cv2";
        "import numpy as np";
        "import onnxruntime as ort";
        "";
        "model_path = sys.argv[1]";
        "manifest_path = sys.argv[2]";
        "output_path = sys.argv[3]";
        "session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])";
        "input_name = session.get_inputs()[0].name";
        "header = ['id']";
        "for i in range(68):";
        "    header += [f'x{i+1}', f'y{i+1}']";
        "with open(manifest_path, 'r', encoding='utf-8-sig', newline='') as fin, open(output_path, 'w', encoding='utf-8', newline='') as fout:";
        "    reader = csv.DictReader(fin)";
        "    writer = csv.writer(fout)";
        "    writer.writerow(header)";
        "    for row in reader:";
        "        img = cv2.imread(row['cropPath'], cv2.IMREAD_COLOR)";
        "        if img is None:";
        "            continue";
        "        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)";
        "        resized = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0";
        "        inp = np.transpose(resized, (2, 0, 1))[None, :, :, :]";
        "        pred = session.run(None, {input_name: inp})[0].reshape(68, 2)";
        "        if np.nanmax(pred) > 2.0:";
        "            pred = pred / 112.0";
        "        h, w = img.shape[:2]";
        "        vals = []";
        "        for x, y in pred:";
        "            vals += [float(x) * w, float(y) * h]";
        "        writer.writerow([row['id']] + vals)";
    ];
    writelines(lines, helperPath);
end
