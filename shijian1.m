%% 线性代数课程项目：快速人脸预处理
% 流程：OpenCV/Viola-Jones级联检测 -> 肤色区域兜底 -> 人脸紧裁剪 -> 直方图均衡化 -> 尺度归一化
% 说明：本脚本已停用 MTCNN 和批量68点候选推理，目标是在保证人脸区域占比的同时缩短运行时间。

clear; clc;
scriptTimer = tic;
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
targetFaceOccupancy = 0.96;
minimumVerifiedFaceOccupancy = 0.95;
resultMatPath = fullfile(scriptDir, 'shijian_last_result.mat');
if isfile(resultMatPath)
    delete(resultMatPath);
end

% =========================================================================
% 2. 文件列表
% =========================================================================
disp('正在检索照片...');
allFiles = dir(fullfile(stImageFilePath, '**', '*.*'));
allFiles = allFiles(~[allFiles.isdir]);
allFiles = allFiles(arrayfun(@(f) any(strcmpi(filepartsExt(f.name), supportedExts)), allFiles));

subfolderCounts = containers.Map('KeyType', 'char', 'ValueType', 'int32');
processedPaths = containers.Map('KeyType', 'char', 'ValueType', 'logical');

validLabels = strings(0, 1);
validNames = strings(0, 1);
validOriginalPaths = strings(0, 1);
validPreprocessedPaths = strings(0, 1);
validFaceOccupancies = zeros(0, 1);
validPreprocessMethods = strings(0, 1);
validQualityScores = zeros(0, 1);
validRepairFlags = false(0, 1);
failedNames = strings(0, 1);
failedReasons = strings(0, 1);

totalImages = 0;
imageCount = 0;
processedCount = 0;
faceOnlyCheckCount = 0;
faceOnlyPassCount = 0;
mtcnnCount = 0;
mtcnnSuccessRate = 0;

% =========================================================================
% 3. 快速预处理：级联检测 + 肤色兜底 + 尺度归一化
% =========================================================================
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
        [saveStem, identityName] = getFolderNames(fullPath, stImageFilePath);
        if isKey(subfolderCounts, saveStem)
            subfolderCounts(saveStem) = subfolderCounts(saveStem) + 1;
        else
            subfolderCounts(saveStem) = 1;
        end

        imageCount = imageCount + 1;
        sampleName = string(allFiles(k).name);
        saveStemWithIndex = string(sprintf('%s_%d', saveStem, subfolderCounts(saveStem)));

        [faceNormalized, faceCropMeta, preprocessMethod] = fastFacePreprocess(img, normalizedSize, targetFaceOccupancy, "default");
        saveName = sprintf('%s_fast_face.jpg', saveStemWithIndex);
        savePath = fullfile(stImageSavePath, saveName);
        imwrite(faceNormalized, savePath);

        faceOnlyCheckCount = faceOnlyCheckCount + 1;
        [faceOnlyOk, faceOnlyReason, faceOccupancy] = verifyPreprocessedFaceOnly(savePath, faceCropMeta, minimumVerifiedFaceOccupancy);
        if ~faceOnlyOk
            failedNames(end + 1, 1) = sampleName; %#ok<SAGROW>
            failedReasons(end + 1, 1) = "预处理后人脸区域校验失败：" + faceOnlyReason; %#ok<SAGROW>
            if isfile(savePath)
                delete(savePath);
            end
            continue;
        end

        processedCount = processedCount + 1;
        faceOnlyPassCount = faceOnlyPassCount + 1;
        validLabels(processedCount, 1) = string(identityName);
        validNames(processedCount, 1) = sampleName;
        validOriginalPaths(processedCount, 1) = string(fullPath);
        validPreprocessedPaths(processedCount, 1) = string(savePath);
        validFaceOccupancies(processedCount, 1) = faceOccupancy;
        validPreprocessMethods(processedCount, 1) = preprocessMethod;
        [~, initialQualityScore] = assessSavedFaceQuality(savePath);
        validQualityScores(processedCount, 1) = initialQualityScore;
        validRepairFlags(processedCount, 1) = false;
        fprintf('已保存预处理图片：%s，方法：%s，占比：%.2f%%\n', saveName, preprocessMethod, 100 * faceOccupancy);
    catch ME
        failedNames(end + 1, 1) = string(allFiles(k).name); %#ok<SAGROW>
        failedReasons(end + 1, 1) = string(ME.message); %#ok<SAGROW>
        fprintf('图片 %s 预处理失败：%s\n', fileName, ME.message);
    end
end

% =========================================================================
% 4. 预处理结果统计
% =========================================================================
[validPreprocessedPaths, validFaceOccupancies, validPreprocessMethods, validQualityScores, validRepairFlags, repairedCount, deletedLowQualityCount] = ...
    repairLowQualitySavedFaces(validNames, validOriginalPaths, validPreprocessedPaths, validFaceOccupancies, validPreprocessMethods, validQualityScores, validRepairFlags, normalizedSize, targetFaceOccupancy, minimumVerifiedFaceOccupancy);

preprocessCoverageRate = 100 * imageCount / max(totalImages, 1);
preprocessSuccessRate = 100 * processedCount / max(totalImages, 1);
fprintf('\nMTCNN人脸检测：已停用\n');
fprintf('快速预处理覆盖率：%.2f%% (%d/%d)\n', preprocessCoverageRate, imageCount, totalImages);
fprintf('快速人脸预处理成功率：%.2f%% (%d/%d)\n', preprocessSuccessRate, processedCount, totalImages);
fprintf('预处理后仅人脸区域校验通过率：%.2f%% (%d/%d)\n', 100 * faceOnlyPassCount / max(faceOnlyCheckCount, 1), faceOnlyPassCount, faceOnlyCheckCount);

fprintf('LOW_QUALITY_DELETED_AND_REPROCESSED: %d, REPAIRED_SAVED: %d\n', deletedLowQualityCount, repairedCount);

scriptRuntimeSeconds = toc(scriptTimer);
fprintf('SCRIPT_RUNTIME_SECONDS: %.2f\n', scriptRuntimeSeconds);

if ~isempty(failedNames)
    disp('失败样例（最多显示前10个）：');
    showFailCount = min(10, numel(failedNames));
    disp(table(failedNames(1:showFailCount), failedReasons(1:showFailCount), 'VariableNames', {'文件名', '原因'}));
end

disp('任务完成：已停用 MTCNN/批量68点候选推理，并使用快速级联检测、肤色兜底裁剪、直方图均衡化和尺度归一化完成预处理。');

shijianRunResult = buildRunResult( ...
    totalImages, mtcnnCount, processedCount, validNames, validLabels, validOriginalPaths, validPreprocessedPaths, validFaceOccupancies, validPreprocessMethods, validQualityScores, validRepairFlags, failedNames, failedReasons, ...
    mtcnnSuccessRate, preprocessSuccessRate, stImageSavePath, scriptRuntimeSeconds);
assignin('base', 'shijianRunResult', shijianRunResult);
save(resultMatPath, 'shijianRunResult');
fprintf('RUN_RESULT_MAT_PATH: %s\n', resultMatPath);

%% 局部函数
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
        % 部分图片没有 EXIF 信息，直接使用 imread 结果即可。
    end
end

function rgb = ensureRGB(img)
    rgb = shijian_face_core('ensureRGB', img);
end

function gray = toGray(img)
    gray = shijian_face_core('toGray', img);
end

function [faceNormalized, cropMeta, methodName] = fastFacePreprocess(img, normalizedSize, targetFaceOccupancy, mode)
    if nargin < 4
        mode = "default";
    end
    img = im2uint8(ensureRGB(img));
    if mode == "repair"
        [faceBox, methodName] = detectRepairFaceBox(img);
    else
        [faceBox, methodName] = detectFastFaceBox(img);
    end
    cropBox = faceCropBoxFromFaceBox(faceBox, size(img), targetFaceOccupancy);
    faceCrop = imcrop(img, cropBox);
    if isempty(faceCrop)
        cropBox = [1, 1, size(img, 2), size(img, 1)];
        faceBox = cropBox;
        methodName = "full_image_fallback";
        faceCrop = img;
    end

    faceGray = im2uint8(toGray(faceCrop));
    faceGray = localContrastNormalize(faceGray);
    faceNormalized = imresize(faceGray, normalizedSize);

    cropMeta = struct();
    cropMeta.valid = true;
    cropMeta.method = methodName;
    cropMeta.faceBox = faceBox;
    cropMeta.cropBox = cropBox;
    cropMeta.occupancy = faceOccupancyInCrop(faceBox, cropBox);
end

function [bbox, methodName] = detectFastFaceBox(img)
    [bbox, ok] = detectCascadeFaceBox(img);
    if ok
        methodName = "cascade";
        return;
    end

    [bbox, ok] = detectSkinFaceBox(img);
    if ok
        methodName = "skin_region";
        return;
    end

    bbox = fallbackFaceBox(size(img));
    methodName = "center_fallback";
end

function [bbox, methodName] = detectRepairFaceBox(img)
    candidates = zeros(0, 4);
    methods = strings(0, 1);

    [box, ok] = detectCascadeFaceBox(img);
    if ok
        candidates(end + 1, :) = box; %#ok<AGROW>
        methods(end + 1, 1) = "repair_cascade"; %#ok<AGROW>
    end

    [box, ok] = detectSkinFaceBox(img);
    if ok
        candidates(end + 1, :) = box; %#ok<AGROW>
        methods(end + 1, 1) = "repair_skin_region"; %#ok<AGROW>
    end

    candidates(end + 1, :) = fallbackFaceBox(size(img)); %#ok<AGROW>
    methods(end + 1, 1) = "repair_center_fallback"; %#ok<AGROW>

    imgSize = size(img);
    bestScore = -inf;
    bbox = candidates(1, :);
    methodName = methods(1);
    for i = 1:size(candidates, 1)
        b = clipRect(candidates(i, :), imgSize);
        crop = imcrop(img, faceCropBoxFromFaceBox(b, imgSize, 0.97));
        if isempty(crop)
            continue;
        end
        gray = localContrastNormalize(im2uint8(toGray(crop)));
        gray = imresize(gray, [160, 160]);
        score = assessFaceImageQuality(gray);
        if score > bestScore
            bestScore = score;
            bbox = b;
            methodName = methods(i);
        end
    end
end

function [validPreprocessedPaths, validFaceOccupancies, validPreprocessMethods, validQualityScores, validRepairFlags, repairedCount, deletedLowQualityCount] = repairLowQualitySavedFaces(validNames, validOriginalPaths, validPreprocessedPaths, validFaceOccupancies, validPreprocessMethods, validQualityScores, validRepairFlags, normalizedSize, targetFaceOccupancy, minimumVerifiedFaceOccupancy)
    qualityThreshold = 0.48;
    repairedCount = 0;
    deletedLowQualityCount = 0;

    for i = 1:numel(validPreprocessedPaths)
        savePath = char(validPreprocessedPaths(i));
        [qualityOk, qualityScore] = assessSavedFaceQuality(savePath);
        validQualityScores(i, 1) = qualityScore;
        if qualityOk
            continue;
        end

        deletedLowQualityCount = deletedLowQualityCount + 1;
        try
            originalImg = im2uint8(ensureRGB(readImageUpright(char(validOriginalPaths(i)))));
            if isfile(savePath)
                delete(savePath);
            end
            [faceNormalized, faceCropMeta, preprocessMethod] = fastFacePreprocess(originalImg, normalizedSize, max(targetFaceOccupancy, 0.97), "repair");
            imwrite(faceNormalized, savePath);
            [faceOnlyOk, ~, faceOccupancy] = verifyPreprocessedFaceOnly(savePath, faceCropMeta, minimumVerifiedFaceOccupancy);
            [repairQualityOk, repairQualityScore] = assessSavedFaceQuality(savePath);
            if faceOnlyOk && (repairQualityOk || repairQualityScore >= qualityScore)
                validFaceOccupancies(i, 1) = faceOccupancy;
                validPreprocessMethods(i, 1) = preprocessMethod;
                validQualityScores(i, 1) = repairQualityScore;
                validRepairFlags(i, 1) = true;
                repairedCount = repairedCount + 1;
            else
                if isfile(savePath)
                    delete(savePath);
                end
                [faceNormalized, faceCropMeta, preprocessMethod] = fastFacePreprocess(originalImg, normalizedSize, targetFaceOccupancy, "default");
                imwrite(faceNormalized, savePath);
                [~, ~, faceOccupancy] = verifyPreprocessedFaceOnly(savePath, faceCropMeta, minimumVerifiedFaceOccupancy);
                [~, fallbackQualityScore] = assessSavedFaceQuality(savePath);
                validFaceOccupancies(i, 1) = faceOccupancy;
                validPreprocessMethods(i, 1) = preprocessMethod + "_restored";
                validQualityScores(i, 1) = fallbackQualityScore;
            end
            fprintf('REPROCESS_LOW_QUALITY_FACE: %s, oldScore=%.3f, newScore=%.3f\n', validNames(i), qualityScore, validQualityScores(i, 1));
        catch ME
            warning('Failed to reprocess low quality face "%s": %s', savePath, ME.message);
            try
                originalImg = im2uint8(ensureRGB(readImageUpright(char(validOriginalPaths(i)))));
                [faceNormalized, faceCropMeta, preprocessMethod] = fastFacePreprocess(originalImg, normalizedSize, targetFaceOccupancy, "default");
                imwrite(faceNormalized, savePath);
                [~, ~, faceOccupancy] = verifyPreprocessedFaceOnly(savePath, faceCropMeta, minimumVerifiedFaceOccupancy);
                [~, restoredQualityScore] = assessSavedFaceQuality(savePath);
                validFaceOccupancies(i, 1) = faceOccupancy;
                validPreprocessMethods(i, 1) = preprocessMethod + "_exception_restored";
                validQualityScores(i, 1) = restoredQualityScore;
            catch restoreME
                warning('Failed to restore low quality face "%s": %s', savePath, restoreME.message);
            end
        end
    end

    finalLowQuality = validQualityScores < qualityThreshold;
    if any(finalLowQuality)
        fprintf('LOW_QUALITY_AFTER_REPROCESS_COUNT: %d\n', sum(finalLowQuality));
    end
end

function [ok, score] = assessSavedFaceQuality(imagePath)
    ok = false;
    score = 0;
    if ~isfile(imagePath)
        return;
    end
    try
        img = imread(imagePath);
        score = assessFaceImageQuality(img);
        ok = score >= 0.48;
    catch
        score = 0;
    end
end

function score = assessFaceImageQuality(img)
    gray = im2single(toGray(img));
    if isempty(gray)
        score = 0;
        return;
    end

    lapKernel = [0 1 0; 1 -4 1; 0 1 0];
    lap = imfilter(gray, lapKernel, 'replicate');
    sharpness = var(lap(:), 0);
    sharpScore = min(1, sharpness / 0.006);
    contrastScore = min(1, std(gray(:), 0) / 0.22);
    entropyScore = min(1, entropy(im2uint8(gray)) / 7.2);
    meanVal = mean(gray(:));
    brightnessScore = max(0, 1 - abs(meanVal - 0.50) / 0.42);
    borderWidth = max(2, round(min(size(gray, 1), size(gray, 2)) * 0.06));
    topPixels = gray(1:borderWidth, :);
    bottomPixels = gray(end-borderWidth+1:end, :);
    leftPixels = gray(:, 1:borderWidth);
    rightPixels = gray(:, end-borderWidth+1:end);
    borderPixels = [topPixels(:); bottomPixels(:); leftPixels(:); rightPixels(:)];
    borderPenalty = max(0, 1 - mean(borderPixels(:)));

    score = 0.34 * sharpScore + 0.28 * contrastScore + 0.22 * entropyScore + 0.12 * brightnessScore + 0.04 * borderPenalty;
    score = max(0, min(1, score));
end

function [bestBox, ok] = detectCascadeFaceBox(img)
    persistent detectors detectorReady
    ok = false;
    bestBox = [];

    if isempty(detectorReady)
        detectorReady = false;
        detectors = {};
        detectorNames = {'FrontalFaceCART', 'FrontalFaceLBP', 'ProfileFace'};
        for i = 1:numel(detectorNames)
            try
                d = vision.CascadeObjectDetector(detectorNames{i});
                d.MergeThreshold = 4;
                d.ScaleFactor = 1.08;
                detectors{end + 1} = d; %#ok<AGROW>
            catch
            end
        end
        detectorReady = ~isempty(detectors);
    end

    if ~detectorReady
        return;
    end

    gray = toGray(img);
    scale = min(1, 640 / max(size(gray, 1), size(gray, 2)));
    if scale < 1
        detectImg = imresize(gray, scale);
    else
        detectImg = gray;
    end

    allBoxes = zeros(0, 4);
    for i = 1:numel(detectors)
        try
            b = step(detectors{i}, detectImg);
            if ~isempty(b)
                allBoxes = [allBoxes; double(b) ./ scale]; %#ok<AGROW>
            end
        catch
        end
    end

    if isempty(allBoxes)
        return;
    end

    bestBox = selectBestFaceBox(allBoxes, size(img));
    ok = ~isempty(bestBox);
end

function [bbox, ok] = detectSkinFaceBox(img)
    ok = false;
    bbox = [];
    try
        rgb = im2uint8(ensureRGB(img));
        ycbcr = rgb2ycbcr(rgb);
        cb = ycbcr(:, :, 2);
        cr = ycbcr(:, :, 3);
        mask = cb >= 75 & cb <= 135 & cr >= 130 & cr <= 180;
        mask = imclose(mask, strel('disk', 5));
        mask = imopen(mask, strel('disk', 3));
        mask = imfill(mask, 'holes');
        mask = bwareaopen(mask, max(80, round(numel(mask) * 0.002)));
        cc = bwconncomp(mask);
        if cc.NumObjects == 0
            return;
        end

        stats = regionprops(cc, 'BoundingBox', 'Area', 'Centroid');
        imgH = size(img, 1);
        imgW = size(img, 2);
        bestScore = -inf;
        for i = 1:numel(stats)
            b = stats(i).BoundingBox;
            areaRatio = stats(i).Area / (imgH * imgW);
            if areaRatio < 0.01 || areaRatio > 0.75
                continue;
            end
            aspect = b(3) / max(b(4), 1);
            if aspect < 0.35 || aspect > 1.80
                continue;
            end
            centerPenalty = abs(stats(i).Centroid(1) - imgW / 2) / imgW + 0.7 * abs(stats(i).Centroid(2) - imgH * 0.42) / imgH;
            score = areaRatio - 0.18 * centerPenalty - 0.05 * abs(aspect - 0.85);
            if score > bestScore
                bestScore = score;
                bbox = b;
            end
        end

        if isempty(bbox)
            return;
        end
        bbox = clipRect(bbox, [imgH, imgW]);
        ok = bbox(3) >= 30 && bbox(4) >= 30;
    catch
        ok = false;
        bbox = [];
    end
end

function bbox = fallbackFaceBox(imgSize)
    h = double(imgSize(1));
    w = double(imgSize(2));
    if h >= w
        boxW = 0.82 * w;
        boxH = min(0.70 * h, 1.18 * boxW);
        cx = 0.50 * w;
        cy = 0.38 * h;
    else
        boxH = 0.82 * h;
        boxW = min(0.56 * w, 0.90 * boxH);
        cx = 0.50 * w;
        cy = 0.46 * h;
    end
    bbox = clipRect([cx - boxW / 2, cy - boxH / 2, boxW, boxH], [h, w]);
end

function bestBox = selectBestFaceBox(boxes, imgSize)
    h = double(imgSize(1));
    w = double(imgSize(2));
    bestScore = -inf;
    bestBox = [];
    for i = 1:size(boxes, 1)
        b = clipRect(boxes(i, :), [h, w]);
        if b(3) < 30 || b(4) < 30
            continue;
        end
        areaRatio = (b(3) * b(4)) / (h * w);
        centerX = b(1) + b(3) / 2;
        centerY = b(2) + b(4) / 2;
        centerPenalty = abs(centerX - w / 2) / w + 0.7 * abs(centerY - h * 0.42) / h;
        aspectPenalty = abs((b(3) / max(b(4), 1)) - 0.82);
        score = areaRatio - 0.18 * centerPenalty - 0.03 * aspectPenalty;
        if score > bestScore
            bestScore = score;
            bestBox = b;
        end
    end
end

function cropBox = faceCropBoxFromFaceBox(faceBox, imgSize, targetFaceOccupancy)
    h = double(imgSize(1));
    w = double(imgSize(2));
    faceBox = clipRect(faceBox, [h, w]);
    scale = 1 / sqrt(max(0.01, min(0.99, targetFaceOccupancy)));
    cropW = max(faceBox(3), faceBox(3) * scale);
    cropH = max(faceBox(4), faceBox(4) * scale);
    cx = faceBox(1) + faceBox(3) / 2;
    cy = faceBox(2) + faceBox(4) / 2;
    cropBox = clipRect([cx - cropW / 2, cy - cropH / 2, cropW, cropH], [h, w]);
end

function rect = clipRect(rect, imgSize)
    h = double(imgSize(1));
    w = double(imgSize(2));
    x1 = max(1, min(w, double(rect(1))));
    y1 = max(1, min(h, double(rect(2))));
    x2 = max(1, min(w, double(rect(1)) + double(rect(3)) - 1));
    y2 = max(1, min(h, double(rect(2)) + double(rect(4)) - 1));
    rect = [x1, y1, max(1, x2 - x1 + 1), max(1, y2 - y1 + 1)];
end

function occupancy = faceOccupancyInCrop(faceBox, cropBox)
    x1 = max(faceBox(1), cropBox(1));
    y1 = max(faceBox(2), cropBox(2));
    x2 = min(faceBox(1) + faceBox(3), cropBox(1) + cropBox(3));
    y2 = min(faceBox(2) + faceBox(4), cropBox(2) + cropBox(4));
    overlapW = max(0, x2 - x1);
    overlapH = max(0, y2 - y1);
    occupancy = min(1, (overlapW * overlapH) / max(1, cropBox(3) * cropBox(4)));
end

function faceGray = localContrastNormalize(faceGray)
    faceGray = im2uint8(faceGray);
    try
        faceGray = adapthisteq(faceGray, 'ClipLimit', 0.015, 'Distribution', 'rayleigh');
    catch
        faceGray = histeq(faceGray);
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

function [ok, reason, occupancy] = verifyPreprocessedFaceOnly(imagePath, cropMeta, minimumFaceOccupancy)
    ok = false;
    reason = "";
    occupancy = 0;
    if ~isfile(imagePath)
        reason = "预处理图片文件不存在";
        return;
    end

    imgInfo = imfinfo(imagePath);
    if imgInfo.Width ~= 160 || imgInfo.Height ~= 160
        reason = sprintf('预处理图片尺寸错误：%dx%d', imgInfo.Width, imgInfo.Height);
        return;
    end

    if ~isfield(cropMeta, 'valid') || ~cropMeta.valid
        reason = "人脸裁剪元数据无效，无法确认人脸占比";
        return;
    end

    occupancy = cropMeta.occupancy;
    ok = occupancy >= minimumFaceOccupancy;
    if ~ok
        reason = sprintf('几何人脸占比不足：%.2f%%', 100 * occupancy);
    end
end

function runResult = buildRunResult(totalImages, mtcnnCount, processedCount, validNames, validLabels, validOriginalPaths, validPreprocessedPaths, validFaceOccupancies, validPreprocessMethods, validQualityScores, validRepairFlags, failedNames, failedReasons, mtcnnSuccessRate, preprocessSuccessRate, savePath, scriptRuntimeSeconds)
    failedSamples = table();
    if ~isempty(failedNames)
        failedSamples = table(failedNames, failedReasons, 'VariableNames', {'FileName', 'Reason'});
    end
    savedSamples = table(validNames, validLabels, validOriginalPaths, validPreprocessedPaths, validFaceOccupancies, validPreprocessMethods, validQualityScores, validRepairFlags, ...
        'VariableNames', {'FileName', 'TrueLabel', 'OriginalPath', 'PreprocessedPath', 'FaceOccupancy', 'PreprocessMethod', 'QualityScore', 'WasReprocessed'});

    runResult = struct( ...
        'timestamp', char(datetime("now", "Format", "yyyy-MM-dd'T'HH:mm:ss")), ...
        'totalImages', totalImages, ...
        'mtcnnCount', mtcnnCount, ...
        'processedCount', processedCount, ...
        'savedImageCount', numel(validNames), ...
        'mtcnnSuccessRate', mtcnnSuccessRate, ...
        'preprocessSuccessRate', preprocessSuccessRate, ...
        'scriptRuntimeSeconds', scriptRuntimeSeconds, ...
        'savedImageFolder', savePath, ...
        'savedSamples', savedSamples, ...
        'failedSamples', failedSamples);
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
