function gui
% GUI for PCA face recognition results, failed-sample comparison, and
% single-image recognition.

scriptDir = fileparts(mfilename('fullpath'));
pcaScriptPath = fullfile(scriptDir, 'PCA.m');
resultMatPath = fullfile(scriptDir, 'PCA_last_result.mat');

app = struct();
app.Result = [];

fig = uifigure('Name', 'PCA Face Recognition GUI', 'Position', [80, 60, 1320, 780]);
mainGrid = uigridlayout(fig, [3, 4]);
mainGrid.RowHeight = {64, '1x', 250};
mainGrid.ColumnWidth = {280, '1x', 340, 340};
mainGrid.Padding = [14, 12, 14, 12];
mainGrid.RowSpacing = 12;
mainGrid.ColumnSpacing = 12;

runButton = uibutton(mainGrid, 'push', ...
    'Text', '一键运行 PCA.m', ...
    'FontSize', 17, ...
    'ButtonPushedFcn', @runPcaButtonPushed);
runButton.Layout.Row = 1;
runButton.Layout.Column = 1;

statusLabel = uilabel(mainGrid, ...
    'Text', '状态：等待运行', ...
    'FontSize', 15, ...
    'FontWeight', 'bold');
statusLabel.Layout.Row = 1;
statusLabel.Layout.Column = 2;

loadButton = uibutton(mainGrid, 'push', ...
    'Text', '刷新结果', ...
    'FontSize', 15, ...
    'ButtonPushedFcn', @loadResultButtonPushed);
loadButton.Layout.Row = 1;
loadButton.Layout.Column = 3;

recognizeButton = uibutton(mainGrid, 'push', ...
    'Text', '选择照片识别', ...
    'FontSize', 15, ...
    'ButtonPushedFcn', @recognizePhotoButtonPushed);
recognizeButton.Layout.Row = 1;
recognizeButton.Layout.Column = 4;

meanPanel = uipanel(mainGrid, 'Title', 'PCA平均脸与准确率');
meanPanel.Layout.Row = 2;
meanPanel.Layout.Column = 1;
meanGrid = uigridlayout(meanPanel, [2, 1]);
meanGrid.RowHeight = {'1x', 120};
meanAxes = uiaxes(meanGrid);
meanAxes.XTick = [];
meanAxes.YTick = [];
meanAxes.Box = 'on';
title(meanAxes, '平均脸');
metricLabel = uilabel(meanGrid, ...
    'Text', "准确率：--" + newline + "正确数：--" + newline + "类别数：--", ...
    'FontSize', 15, ...
    'FontWeight', 'bold');

tablePanel = uipanel(mainGrid, 'Title', '未能正确识别的样本');
tablePanel.Layout.Row = 2;
tablePanel.Layout.Column = 2;
tableGrid = uigridlayout(tablePanel, [1, 1]);
failTable = uitable(tableGrid, ...
    'Data', table(), ...
    'ColumnName', {'FileName', 'TrueLabel', 'PredictedLabel', 'ImagePath', 'PredictedFileName', 'PredictedImagePath'}, ...
    'CellSelectionCallback', @failTableSelectionChanged);

comparePanel = uipanel(mainGrid, 'Title', '失败样本对比');
comparePanel.Layout.Row = 2;
comparePanel.Layout.Column = 3;
compareGrid = uigridlayout(comparePanel, [2, 2]);
compareGrid.RowHeight = {'1x', 46};
compareGrid.ColumnWidth = {'1x', '1x'};
failAxes = uiaxes(compareGrid);
failAxes.XTick = [];
failAxes.YTick = [];
failAxes.Box = 'on';
title(failAxes, '未正确识别照片');
predAxes = uiaxes(compareGrid);
predAxes.XTick = [];
predAxes.YTick = [];
predAxes.Box = 'on';
title(predAxes, '被识别成的照片');
failInfoLabel = uilabel(compareGrid, 'Text', '真实：--', 'FontSize', 12);
failInfoLabel.Layout.Row = 2;
failInfoLabel.Layout.Column = 1;
predInfoLabel = uilabel(compareGrid, 'Text', '预测：--', 'FontSize', 12);
predInfoLabel.Layout.Row = 2;
predInfoLabel.Layout.Column = 2;

recognizePanel = uipanel(mainGrid, 'Title', '提供照片进行识别');
recognizePanel.Layout.Row = 2;
recognizePanel.Layout.Column = 4;
recognizeGrid = uigridlayout(recognizePanel, [2, 2]);
recognizeGrid.RowHeight = {'1x', 76};
recognizeGrid.ColumnWidth = {'1x', '1x'};
queryAxes = uiaxes(recognizeGrid);
queryAxes.XTick = [];
queryAxes.YTick = [];
queryAxes.Box = 'on';
title(queryAxes, '输入照片');
matchAxes = uiaxes(recognizeGrid);
matchAxes.XTick = [];
matchAxes.YTick = [];
matchAxes.Box = 'on';
title(matchAxes, '最相似训练照片');
recognizeResultLabel = uilabel(recognizeGrid, ...
    'Text', "识别人物：--" + newline + "相似度：--", ...
    'FontSize', 13, ...
    'FontWeight', 'bold');
recognizeResultLabel.Layout.Row = 2;
recognizeResultLabel.Layout.Column = [1, 2];

summaryPanel = uipanel(mainGrid, 'Title', '运行摘要');
summaryPanel.Layout.Row = 3;
summaryPanel.Layout.Column = [1, 4];
summaryGrid = uigridlayout(summaryPanel, [1, 1]);
summaryText = uitextarea(summaryGrid, ...
    'Value', {'点击“一键运行 PCA.m”生成结果；点击失败样本行可对比原照片和被识别成的照片；点击“选择照片识别”可识别外部照片。'}, ...
    'Editable', 'off', ...
    'FontSize', 13);

if isfile(resultMatPath)
    loadAndRenderResult(false);
end

    function runPcaButtonPushed(~, ~)
        if ~isfile(pcaScriptPath)
            uialert(fig, sprintf('未找到 PCA.m：%s', pcaScriptPath), '文件不存在');
            return;
        end

        setControlsEnabled(false);
        statusLabel.Text = '状态：正在运行 PCA.m...';
        summaryText.Value = {'PCA 正在运行，请稍等。'};
        drawnow;

        try
            matlabExe = fullfile(matlabroot, 'bin', 'matlab.exe');
            if ~isfile(matlabExe)
                matlabExe = fullfile(matlabroot, 'bin', 'matlab');
            end
            command = sprintf('"%s" -batch "cd(''%s''); run(''PCA.m'');"', matlabExe, escapeSingleQuotes(scriptDir));
            [status, cmdout] = system(command);
            if status ~= 0
                error('PCA.m 运行失败：%s', cmdout);
            end
            loadAndRenderResult(true);
            statusLabel.Text = '状态：PCA运行完成';
        catch ME
            statusLabel.Text = '状态：PCA运行失败';
            summaryText.Value = splitlines(getReport(ME, 'extended', 'hyperlinks', 'off'));
            uialert(fig, ME.message, 'PCA运行失败');
        end

        setControlsEnabled(true);
    end

    function loadResultButtonPushed(~, ~)
        loadAndRenderResult(true);
    end

    function recognizePhotoButtonPushed(~, ~)
        if isempty(app.Result) || ~isfield(app.Result, 'RecognitionModel')
            uialert(fig, '请先运行 PCA.m 或刷新读取包含 RecognitionModel 的结果文件。', '缺少模型');
            return;
        end

        [fileName, folderName] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff;*.webp', 'Image files'}, '选择要识别的照片');
        if isequal(fileName, 0)
            return;
        end

        queryPath = fullfile(folderName, fileName);
        try
            [predictedLabel, similarity, matchedPath, matchedName] = recognizeSingleImage(queryPath, app.Result.RecognitionModel);
            imshow(imread(queryPath), 'Parent', queryAxes);
            title(queryAxes, fileName, 'Interpreter', 'none');
            if isfile(matchedPath)
                imshow(imread(matchedPath), 'Parent', matchAxes);
                title(matchAxes, matchedName, 'Interpreter', 'none');
            else
                cla(matchAxes);
                title(matchAxes, '匹配照片不存在');
            end
            recognizeResultLabel.Text = sprintf('识别人物：%s\n相似度：%.2f%%\n当前PCA准确率：%.2f%%', ...
                predictedLabel, 100 * similarity, app.Result.BestAccuracy);
        catch ME
            uialert(fig, ME.message, '识别失败');
        end
    end

    function loadAndRenderResult(showAlert)
        if ~isfile(resultMatPath)
            if showAlert
                uialert(fig, sprintf('未找到结果文件：%s', resultMatPath), '结果不存在');
            end
            return;
        end

        data = load(resultMatPath, 'PCAResult');
        if ~isfield(data, 'PCAResult')
            uialert(fig, 'PCA_last_result.mat 中没有 PCAResult 变量。', '结果格式错误');
            return;
        end
        app.Result = data.PCAResult;

        renderMetrics();
        renderMeanFace();
        renderMisclassifiedTable();
        renderSummary();
    end

    function renderMetrics()
        result = app.Result;
        metricLabel.Text = sprintf(['准确率：%.2f%%\n', ...
            '正确数：%d/%d\n图片数：%d\n类别数：%d\nPCA维度：%d'], ...
            result.BestAccuracy, ...
            result.BestResult.CorrectCount, ...
            result.BestResult.TestCount, ...
            result.ImageCount, ...
            result.ClassCount, ...
            result.BestResult.Components);
    end

    function renderMeanFace()
        cla(meanAxes);
        if isfield(app.Result, 'MeanFaceImage') && ~isempty(app.Result.MeanFaceImage)
            imshow(app.Result.MeanFaceImage, 'Parent', meanAxes);
            title(meanAxes, 'PCA平均脸');
        else
            text(meanAxes, 0.5, 0.5, '结果中没有平均脸', 'HorizontalAlignment', 'center');
        end
    end

    function renderMisclassifiedTable()
        misclassified = app.Result.BestResult.MisclassifiedSamples;
        if isempty(misclassified)
            failTable.Data = table("无", "无", "无", "无", "无", "无", ...
                'VariableNames', {'FileName', 'TrueLabel', 'PredictedLabel', 'ImagePath', 'PredictedFileName', 'PredictedImagePath'});
            return;
        end
        if ~ismember('PredictedFileName', misclassified.Properties.VariableNames)
            misclassified.PredictedFileName = strings(height(misclassified), 1);
        end
        if ~ismember('PredictedImagePath', misclassified.Properties.VariableNames)
            misclassified.PredictedImagePath = strings(height(misclassified), 1);
        end
        failTable.Data = misclassified;
    end

    function renderSummary()
        result = app.Result;
        misclassifiedCount = height(result.BestResult.MisclassifiedSamples);
        hasModel = isfield(result, 'RecognitionModel');
        summaryText.Value = {
            sprintf('结果时间：%s', result.Timestamp)
            sprintf('数据集：%s', result.DatasetPath)
            sprintf('当前PCA准确率：%.2f%%，正确 %d / 测试 %d', result.BestAccuracy, result.BestResult.CorrectCount, result.BestResult.TestCount)
            sprintf('最佳参数：trainRatio=%.2f, seed=%d, components=%d, metric=%s', result.BestResult.TrainRatio, result.BestResult.Seed, result.BestResult.Components, result.BestResult.Metric)
            sprintf('识别失败样本数：%d', misclassifiedCount)
            sprintf('是否可进行单张照片识别：%s', string(hasModel))
            '点击失败样本表格行，会显示未正确识别照片和被识别成的训练照片。'
            };
    end

    function failTableSelectionChanged(~, event)
        if isempty(event.Indices) || isempty(failTable.Data) || ~istable(failTable.Data)
            return;
        end
        row = event.Indices(1);
        tableData = failTable.Data;
        if row < 1 || row > height(tableData) || ~ismember('ImagePath', tableData.Properties.VariableNames)
            return;
        end

        imagePath = string(tableData.ImagePath(row));
        predictedPath = "";
        if ismember('PredictedImagePath', tableData.Properties.VariableNames)
            predictedPath = string(tableData.PredictedImagePath(row));
        end

        showImageOnAxes(imagePath, failAxes, string(tableData.FileName(row)));
        if predictedPath ~= "" && isfile(predictedPath)
            showImageOnAxes(predictedPath, predAxes, string(tableData.PredictedFileName(row)));
        else
            cla(predAxes);
            title(predAxes, '旧结果未保存预测照片路径');
        end

        failInfoLabel.Text = sprintf('真实：%s', string(tableData.TrueLabel(row)));
        predInfoLabel.Text = sprintf('预测：%s', string(tableData.PredictedLabel(row)));
    end

    function showImageOnAxes(imagePath, ax, titleText)
        cla(ax);
        if imagePath == "无" || ~isfile(imagePath)
            title(ax, '照片不存在');
            return;
        end
        imshow(imread(imagePath), 'Parent', ax);
        title(ax, titleText, 'Interpreter', 'none');
    end

    function setControlsEnabled(isEnabled)
        value = 'off';
        if isEnabled
            value = 'on';
        end
        runButton.Enable = value;
        loadButton.Enable = value;
        recognizeButton.Enable = value;
    end

    function escaped = escapeSingleQuotes(text)
        escaped = strrep(char(text), '''', '''''');
    end
end

function [predictedLabel, similarity, matchedPath, matchedName] = recognizeSingleImage(imagePath, model)
    img = imread(imagePath);
    img = guiOwnGray(img);
    img = guiResizeBilinear(img, model.ImageSize);
    img = guiLocalNormalize(img);
    feature = reshape(img, 1, []);
    feature = (feature - model.FeatureMu) ./ model.FeatureSigma;
    feature(~isfinite(feature)) = 0;
    projected = (feature - model.PcaMu) * model.Coeff;

    bestDistance = inf;
    bestIndex = 1;
    for i = 1:size(model.TrainFeatures, 1)
        if model.Metric == "cosine"
            distance = guiCosineDistance(projected, model.TrainFeatures(i, :));
        else
            distance = guiEuclideanDistance(projected, model.TrainFeatures(i, :));
        end
        if distance < bestDistance
            bestDistance = distance;
            bestIndex = i;
        end
    end

    predictedLabel = string(model.TrainLabels(bestIndex));
    matchedPath = string(model.TrainImagePaths(bestIndex));
    matchedName = string(model.TrainSampleNames(bestIndex));
    if model.Metric == "cosine"
        similarity = max(0, min(1, 1 - bestDistance));
    else
        similarity = 1 / (1 + bestDistance);
    end
end

function gray = guiOwnGray(img)
    img = double(img);
    if max(img(:)) > 1
        img = img / 255;
    end
    if ndims(img) >= 3 && size(img, 3) >= 3
        gray = 0.2989 * img(:, :, 1) + 0.5870 * img(:, :, 2) + 0.1140 * img(:, :, 3);
    else
        gray = img(:, :, 1);
    end
    gray(gray < 0) = 0;
    gray(gray > 1) = 1;
end

function out = guiResizeBilinear(img, outSize)
    inH = size(img, 1);
    inW = size(img, 2);
    outH = outSize(1);
    outW = outSize(2);
    out = zeros(outH, outW);
    rowScale = (inH - 1) / max(1, outH - 1);
    colScale = (inW - 1) / max(1, outW - 1);
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

function img = guiLocalNormalize(img)
    mu = sum(img(:)) / numel(img);
    diff = img(:) - mu;
    sigma = sqrt(sum(diff .* diff) / max(1, numel(diff) - 1));
    if sigma < 1e-6 || ~isfinite(sigma)
        sigma = 1;
    end
    img = (img - mu) ./ sigma;
end

function distance = guiCosineDistance(a, b)
    denom = sqrt(a(:)' * a(:)) * sqrt(b(:)' * b(:));
    if denom < 1e-12
        distance = 1;
    else
        distance = 1 - (a * b') / denom;
    end
end

function distance = guiEuclideanDistance(a, b)
    diff = a - b;
    distance = sqrt(diff * diff');
end
