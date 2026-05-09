function gui
% GUI for running PCA.m and inspecting PCA face recognition results.

scriptDir = fileparts(mfilename('fullpath'));
pcaScriptPath = fullfile(scriptDir, 'PCA.m');
resultMatPath = fullfile(scriptDir, 'PCA_last_result.mat');

app = struct();
app.Result = [];
app.SelectedImagePath = "";

fig = uifigure('Name', 'PCA人脸识别GUI', 'Position', [100, 80, 1180, 720]);
mainGrid = uigridlayout(fig, [3, 3]);
mainGrid.RowHeight = {62, '1x', 210};
mainGrid.ColumnWidth = {310, '1x', 360};
mainGrid.Padding = [16, 14, 16, 14];
mainGrid.RowSpacing = 12;
mainGrid.ColumnSpacing = 14;

runButton = uibutton(mainGrid, 'push', ...
    'Text', '一键运行 PCA.m', ...
    'FontSize', 18, ...
    'ButtonPushedFcn', @runPcaButtonPushed);
runButton.Layout.Row = 1;
runButton.Layout.Column = 1;

statusLabel = uilabel(mainGrid, ...
    'Text', '状态：等待运行', ...
    'FontSize', 15, ...
    'FontWeight', 'bold');
statusLabel.Layout.Row = 1;
statusLabel.Layout.Column = 2;

openResultButton = uibutton(mainGrid, 'push', ...
    'Text', '刷新/读取结果', ...
    'FontSize', 15, ...
    'ButtonPushedFcn', @loadResultButtonPushed);
openResultButton.Layout.Row = 1;
openResultButton.Layout.Column = 3;

leftPanel = uipanel(mainGrid, 'Title', 'PCA平均脸');
leftPanel.Layout.Row = 2;
leftPanel.Layout.Column = 1;
leftGrid = uigridlayout(leftPanel, [2, 1]);
leftGrid.RowHeight = {'1x', 96};
meanAxes = uiaxes(leftGrid);
meanAxes.Layout.Row = 1;
meanAxes.XTick = [];
meanAxes.YTick = [];
meanAxes.Box = 'on';
title(meanAxes, '平均脸');

metricLabel = uilabel(leftGrid, ...
    'Text', "准确率：--" + newline + "图片数：--" + newline + "类别数：--", ...
    'FontSize', 16, ...
    'FontWeight', 'bold');
metricLabel.Layout.Row = 2;

tablePanel = uipanel(mainGrid, 'Title', '识别失败样本');
tablePanel.Layout.Row = 2;
tablePanel.Layout.Column = 2;
tableGrid = uigridlayout(tablePanel, [1, 1]);
failTable = uitable(tableGrid, ...
    'Data', table(), ...
    'ColumnName', {'FileName', 'TrueLabel', 'PredictedLabel', 'ImagePath'}, ...
    'CellSelectionCallback', @failTableSelectionChanged);
failTable.Layout.Row = 1;

previewPanel = uipanel(mainGrid, 'Title', '失败照片预览');
previewPanel.Layout.Row = 2;
previewPanel.Layout.Column = 3;
previewGrid = uigridlayout(previewPanel, [2, 1]);
previewGrid.RowHeight = {'1x', 60};
previewAxes = uiaxes(previewGrid);
previewAxes.Layout.Row = 1;
previewAxes.XTick = [];
previewAxes.YTick = [];
previewAxes.Box = 'on';
title(previewAxes, '点击失败样本查看照片');
previewLabel = uilabel(previewGrid, ...
    'Text', '未选择照片', ...
    'FontSize', 13);
previewLabel.Layout.Row = 2;

summaryPanel = uipanel(mainGrid, 'Title', '运行摘要');
summaryPanel.Layout.Row = 3;
summaryPanel.Layout.Column = [1, 3];
summaryGrid = uigridlayout(summaryPanel, [1, 1]);
summaryText = uitextarea(summaryGrid, ...
    'Value', {'点击“一键运行 PCA.m”开始；运行完成后会自动加载准确率、平均脸和识别失败样本。'}, ...
    'Editable', 'off', ...
    'FontSize', 13);
summaryText.Layout.Row = 1;

if isfile(resultMatPath)
    loadAndRenderResult(false);
end

    function runPcaButtonPushed(~, ~)
        if ~isfile(pcaScriptPath)
            uialert(fig, sprintf('未找到 PCA.m：%s', pcaScriptPath), '文件不存在');
            return;
        end

        runButton.Enable = 'off';
        openResultButton.Enable = 'off';
        statusLabel.Text = '状态：PCA运行中，请稍等...';
        summaryText.Value = {'正在运行 PCA.m，这一步会搜索最佳 PCA 参数。'};
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

        runButton.Enable = 'on';
        openResultButton.Enable = 'on';
    end

    function loadResultButtonPushed(~, ~)
        loadAndRenderResult(true);
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
        metricLabel.Text = sprintf(['准确率：%.2f%% (%d/%d)\n', ...
            '图片数：%d\n类别数：%d\nPCA维度：%d'], ...
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
            failTable.Data = table("无", "无", "无", "无", ...
                'VariableNames', {'FileName', 'TrueLabel', 'PredictedLabel', 'ImagePath'});
        else
            failTable.Data = misclassified;
        end
    end

    function renderSummary()
        result = app.Result;
        misclassifiedCount = height(result.BestResult.MisclassifiedSamples);
        summaryText.Value = {
            sprintf('结果时间：%s', result.Timestamp)
            sprintf('数据集：%s', result.DatasetPath)
            sprintf('最佳准确率：%.2f%%，正确 %d / 测试 %d', result.BestAccuracy, result.BestResult.CorrectCount, result.BestResult.TestCount)
            sprintf('最佳参数：trainRatio=%.2f, seed=%d, components=%d, metric=%s', result.BestResult.TrainRatio, result.BestResult.Seed, result.BestResult.Components, result.BestResult.Metric)
            sprintf('识别失败样本数：%d', misclassifiedCount)
            '点击表格中的失败样本行，可以在右侧预览对应照片。'
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
        if imagePath == "无" || ~isfile(imagePath)
            previewLabel.Text = '照片文件不存在';
            cla(previewAxes);
            return;
        end

        try
            img = imread(imagePath);
            imshow(img, 'Parent', previewAxes);
            title(previewAxes, string(tableData.FileName(row)), 'Interpreter', 'none');
            previewLabel.Text = sprintf('真实：%s    预测：%s', string(tableData.TrueLabel(row)), string(tableData.PredictedLabel(row)));
            app.SelectedImagePath = imagePath;
        catch ME
            previewLabel.Text = "无法读取照片：" + string(ME.message);
        end
    end

    function escaped = escapeSingleQuotes(text)
        escaped = strrep(char(text), '''', '''''');
    end
end
