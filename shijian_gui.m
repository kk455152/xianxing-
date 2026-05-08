function shijian_gui
scriptDir = fileparts(mfilename('fullpath'));
addpath(scriptDir);
scriptPath = fullfile(scriptDir, 'shijian1.m');
resultPath = fullfile(scriptDir, 'shijian_last_result.mat');
currentResult = [];

if ~isfile(scriptPath)
    error('Cannot find script file: %s', scriptPath);
end

fig = uifigure( ...
    'Name', '人脸识别运行界面', ...
    'Position', [120 80 1260 780], ...
    'Color', [0.98 0.98 0.99]);

mainGrid = uigridlayout(fig, [4 1]);
mainGrid.RowHeight = {70, 120, '1x', '1x'};
mainGrid.ColumnWidth = {'1x'};
mainGrid.Padding = [16 16 16 16];
mainGrid.RowSpacing = 12;

topGrid = uigridlayout(mainGrid, [1 7]);
topGrid.Layout.Row = 1;
topGrid.ColumnWidth = {170, 210, 170, 165, 140, '1x', 110};
topGrid.ColumnSpacing = 10;
topGrid.Padding = [0 0 0 0];

runButton = uibutton(topGrid, 'push', ...
    'Text', '运行 shijian1.m', ...
    'FontSize', 15, ...
    'ButtonPushedFcn', @onRunButtonPushed);
runButton.Layout.Column = 1;

viewFacesButton = uibutton(topGrid, 'push', ...
    'Text', '查看本次保存的人脸图片', ...
    'FontSize', 15, ...
    'ButtonPushedFcn', @onViewSavedFacesButtonPushed);
viewFacesButton.Layout.Column = 2;

cameraButton = uibutton(topGrid, 'push', ...
    'Text', '摄像头实时识别', ...
    'FontSize', 15, ...
    'ButtonPushedFcn', @onCameraButtonPushed);
cameraButton.Layout.Column = 3;

openFolderButton = uibutton(topGrid, 'push', ...
    'Text', '打开结果文件夹', ...
    'FontSize', 15, ...
    'ButtonPushedFcn', @onOpenFolderButtonPushed);
openFolderButton.Layout.Column = 4;

openMatButton = uibutton(topGrid, 'push', ...
    'Text', '打开结果MAT', ...
    'FontSize', 15, ...
    'ButtonPushedFcn', @onOpenMatButtonPushed);
openMatButton.Layout.Column = 5;

statusLabel = uilabel(topGrid, ...
    'Text', '状态：等待运行', ...
    'FontSize', 14, ...
    'HorizontalAlignment', 'left');
statusLabel.Layout.Column = 6;

refreshButton = uibutton(topGrid, 'push', ...
    'Text', '刷新', ...
    'FontSize', 15, ...
    'ButtonPushedFcn', @(~,~) loadAndRenderResult());
refreshButton.Layout.Column = 7;

summaryGrid = uigridlayout(mainGrid, [2 4]);
summaryGrid.Layout.Row = 2;
summaryGrid.RowHeight = {24, '1x'};
summaryGrid.ColumnWidth = {'1x', '1x', '1x', '1x'};
summaryGrid.Padding = [0 0 0 0];
summaryGrid.ColumnSpacing = 12;
summaryGrid.RowSpacing = 6;

summaryTitles = ["最佳识别准确率", "MTCNN检测成功率", "68点预处理成功率", "本次保存图片数"];
summaryValues = gobjects(1, 4);
for i = 1:4
    titleLabel = uilabel(summaryGrid, 'Text', summaryTitles(i), 'FontSize', 13, 'FontWeight', 'bold');
    titleLabel.Layout.Row = 1;
    titleLabel.Layout.Column = i;

    valueLabel = uilabel(summaryGrid, 'Text', '--', 'FontSize', 22, 'FontWeight', 'bold');
    valueLabel.Layout.Row = 2;
    valueLabel.Layout.Column = i;
    summaryValues(i) = valueLabel;
end

misPanel = uipanel(mainGrid, 'Title', '未正确识别的样本');
misPanel.Layout.Row = 3;
misPanel.FontSize = 15;
misGrid = uigridlayout(misPanel, [1 1]);
misGrid.Padding = [6 6 6 6];
misTable = uitable(misGrid, ...
    'Data', emptySampleTable(), ...
    'ColumnName', {'FileName', 'TrueLabel', 'PredictedLabel', 'OriginalPath', 'PreprocessedPath'}, ...
    'RowStriping', 'on', ...
    'CellSelectionCallback', @onMisclassifiedTableSelected);

failPanel = uipanel(mainGrid, 'Title', '预处理失败样本');
failPanel.Layout.Row = 4;
failPanel.FontSize = 15;
failGrid = uigridlayout(failPanel, [1 1]);
failGrid.Padding = [6 6 6 6];
failTable = uitable(failGrid, ...
    'Data', emptyFailureTable(), ...
    'ColumnName', {'FileName', 'Reason'}, ...
    'RowStriping', 'on');

loadAndRenderResult();

    function onRunButtonPushed(~, ~)
        setButtonsEnabled('off');
        statusLabel.Text = '状态：正在运行，请稍等...';
        drawnow;

        try
            if isfile(resultPath)
                delete(resultPath);
            end

            evalin('base', sprintf('run(''%s'');', strrep(scriptPath, '''', '''''')));
            loadAndRenderResult();
            statusLabel.Text = '状态：运行完成';
        catch ME
            statusLabel.Text = '状态：运行失败';
            uialert(fig, getReport(ME, 'extended', 'hyperlinks', 'off'), '运行失败', 'Icon', 'error');
        end

        setButtonsEnabled('on');
    end

    function onViewSavedFacesButtonPushed(~, ~)
        result = getCurrentResult();
        if isempty(result) || ~isfield(result, 'savedSamples') || isempty(result.savedSamples)
            uialert(fig, '当前还没有可查看的人脸图片，请先运行一次脚本。', '提示');
            return;
        end

        samples = normalizeSavedSamples(result.savedSamples);
        validMask = arrayfun(@(p) isfile(char(p)), samples.PreprocessedPath);
        samples = samples(validMask, :);
        if isempty(samples)
            uialert(fig, '没有找到本次保存的预处理图片文件。', '提示');
            return;
        end

        showSavedFacesGallery(samples);
    end

    function onCameraButtonPushed(~, ~)
        result = getCurrentResult();
        if isempty(result) || ~isfield(result, 'savedSamples') || isempty(result.savedSamples)
            uialert(fig, '请先运行一次 shijian1.m，生成训练样本后再使用摄像头实时识别。', '提示');
            return;
        end

        try
            statusLabel.Text = '状态：正在构建摄像头实时识别模型...';
            drawnow;
            showCameraRecognitionWindow(result.savedSamples);
            statusLabel.Text = '状态：摄像头实时识别窗口已打开';
        catch ME
            statusLabel.Text = '状态：摄像头实时识别启动失败';
            uialert(fig, getReport(ME, 'extended', 'hyperlinks', 'off'), '摄像头实时识别启动失败', 'Icon', 'error');
        end
    end

    function onOpenFolderButtonPushed(~, ~)
        result = getCurrentResult();
        if ~isempty(result) && isfield(result, 'savedImageFolder') && isfolder(result.savedImageFolder)
            winopen(result.savedImageFolder);
            return;
        end
        uialert(fig, '当前还没有可打开的结果文件夹。', '提示');
    end

    function onOpenMatButtonPushed(~, ~)
        if isfile(resultPath)
            winopen(resultPath);
        else
            uialert(fig, '当前还没有结果 MAT 文件。', '提示');
        end
    end

    function onMisclassifiedTableSelected(~, event)
        if isempty(event.Indices)
            return;
        end

        rowIndex = event.Indices(1);
        data = misTable.Data;
        if isempty(data) || rowIndex > height(data)
            return;
        end

        row = data(rowIndex, :);
        if ~all(ismember({'OriginalPath', 'PreprocessedPath'}, row.Properties.VariableNames))
            uialert(fig, '当前结果缺少原图或预处理图路径，请重新运行一次脚本。', '提示');
            return;
        end

        showSamplePreview(row);
    end

    function loadAndRenderResult()
        if ~isfile(resultPath)
            currentResult = [];
            summaryValues(1).Text = '--';
            summaryValues(2).Text = '--';
            summaryValues(3).Text = '--';
            summaryValues(4).Text = '--';
            misTable.Data = emptySampleTable();
            failTable.Data = emptyFailureTable();
            statusLabel.Text = '状态：等待运行';
            return;
        end

        S = load(resultPath, 'shijianRunResult');
        if ~isfield(S, 'shijianRunResult')
            uialert(fig, '结果 MAT 文件中缺少 shijianRunResult。', '结果读取失败', 'Icon', 'error');
            return;
        end

        currentResult = S.shijianRunResult;
        summaryValues(1).Text = formatPercent(currentResult.bestAccuracy, currentResult.bestCorrectCount, currentResult.bestTestCount);
        summaryValues(2).Text = formatRate(currentResult.mtcnnSuccessRate, currentResult.mtcnnCount, currentResult.totalImages);
        summaryValues(3).Text = formatRate(currentResult.preprocessSuccessRate, currentResult.processedCount, currentResult.totalImages);
        summaryValues(4).Text = sprintf('%d', currentResult.savedImageCount);

        if isfield(currentResult, 'misclassifiedSamples') && ~isempty(currentResult.misclassifiedSamples)
            misTable.Data = normalizeMisclassifiedTable(currentResult.misclassifiedSamples);
        else
            misTable.Data = emptySampleTable();
        end

        if isfield(currentResult, 'failedSamples') && ~isempty(currentResult.failedSamples)
            failTable.Data = normalizeFailureTable(currentResult.failedSamples);
        else
            failTable.Data = emptyFailureTable();
        end

        if isfield(currentResult, 'timestamp')
            statusLabel.Text = sprintf('状态：已加载 %s 的结果', char(currentResult.timestamp));
        else
            statusLabel.Text = '状态：结果已加载';
        end
    end

    function result = getCurrentResult()
        if isempty(currentResult)
            loadAndRenderResult();
        end
        result = currentResult;
    end

    function setButtonsEnabled(state)
        runButton.Enable = state;
        refreshButton.Enable = state;
        viewFacesButton.Enable = state;
        cameraButton.Enable = state;
        openFolderButton.Enable = state;
        openMatButton.Enable = state;
    end
end

function showCameraRecognitionWindow(samples)
if exist('mtcnn.Detector', 'class') ~= 8
    error('未找到 MTCNN Face Detection，请先确认 mtcnn.Detector 已加入 MATLAB 路径。');
end

hasMatlabWebcam = exist('webcamlist', 'file') == 2 && exist('webcam', 'file') == 2;
pythonExe = 'C:\Users\32545\AppData\Local\Python\pythoncore-3.14-64\python.exe';
opencvHelperPath = fullfile(fileparts(mfilename('fullpath')), 'opencv_camera_stream.py');
if ~hasMatlabWebcam
    if ~isfile(pythonExe)
        error('当前 MATLAB 缺少 webcam 支持，并且未找到 Python：%s', pythonExe);
    end
    if ~isfile(opencvHelperPath)
        error('当前 MATLAB 缺少 webcam 支持，并且未找到 OpenCV 摄像头辅助脚本：%s', opencvHelperPath);
    end
end

samples = normalizeSavedSamples(samples);
validMask = arrayfun(@(p) isfile(char(p)), samples.PreprocessedPath);
samples = samples(validMask, :);
if isempty(samples)
    error('没有找到可用于训练实时识别模型的预处理人脸图片。');
end

model = trainLiveRecognitionModel(samples);
if hasMatlabWebcam
    cameraNames = webcamlist;
    if isstring(cameraNames)
        cameraNames = cellstr(cameraNames);
    end
    if isempty(cameraNames)
        error('没有检测到可用摄像头。');
    end
    defaultCameraIdx = chooseFrontCamera(cameraNames);
    cameraModeText = 'MATLAB webcam';
else
    cameraNames = {'OpenCV 摄像头 0 DSHOW', 'OpenCV 摄像头 0 MSMF', 'OpenCV 摄像头 0 ANY', ...
        'OpenCV 摄像头 1 DSHOW', 'OpenCV 摄像头 1 MSMF', 'OpenCV 摄像头 1 ANY'};
    defaultCameraIdx = 1;
    cameraModeText = 'Python OpenCV';
end

cam = [];
faceDetector = [];
liveTimer = [];
streamFolder = "";
streamFramePath = "";
streamStopPath = "";
streamLogPath = "";
lastFrameTime = tic;
frameCounter = 0;
lastFaceBbox = [];
lastFaceImage = [];
lastPredictedLabel = "";
lastDistanceValue = NaN;
lastConfidenceValue = NaN;
lastDetectionFrame = -inf;
lastRecognitionFrame = -inf;
detectionIntervalFrames = 8;
recognitionIntervalFrames = 6;
maxCachedBboxAge = 30;
recentPredictionLabels = strings(0, 1);
recentPredictionConfidence = [];
predictionSmoothingWindow = 7;
isProcessingFrame = false;
liveImageHandle = [];
faceImageHandle = [];
bboxHandle = [];
labelHandle = [];

camFig = uifigure('Name', '摄像头实时人脸识别', 'Position', [160 90 1120 720], 'Color', [0.98 0.98 0.99]);
camFig.CloseRequestFcn = @onCloseCameraWindow;
grid = uigridlayout(camFig, [2 2]);
grid.RowHeight = {52, '1x'};
grid.ColumnWidth = {'1x', 300};
grid.Padding = [12 12 12 12];
grid.RowSpacing = 10;
grid.ColumnSpacing = 12;

controlGrid = uigridlayout(grid, [1 6]);
controlGrid.Layout.Row = 1;
controlGrid.Layout.Column = [1 2];
controlGrid.ColumnWidth = {190, '1x', 120, 120, 140, 180};
controlGrid.Padding = [0 0 0 0];
controlGrid.ColumnSpacing = 8;

cameraDropDown = uidropdown(controlGrid, 'Items', cameraNames, 'FontSize', 13);
cameraDropDown.Layout.Column = 1;
cameraDropDown.Value = cameraNames{defaultCameraIdx};

liveStatusLabel = uilabel(controlGrid, 'Text', '状态：等待开启摄像头', 'FontSize', 14, 'HorizontalAlignment', 'left');
liveStatusLabel.Layout.Column = 2;

startButton = uibutton(controlGrid, 'push', 'Text', '开启摄像头', 'FontSize', 14, 'ButtonPushedFcn', @onStartCamera);
startButton.Layout.Column = 3;

stopButton = uibutton(controlGrid, 'push', 'Text', '停止', 'FontSize', 14, 'Enable', 'off', 'ButtonPushedFcn', @onStopCamera);
stopButton.Layout.Column = 4;

settingsButton = uibutton(controlGrid, 'push', 'Text', '摄像头设置', 'FontSize', 14, 'ButtonPushedFcn', @onOpenCameraSettings);
settingsButton.Layout.Column = 5;

modelLabel = uilabel(controlGrid, 'Text', sprintf('%s  样本：%d/%d', cameraModeText, model.sampleCount, model.augmentedSampleCount), 'FontSize', 13, 'HorizontalAlignment', 'right');
modelLabel.Layout.Column = 6;

axLive = uiaxes(grid);
axLive.Layout.Row = 2;
axLive.Layout.Column = 1;
axis(axLive, 'image');
axLive.XTick = [];
axLive.YTick = [];
title(axLive, '实时画面');

sideGrid = uigridlayout(grid, [5 1]);
sideGrid.Layout.Row = 2;
sideGrid.Layout.Column = 2;
sideGrid.RowHeight = {42, 70, 34, '1x', 34};
sideGrid.Padding = [0 0 0 0];
sideGrid.RowSpacing = 8;

resultTitle = uilabel(sideGrid, 'Text', '识别结果', 'FontSize', 16, 'FontWeight', 'bold');
resultTitle.Layout.Row = 1;

resultLabel = uilabel(sideGrid, 'Text', '--', 'FontSize', 24, 'FontWeight', 'bold');
resultLabel.Layout.Row = 2;

distanceLabel = uilabel(sideGrid, 'Text', '相似度：--', 'FontSize', 14);
distanceLabel.Layout.Row = 3;

axFace = uiaxes(sideGrid);
axFace.Layout.Row = 4;
axis(axFace, 'image');
axFace.XTick = [];
axFace.YTick = [];
title(axFace, '检测到的人脸');

hintLabel = uilabel(sideGrid, ...
    'Text', '提示：请让人脸正对前置摄像头，光线尽量均匀。', ...
    'FontSize', 12, ...
    'HorizontalAlignment', 'left');
hintLabel.Layout.Row = 5;

    function onStartCamera(~, ~)
        try
            startButton.Enable = 'off';
            cameraDropDown.Enable = 'off';
            liveStatusLabel.Text = '状态：正在初始化摄像头和 MTCNN...';
            drawnow;

            if hasMatlabWebcam
                cam = webcam(cameraDropDown.Value);
            else
                startOpenCVCameraStream();
            end

            faceDetector = mtcnn.Detector('MinSize', 40, 'ConfidenceThresholds', [0.55, 0.65, 0.75]);
            liveTimer = timer( ...
                'ExecutionMode', 'fixedSpacing', ...
                'Period', 0.18, ...
                'BusyMode', 'drop', ...
                'TimerFcn', @onTimerFrame);
            start(liveTimer);

            stopButton.Enable = 'on';
            liveStatusLabel.Text = '状态：实时识别中';
        catch ME
            cleanupCamera();
            startButton.Enable = 'on';
            cameraDropDown.Enable = 'on';
            liveStatusLabel.Text = '状态：启动失败';
            uialert(camFig, getReport(ME, 'extended', 'hyperlinks', 'off'), '启动失败', 'Icon', 'error');
        end
    end

    function onStopCamera(~, ~)
        cleanupCamera();
        startButton.Enable = 'on';
        stopButton.Enable = 'off';
        cameraDropDown.Enable = 'on';
        liveStatusLabel.Text = '状态：已停止';
    end

    function onTimerFrame(~, ~)
        if ~isvalid(camFig)
            return;
        end
        if isProcessingFrame
            return;
        end

        isProcessingFrame = true;

        try
            frameCounter = frameCounter + 1;
            frame = getCameraFrame();
            if isempty(frame)
                liveStatusLabel.Text = '状态：等待摄像头画面...';
                isProcessingFrame = false;
                return;
            end

            if isCameraPrivacyFrame(frame)
                updateLiveFrame(frame);
                hideLiveOverlay();
                resultLabel.Text = '摄像头被锁定';
                distanceLabel.Text = '请关闭隐私模式或允许桌面应用访问摄像头';
                clearFacePreview();
                liveStatusLabel.Text = '状态：摄像头返回隐私锁画面，无法检测人脸';
                drawnow limitrate nocallbacks;
                isProcessingFrame = false;
                return;
            end

            if isCameraBlackFrame(frame)
                updateLiveFrame(frame);
                hideLiveOverlay();
                resultLabel.Text = '摄像头黑屏';
                distanceLabel.Text = '可能有旧摄像头进程占用，请停止后重新开启';
                clearFacePreview();
                liveStatusLabel.Text = '状态：摄像头返回黑帧，已建议重启摄像头流';
                drawnow limitrate nocallbacks;
                isProcessingFrame = false;
                return;
            end

            shouldDetect = isempty(lastFaceBbox) || frameCounter - lastDetectionFrame >= detectionIntervalFrames;
            faceBbox = [];
            if shouldDetect
                [detectedBbox, ~] = detectFaceMTCNNForGui(frame, faceDetector, [420, 640]);
                lastDetectionFrame = frameCounter;
                if ~isempty(detectedBbox)
                    lastFaceBbox = detectedBbox;
                    faceBbox = detectedBbox;
                end
            end

            if isempty(faceBbox) && ~isempty(lastFaceBbox) && frameCounter - lastDetectionFrame <= maxCachedBboxAge
                faceBbox = lastFaceBbox;
            end

            updateLiveFrame(frame);

            if isempty(faceBbox)
                hideLiveOverlay();
                resultLabel.Text = '未检测到人脸';
                distanceLabel.Text = '相似度：--';
                clearFacePreview();
                lastFaceBbox = [];
                lastFaceImage = [];
                lastPredictedLabel = "";
                liveStatusLabel.Text = sprintf('状态：实时识别中 %.1f FPS', frameRateText());
                drawnow limitrate nocallbacks;
                isProcessingFrame = false;
                return;
            end

            shouldRecognize = isempty(lastFaceImage) || frameCounter - lastRecognitionFrame >= recognitionIntervalFrames || shouldDetect;
            if shouldRecognize
                lastFaceImage = preprocessLiveFace(frame, faceBbox);
                [rawLabel, rawDistance, rawConfidence] = recognizeLiveFace(lastFaceImage, model);
                [lastPredictedLabel, lastDistanceValue, lastConfidenceValue] = smoothLivePrediction(rawLabel, rawDistance, rawConfidence);
                lastRecognitionFrame = frameCounter;
            end

            updateLiveOverlay(faceBbox, lastPredictedLabel);

            updateFacePreview(lastFaceImage);
            resultLabel.Text = char(lastPredictedLabel);
            distanceLabel.Text = sprintf('相似度：%.1f%%  距离：%.3f', lastConfidenceValue, lastDistanceValue);
            liveStatusLabel.Text = sprintf('状态：实时识别中 %.1f FPS', frameRateText());
            drawnow limitrate nocallbacks;
        catch ME
            liveStatusLabel.Text = sprintf('状态：识别帧失败：%s', ME.message);
        end
        isProcessingFrame = false;
    end

    function onOpenCameraSettings(~, ~)
        system('start ms-settings:privacy-webcam');
    end

    function updateLiveFrame(frame)
        if isempty(liveImageHandle) || ~isvalid(liveImageHandle)
            cla(axLive);
            liveImageHandle = image(axLive, frame);
            axis(axLive, 'image');
            axLive.XTick = [];
            axLive.YTick = [];
            title(axLive, '实时画面');
        else
            liveImageHandle.CData = frame;
        end
    end

    function updateLiveOverlay(faceBbox, labelText)
        if isempty(bboxHandle) || ~isvalid(bboxHandle)
            bboxHandle = rectangle(axLive, ...
                'Position', faceBbox, ...
                'EdgeColor', [0.0 0.65 0.20], ...
                'LineWidth', 2);
        else
            bboxHandle.Position = faceBbox;
            bboxHandle.Visible = 'on';
        end

        labelPosition = [faceBbox(1), max(1, faceBbox(2) - 12), 0];
        if isempty(labelHandle) || ~isvalid(labelHandle)
            labelHandle = text(axLive, labelPosition(1), labelPosition(2), char(labelText), ...
                'Color', 'yellow', ...
                'FontWeight', 'bold', ...
                'FontSize', 14, ...
                'BackgroundColor', [0 0 0], ...
                'Margin', 3, ...
                'Interpreter', 'none');
        else
            labelHandle.Position = labelPosition;
            labelHandle.String = char(labelText);
            labelHandle.Visible = 'on';
        end
    end

    function hideLiveOverlay()
        if ~isempty(bboxHandle) && isvalid(bboxHandle)
            bboxHandle.Visible = 'off';
        end
        if ~isempty(labelHandle) && isvalid(labelHandle)
            labelHandle.Visible = 'off';
        end
    end

    function updateFacePreview(faceImage)
        if isempty(faceImage)
            clearFacePreview();
            return;
        end

        if isempty(faceImageHandle) || ~isvalid(faceImageHandle)
            cla(axFace);
            faceImageHandle = image(axFace, repmat(faceImage, 1, 1, 3));
            axis(axFace, 'image');
            axFace.XTick = [];
            axFace.YTick = [];
            title(axFace, '检测到的人脸');
        else
            faceImageHandle.CData = repmat(faceImage, 1, 1, 3);
        end
    end

    function clearFacePreview()
        if ~isempty(faceImageHandle) && isvalid(faceImageHandle)
            faceImageHandle.CData = uint8(255 * ones(160, 160, 3));
        else
            cla(axFace);
        end
    end

    function startOpenCVCameraStream()
        stopExistingOpenCVStreams();
        streamFolder = string(tempname);
        mkdir(streamFolder);
        streamFramePath = string(fullfile(streamFolder, 'latest.jpg'));
        streamStopPath = string(fullfile(streamFolder, 'stop.flag'));
        streamLogPath = string(fullfile(streamFolder, 'camera.log'));

        selectedCamera = string(cameraDropDown.Value);
        tokens = regexp(selectedCamera, 'OpenCV 摄像头\s+(\d+)\s+(\w+)', 'tokens', 'once');
        if isempty(tokens)
            cameraIndex = 0;
            backendName = "DSHOW";
        else
            cameraIndex = str2double(tokens{1});
            backendName = string(tokens{2});
        end
        if isempty(cameraIndex) || isnan(cameraIndex)
            cameraIndex = 0;
        end

        cmd = sprintf('start "" /B "%s" "%s" %d "%s" "%s" "%s" "%s"', ...
            pythonExe, opencvHelperPath, cameraIndex, streamFramePath, streamStopPath, streamLogPath, backendName);
        [status, cmdout] = system(cmd);
        if status ~= 0
            error('OpenCV 摄像头进程启动失败：%s', cmdout);
        end
    end

    function stopExistingOpenCVStreams()
        cleanupCmd = ['powershell -NoProfile -Command "', ...
            'Get-CimInstance Win32_Process | Where-Object { $_.Name -eq ''python.exe'' -and $_.CommandLine -like ''*opencv_camera_stream.py*'' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }', ...
            '"'];
        system(cleanupCmd);
        pause(0.2);
    end

    function frame = getCameraFrame()
        frame = [];
        if hasMatlabWebcam
            if isempty(cam)
                return;
            end
            frame = im2uint8(ensureRGB(snapshot(cam)));
            return;
        end

        if strlength(streamFramePath) == 0 || ~isfile(streamFramePath)
            return;
        end

        try
            frame = im2uint8(ensureRGB(imread(streamFramePath)));
        catch
            frame = [];
        end
    end

    function fps = frameRateText()
        elapsed = toc(lastFrameTime);
        lastFrameTime = tic;
        fps = 1 / max(elapsed, eps);
    end

    function onCloseCameraWindow(~, ~)
        cleanupCamera();
        delete(camFig);
    end

    function cleanupCamera()
        if ~isempty(liveTimer) && isvalid(liveTimer)
            stop(liveTimer);
            delete(liveTimer);
        end
        liveTimer = [];
        frameCounter = 0;
        lastFaceBbox = [];
        lastFaceImage = [];
        lastPredictedLabel = "";
        lastDistanceValue = NaN;
        lastConfidenceValue = NaN;
        lastDetectionFrame = -inf;
        lastRecognitionFrame = -inf;
        recentPredictionLabels = strings(0, 1);
        recentPredictionConfidence = [];
        isProcessingFrame = false;
        liveImageHandle = [];
        faceImageHandle = [];
        bboxHandle = [];
        labelHandle = [];

        if ~hasMatlabWebcam && strlength(streamStopPath) > 0
            try
                fid = fopen(streamStopPath, 'w');
                if fid > 0
                    fclose(fid);
                end
            catch
            end
            pause(0.2);
        end

        cam = [];
        faceDetector = [];
    end

    function [label, distanceValue, confidenceValue] = smoothLivePrediction(rawLabel, rawDistance, rawConfidence)
        recentPredictionLabels(end + 1, 1) = rawLabel;
        recentPredictionConfidence(end + 1, 1) = rawConfidence;
        if numel(recentPredictionLabels) > predictionSmoothingWindow
            recentPredictionLabels = recentPredictionLabels(end - predictionSmoothingWindow + 1:end);
            recentPredictionConfidence = recentPredictionConfidence(end - predictionSmoothingWindow + 1:end);
        end

        labelsInWindow = unique(recentPredictionLabels);
        scores = zeros(numel(labelsInWindow), 1);
        for ii = 1:numel(labelsInWindow)
            mask = recentPredictionLabels == labelsInWindow(ii);
            scores(ii) = sum(recentPredictionConfidence(mask)) + 12 * nnz(mask);
        end
        [~, bestIdx] = max(scores);
        label = labelsInWindow(bestIdx);
        confidenceValue = mean(recentPredictionConfidence(recentPredictionLabels == label), 'omitnan');
        distanceValue = rawDistance;
    end
end

function showSavedFacesGallery(samples)
galleryFig = uifigure('Name', '本次保存的人脸图片', 'Position', [180 120 980 660], 'Color', [0.98 0.98 0.99]);
grid = uigridlayout(galleryFig, [1 2]);
grid.ColumnWidth = {330, '1x'};
grid.Padding = [12 12 12 12];
grid.ColumnSpacing = 12;

names = compose("%s | %s", samples.FileName, samples.TrueLabel);
listBox = uilistbox(grid, ...
    'Items', cellstr(names), ...
    'ItemsData', cellstr(samples.PreprocessedPath), ...
    'FontSize', 13);
listBox.Layout.Column = 1;

rightGrid = uigridlayout(grid, [2 1]);
rightGrid.Layout.Column = 2;
rightGrid.RowHeight = {36, '1x'};
titleLabel = uilabel(rightGrid, 'Text', '', 'FontSize', 14, 'FontWeight', 'bold');
ax = uiaxes(rightGrid);
ax.Layout.Row = 2;
axis(ax, 'image');
ax.XTick = [];
ax.YTick = [];

listBox.ValueChangedFcn = @(~,~) renderSelectedFace();
if ~isempty(listBox.ItemsData)
    listBox.Value = listBox.ItemsData{1};
    renderSelectedFace();
end

    function renderSelectedFace()
        imgPath = string(listBox.Value);
        selectedIdx = find(samples.PreprocessedPath == imgPath, 1);
        if isempty(selectedIdx) || ~isfile(imgPath)
            cla(ax);
            titleLabel.Text = '图片不存在';
            return;
        end
        titleLabel.Text = char(samples.FileName(selectedIdx));
        imshow(imread(imgPath), 'Parent', ax);
    end
end

function showSamplePreview(row)
originalPath = string(row.OriginalPath(1));
preprocessedPath = string(row.PreprocessedPath(1));

previewFig = uifigure('Name', '错分样本预览', 'Position', [220 140 1120 620], 'Color', [0.98 0.98 0.99]);
grid = uigridlayout(previewFig, [2 2]);
grid.RowHeight = {48, '1x'};
grid.ColumnWidth = {'1x', '1x'};
grid.Padding = [12 12 12 12];
grid.RowSpacing = 8;
grid.ColumnSpacing = 12;

infoText = sprintf('%s    True: %s    Predicted: %s', row.FileName(1), row.TrueLabel(1), row.PredictedLabel(1));
infoLabel = uilabel(grid, 'Text', infoText, 'FontSize', 14, 'FontWeight', 'bold');
infoLabel.Layout.Row = 1;
infoLabel.Layout.Column = [1 2];

axOriginal = uiaxes(grid);
axOriginal.Layout.Row = 2;
axOriginal.Layout.Column = 1;
title(axOriginal, '原图');
axis(axOriginal, 'image');
axOriginal.XTick = [];
axOriginal.YTick = [];

axPreprocessed = uiaxes(grid);
axPreprocessed.Layout.Row = 2;
axPreprocessed.Layout.Column = 2;
title(axPreprocessed, '预处理图');
axis(axPreprocessed, 'image');
axPreprocessed.XTick = [];
axPreprocessed.YTick = [];

showImageOrMessage(axOriginal, originalPath);
showImageOrMessage(axPreprocessed, preprocessedPath);
end

function showImageOrMessage(ax, imgPath)
if isfile(imgPath)
    imshow(imread(imgPath), 'Parent', ax);
else
    cla(ax);
    text(ax, 0.5, 0.5, sprintf('文件不存在\n%s', imgPath), 'HorizontalAlignment', 'center', 'Interpreter', 'none');
    ax.XLim = [0 1];
    ax.YLim = [0 1];
end
end

function samples = normalizeSavedSamples(samples)
needed = {'FileName', 'TrueLabel', 'OriginalPath', 'PreprocessedPath'};
for i = 1:numel(needed)
    if ~ismember(needed{i}, samples.Properties.VariableNames)
        samples.(needed{i}) = strings(height(samples), 1);
    end
end
samples = samples(:, needed);
samples.FileName = string(samples.FileName);
samples.TrueLabel = string(samples.TrueLabel);
samples.OriginalPath = string(samples.OriginalPath);
samples.PreprocessedPath = string(samples.PreprocessedPath);
end

function model = trainLiveRecognitionModel(samples)
featureMatrix = [];
labels = strings(0, 1);
for i = 1:height(samples)
    img = im2uint8(imread(samples.PreprocessedPath(i)));
    augmentedFaces = makeLiveTrainingAugmentations(img);
    for a = 1:numel(augmentedFaces)
        featureMatrix(end + 1, :) = extractLiveRecognitionFeatures(augmentedFaces{a}); %#ok<AGROW>
        labels(end + 1, 1) = samples.TrueLabel(i); %#ok<AGROW>
    end
end

[standardizedFeatures, featureMu, featureSigma] = standardizeLiveTrainFeatures(featureMatrix);
[coeff, score, ~, ~, explained, pcaMu] = pca(standardizedFeatures, ...
    'NumComponents', min([260, size(standardizedFeatures, 1) - 1, size(standardizedFeatures, 2)]));
componentCount = find(cumsum(explained) >= 90, 1);
if isempty(componentCount)
    componentCount = size(score, 2);
end
componentCount = max(1, min(componentCount, size(score, 2)));
trainFeatures = score(:, 1:componentCount);
classLabels = unique(labels);
classCentroids = zeros(numel(classLabels), size(trainFeatures, 2));
for c = 1:numel(classLabels)
    classCentroids(c, :) = mean(trainFeatures(labels == classLabels(c), :), 1);
end

model = struct( ...
    'labels', labels, ...
    'classLabels', classLabels, ...
    'featureMu', featureMu, ...
    'featureSigma', featureSigma, ...
    'pcaCoeff', coeff(:, 1:componentCount), ...
    'pcaMu', pcaMu, ...
    'trainFeatures', trainFeatures, ...
    'classCentroids', classCentroids, ...
    'componentCount', componentCount, ...
    'sampleCount', height(samples), ...
    'augmentedSampleCount', numel(labels));
end

function idx = chooseFrontCamera(cameraNames)
idx = 1;
for i = 1:numel(cameraNames)
    name = lower(string(cameraNames{i}));
    if contains(name, "front") || contains(name, "user") || contains(name, "integrated") || contains(name, "内置")
        idx = i;
        return;
    end
end
end

function augmentedFaces = makeLiveTrainingAugmentations(img)
base = shijian_face_core('preprocessFaceCrop', img);
augmentedFaces = cell(1, 9);
augmentedFaces{1} = base;
augmentedFaces{2} = fliplr(base);
augmentedFaces{3} = imadjust(base, stretchlim(base, [0.02 0.98]), []);
augmentedFaces{4} = imresize(imcrop(base, [8 8 143 143]), [160 160]);
augmentedFaces{5} = imresize(imcrop(base, [1 1 151 151]), [160 160]);
augmentedFaces{6} = imadjust(base, [], [], 0.80);
augmentedFaces{7} = imadjust(base, [], [], 1.20);
augmentedFaces{8} = imrotate(base, 3, 'bilinear', 'crop');
augmentedFaces{9} = imrotate(base, -3, 'bilinear', 'crop');
end

function rgb = ensureRGB(img)
rgb = shijian_face_core('ensureRGB', img);
end

function locked = isCameraPrivacyFrame(frame)
frame = im2uint8(frame);
gray = rgb2gray(ensureRGB(frame));
channelDiff = max(abs(double(frame(:, :, 1)) - double(frame(:, :, 2))), [], 'all') + ...
    max(abs(double(frame(:, :, 2)) - double(frame(:, :, 3))), [], 'all');

frameStd = std(double(gray(:)));
frameRange = double(max(gray(:))) - double(min(gray(:)));
[h, w] = size(gray);
centerPatch = gray(round(h * 0.35):round(h * 0.65), round(w * 0.35):round(w * 0.65));
centerHasWhiteIcon = nnz(centerPatch > 220) > 80;

locked = channelDiff < 8 && frameStd < 15 && frameRange > 60 && centerHasWhiteIcon;
end

function black = isCameraBlackFrame(frame)
gray = rgb2gray(ensureRGB(im2uint8(frame)));
frameMean = mean(double(gray(:)));
frameStd = std(double(gray(:)));
brightRatio = nnz(gray > 40) / numel(gray);
black = frameMean < 15 || (frameMean < 30 && frameStd < 12 && brightRatio < 0.02);
end

function [bbox, score] = detectFaceMTCNNForGui(img, detector, maxDetectionSides)
[bbox, score] = shijian_face_core('detectFaceMTCNN', img, detector, maxDetectionSides);
if ~isempty(bbox)
    bbox = expandAndClipBboxForGui(bbox, 0.15, size(img));
end
end

function box = expandAndClipBboxForGui(bbox, marginRatio, imgSize)
box = shijian_face_core('expandAndClipBbox', bbox, marginRatio, imgSize);
end

function faceImage = preprocessLiveFace(frame, faceBbox)
faceCrop = imcrop(frame, faceBbox);
if isempty(faceCrop)
    faceCrop = frame;
end
faceImage = shijian_face_core('preprocessFaceCrop', faceCrop);
end

function gray = toGrayForGui(img)
gray = shijian_face_core('toGray', img);
end

function [predictedLabel, distanceValue, confidenceValue] = recognizeLiveFace(faceImage, model)
queryFaces = {faceImage, fliplr(faceImage), imadjust(faceImage, stretchlim(faceImage, [0.02 0.98]), [])};
queryFeatures = zeros(numel(queryFaces), size(model.pcaCoeff, 2));
for q = 1:numel(queryFaces)
    feature = extractLiveRecognitionFeatures(queryFaces{q});
    feature = standardizeLiveTestFeatures(feature, model.featureMu, model.featureSigma);
    queryFeatures(q, :) = (feature - model.pcaMu) * model.pcaCoeff;
end

bestScore = inf;
secondScore = inf;
predictedLabel = model.classLabels(1);
for q = 1:size(queryFeatures, 1)
    sampleDistances = pdist2(queryFeatures(q, :), model.trainFeatures, 'cosine');
    centroidDistances = pdist2(queryFeatures(q, :), model.classCentroids, 'cosine');

    for c = 1:numel(model.classLabels)
        classDistances = sort(sampleDistances(model.labels == model.classLabels(c)));
        topCount = min(5, numel(classDistances));
        topMeanDistance = mean(classDistances(1:topCount));
        classScore = 0.75 * topMeanDistance + 0.25 * centroidDistances(c);

        if classScore < bestScore
            secondScore = bestScore;
            bestScore = classScore;
            predictedLabel = model.classLabels(c);
        elseif classScore < secondScore
            secondScore = classScore;
        end
    end
end

distanceValue = bestScore;
margin = max(0, secondScore - bestScore);
confidenceValue = max(0, min(100, (1 - distanceValue) * 80 + margin * 220));
end

function featureVector = extractLiveRecognitionFeatures(faceGray)
featureVector = shijian_face_core('extractRecognitionFeatures', faceGray);
end

function [standardizedFeatures, featureMu, featureSigma] = standardizeLiveTrainFeatures(features)
[standardizedFeatures, featureMu, featureSigma] = shijian_face_core('standardizeTrainFeatures', features);
end

function standardizedFeatures = standardizeLiveTestFeatures(features, featureMu, featureSigma)
standardizedFeatures = shijian_face_core('standardizeTestFeatures', features, featureMu, featureSigma);
end

function textValue = formatPercent(accuracy, correctCount, testCount)
if isnan(accuracy)
    textValue = '--';
else
    textValue = sprintf('%.2f%% (%d/%d)', accuracy, correctCount, testCount);
end
end

function textValue = formatRate(rate, successCount, totalCount)
if isnan(rate)
    textValue = '--';
else
    textValue = sprintf('%.2f%% (%d/%d)', rate, successCount, totalCount);
end
end

function tbl = normalizeMisclassifiedTable(tbl)
needed = {'FileName', 'TrueLabel', 'PredictedLabel', 'OriginalPath', 'PreprocessedPath'};
for i = 1:numel(needed)
    if ~ismember(needed{i}, tbl.Properties.VariableNames)
        tbl.(needed{i}) = strings(height(tbl), 1);
    end
end
tbl = tbl(:, needed);
end

function tbl = normalizeFailureTable(tbl)
needed = {'FileName', 'Reason'};
for i = 1:numel(needed)
    if ~ismember(needed{i}, tbl.Properties.VariableNames)
        tbl.(needed{i}) = strings(height(tbl), 1);
    end
end
tbl = tbl(:, needed);
end

function tbl = emptySampleTable()
tbl = table(strings(0, 1), strings(0, 1), strings(0, 1), strings(0, 1), strings(0, 1), ...
    'VariableNames', {'FileName', 'TrueLabel', 'PredictedLabel', 'OriginalPath', 'PreprocessedPath'});
end

function tbl = emptyFailureTable()
tbl = table(strings(0, 1), strings(0, 1), ...
    'VariableNames', {'FileName', 'Reason'});
end
