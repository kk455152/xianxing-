function varargout = shijian_face_core(action, varargin)
switch action
    case 'ensureRGB'
        varargout{1} = ensureRGB(varargin{1});
    case 'toGray'
        varargout{1} = toGray(varargin{1});
    case 'detectFaceMTCNN'
        [varargout{1}, varargout{2}] = detectFaceMTCNN(varargin{:});
    case 'expandAndClipBbox'
        varargout{1} = expandAndClipBbox(varargin{:});
    case 'extractRecognitionFeatures'
        varargout{1} = extractRecognitionFeatures(varargin{1});
    case 'standardizeTrainFeatures'
        [varargout{1}, varargout{2}, varargout{3}] = standardizeTrainFeatures(varargin{1});
    case 'standardizeTestFeatures'
        varargout{1} = standardizeTestFeatures(varargin{:});
    case 'preprocessFaceCrop'
        varargout{1} = preprocessFaceCrop(varargin{1});
    otherwise
        error('Unknown shijian_face_core action: %s', action);
end
end

function rgb = ensureRGB(img)
if size(img, 3) == 1
    rgb = repmat(img, 1, 1, 3);
elseif size(img, 3) >= 3
    rgb = img(:, :, 1:3);
else
    rgb = img;
end
end

function gray = toGray(img)
if size(img, 3) == 3
    gray = rgb2gray(img);
else
    gray = img;
end
end

function [bbox, score] = detectFaceMTCNN(img, detector, maxDetectionSides)
bbox = [];
score = NaN;
maxSideOriginal = max(size(img, 1), size(img, 2));

for s = 1:numel(maxDetectionSides)
    scale = min(1, maxDetectionSides(s) / maxSideOriginal);
    if scale < 1
        detectImg = imresize(img, scale);
    else
        detectImg = img;
    end

    [bboxes, scores] = detector.detect(detectImg);
    if isempty(bboxes)
        continue;
    end

    areas = bboxes(:, 3) .* bboxes(:, 4);
    [~, idx] = max(scores(:) .* areas(:));
    bbox = double(bboxes(idx, :)) ./ scale;
    score = double(scores(idx));
    return;
end
end

function box = expandAndClipBbox(bbox, marginRatio, imgSize)
x = double(bbox(1));
y = double(bbox(2));
w = double(bbox(3));
h = double(bbox(4));
marginX = marginRatio * w;
marginY = marginRatio * h;

x1 = max(1, x - marginX);
y1 = max(1, y - marginY);
x2 = min(imgSize(2), x + w + marginX);
y2 = min(imgSize(1), y + h + marginY);
box = [x1, y1, max(1, x2 - x1), max(1, y2 - y1)];
end

function faceNormalized = preprocessFaceCrop(faceCrop)
faceGray = im2uint8(toGray(ensureRGB(faceCrop)));
faceHistEq = histeq(faceGray);
faceNormalized = imresize(faceHistEq, [160, 160]);
end

function featureVector = extractRecognitionFeatures(faceGray)
faceGray = im2single(toGray(faceGray));
faceGray = imresize(faceGray, [160, 160]);
hogFeatures = extractHOGFeatures(faceGray, 'CellSize', [16 16]);
lbpFeatures = extractLBPFeatures(faceGray, 'CellSize', [32 32], 'Normalization', 'L2');
dctCoefficients = dct2(faceGray);
dctFeatures = reshape(dctCoefficients(1:40, 1:40), 1, []);
featureVector = [hogFeatures, lbpFeatures, dctFeatures];
end

function [standardizedFeatures, featureMu, featureSigma] = standardizeTrainFeatures(features)
featureMu = mean(double(features), 1);
featureSigma = std(double(features), 0, 1);
featureSigma(featureSigma == 0 | ~isfinite(featureSigma)) = 1;
standardizedFeatures = standardizeTestFeatures(features, featureMu, featureSigma);
end

function standardizedFeatures = standardizeTestFeatures(features, featureMu, featureSigma)
standardizedFeatures = (double(features) - featureMu) ./ featureSigma;
standardizedFeatures(~isfinite(standardizedFeatures)) = 0;
end
