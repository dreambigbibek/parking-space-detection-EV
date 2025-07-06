% Load ground truth
load('parkingTrainGTfinished.mat');
labelTbl = gTruth.LabelData;

% Class names
classNames = {'Accessible', 'Charger', 'EV'};

% Initialize
counts = zeros(1,3);
ranges = zeros(1,3);
outlierCounts = zeros(1,3);

for i = 1:3
    colData = labelTbl.(classNames{i});
    
    % Count total objects
    counts(i) = sum(cellfun(@(x) size(x,1), colData));
    
    % Extract all boxes safely
    allBoxes = cell2mat(cellfun(@(x) x(size(x,2)>=4,:), colData(~cellfun(@isempty,colData)), 'UniformOutput', false));
    
    % Compute aspect ratios and range
    if ~isempty(allBoxes)
        aspectRatios = allBoxes(:,3) ./ allBoxes(:,4);
        ranges(i) = range(aspectRatios);
        
        % Compute area and outliers
        areas = allBoxes(:,3) .* allBoxes(:,4);
        Q1 = prctile(areas, 25);
        Q3 = prctile(areas, 75);
        IQR = Q3 - Q1;
        outliers = (areas < Q1 - 1.5*IQR) | (areas > Q3 + 1.5*IQR);
        outlierCounts(i) = sum(outliers);
    end
end

%Total EV signs
fprintf("Total EV signs: %d\n", counts(3));

%Number of images with Accessible objects
numAccessibleImages = sum(cellfun(@(x) ~isempty(x), labelTbl.Accessible));
fprintf("Images with Accessible: %d\n", numAccessibleImages);

% Class with fewest objects
[~, idxFewest] = min(counts);
fprintf("Class with fewest objects: %s\n", classNames{idxFewest});

%Class with largest aspect ratio range
[~, idxRange] = max(ranges);
fprintf("Class with widest aspect ratio range: %s\n", classNames{idxRange});

% Class with most area outliers
[~, idxOutlier] = max(outlierCounts);
fprintf("Class with most area outliers: %s\n", classNames{idxOutlier});

% Class model is likely to struggle with (heuristic)
struggleScore = counts.^-1 + ranges + outlierCounts;
[~, idxStruggle] = max(struggleScore);
fprintf("Class model may struggle with most: %s\n", classNames{idxStruggle});
