%% One vs All Top-Down Model
%% Last updated: 15th April 2015

% % 	"options:\n"
% % 	"-s type : set type of solver (default 1)\n"
% % 	"  for multi-class classification\n"
% % 	"	 0 -- L2-regularized logistic regression (primal)\n"
% % 	"	 1 -- L2-regularized L2-loss support vector classification (dual)\n"
% % 	"	 2 -- L2-regularized L2-loss support vector classification (primal)\n"
% % 	"	 3 -- L2-regularized L1-loss support vector classification (dual)\n"
% % 	"	 4 -- support vector classification by Crammer and Singer\n"
% % 	"	 5 -- L1-regularized L2-loss support vector classification\n"
% % 	"	 6 -- L1-regularized logistic regression\n"
% % 	"	 7 -- L2-regularized logistic regression (dual)\n"
% % 	"  for regression\n"
% % 	"	11 -- L2-regularized L2-loss support vector regression (primal)\n"
% % 	"	12 -- L2-regularized L2-loss support vector regression (dual)\n"
% % 	"	13 -- L2-regularized L1-loss support vector regression (dual)\n"
% % 	"-c cost : set the parameter C (default 1)\n"
% % 	"-p epsilon : set the epsilon in loss function of SVR (default 0.1)\n"
% % 	"-e epsilon : set tolerance of termination criterion\n"
% % 	"	-s 0 and 2\n"
% % 	"		|f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,\n"
% % 	"		where f is the primal function and pos/neg are # of\n"
% % 	"		positive/negative data (default 0.01)\n"
% % 	"	-s 11\n"
% % 	"		|f'(w)|_2 <= eps*|f'(w0)|_2 (default 0.001)\n"
% % 	"	-s 1, 3, 4, and 7\n"
% % 	"		Dual maximal violation <= eps; similar to libsvm (default 0.1)\n"
% % 	"	-s 5 and 6\n"
% % 	"		|f'(w)|_1 <= eps*min(pos,neg)/l*|f'(w0)|_1,\n"
% % 	"		where f is the primal function (default 0.01)\n"
% % 	"	-s 12 and 13\n"
% % 	"		|f'(alpha)|_1 <= eps |f'(alpha0)|,\n"
% % 	"		where f is the dual function (default 0.1)\n"
% % 	"-B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)\n"
% % 	"-wi weight: weights adjust the parameter C of different classes (see README for details)\n"
% % 	"-v n: n-fold cross validation mode\n"
% % 	"-q : quiet mode (no outputs)\n"

clc
clear all
close all
localfilepath='/Users/deb/GitHub/code-bioengineering'
%% Dataset path
datasetPath = '../../../../RankingModels/Dataset/';
%% Dataset Folder Name
datasetFolderName = 'diatoms';

%% User parameter Regularization C, Bias B
if(strcmp(datasetFolderName, 'CLEF'))
    noOfLevels = 3;
elseif(strcmp(datasetFolderName, 'NG'))
    noOfLevels = 3;
elseif(strcmp(datasetFolderName, 'IPC'))
    noOfLevels = 3;
elseif(strcmp(datasetFolderName, 'Bioasq'))
    noOfLevels = 5;
elseif(strcmp(datasetFolderName, 'Bioasq_small'))
    noOfLevels = 5;
elseif(strcmp(datasetFolderName, 'Bioasq'))
    noOfLevels = 5;    
elseif(strcmp(datasetFolderName, 'Bioasq'))
    noOfLevels = 3;
else
end

%% User parameter Regularization C, Bias B
C = [0.001 0.01 0.1 1 10 100 1000];
% C = [1000];
B = 1;
%% 7 for L2 regularized logistic regression (dual), 0 (for primal)
modelLearn = 0;

%% Full dataset path
fullPath = strcat(datasetPath, datasetFolderName);

%% liblinear path
addpath(localfilepath.');
%% Dataset Path
addpath(fullPath);

%% Select path based on Operating System
if ispc
    partialPath = [localpath '/NodeRemovalInHierarchy/phdthesis_code/TDClassification1vsAll/Result'];
elseif isunix
	partialPath = [localpath '/SVM/Result'];
else
    % other operating system
end 

filename = 'train.txt';
filePathTrain = strcat(fullPath, '/', filename);
[label_vector, instance_matrix] = libsvmread(filePathTrain);
[id d] = size(instance_matrix);
fprintf('Finished Reading train File\n');

%% Total number of possible labels
labelData = unique(sort(label_vector));

%% Choose best parameter level wise
for nL = 1:noOfLevels
    %% Load the label info for each node (merged labels are in one line)
    groupMatrix = zeros(length(labelData), length(labelData));
    openFile = ['level' num2str(nL) datasetFolderName '.txt'];
    fid = fopen(openFile);
    tline = fgetl(fid);
    labelSet = labelData;
    nLine = 0;
    while ischar(tline)
         nLine = nLine + 1;
         splitLine = strread(tline,'%s','delimiter',' ');
         numElement = length(splitLine);
         groupMatrix(nLine, 1) = str2num(cell2mat(splitLine(1)));
         repItem = str2num(cell2mat(splitLine(1)));
         for numCategories = 2:numElement
             groupMatrix(nLine, numCategories) = str2num(cell2mat(splitLine(numCategories)));
             ind = find(labelSet == str2num(cell2mat(splitLine(numCategories))));
             labelSet(ind) = repItem;
         end
         tline = fgetl(fid);
    end
    fclose(fid);  
    
    %% All labels
    labelSet = unique(sort(labelSet));
    [noLabels noColLabels] = size(labelSet);
    for k = 1:length(label_vector)
            %% This is the class label to be used for the one vs all classification
            [r c] = find(groupMatrix == label_vector(k));
            if(isempty(r) || isempty(c))
                Y(k, 1) = -1;  
            else
                Y(k, 1) = groupMatrix(r, 1); 
            end
    end
    
    %% validation Set
    filenameT = 'validation.txt';
    filePathTest = strcat(fullPath, '/', filenameT);
    [label_vectorT, instance_matrixT] = libsvmread(filePathTest);
    fprintf('Finished Reading test File\n');
    
    for k = 1:length(label_vectorT)
            %% This is the class label to be used for the one vs all classification
            [r c] = find(groupMatrix == label_vectorT(k));
            if(isempty(r) || isempty(c))
                YT(k, 1) = -1;  
            else
                YT(k, 1) = groupMatrix(r, 1);
            end
    end
        
    %% microF1, macroF1, avr Runtime, noE
    micMac = zeros(length(C), 4);
    for regTest = 1:length(C)
        cmd = ['-c ', num2str(C(regTest)), ' -s ', num2str(modelLearn), ' -B ', num2str(B)];

        %% train Model
        startModelParameterLearning = tic;
        model = ovrtrain(Y, instance_matrix, cmd);
        stopModelParameterLearning = toc(startModelParameterLearning)/noLabels;
        
        %% valid test row
        r = find(YT >= 0);
        
        trueLabel = find(Y >= 0);
        labelVal = unique(Y(trueLabel, :));
        
        %% test Model
        [pred ac decv] = ovrpredict(YT(r, :), instance_matrixT(r, :), model);
        [microCalMatrix microPrecisionVal microRecallVal microF1Val macroPrecisionVal macroRecallVal macroF1Val MCCVal] = microMacroMCCVal(YT(r, :), pred, labelVal(:, 1));
        micMac(regTest, 1) = microF1Val; micMac(regTest, 2) = macroF1Val; 
        micMac(regTest, 3) = stopModelParameterLearning; micMac(regTest, 4) = size(r, 1);
        
        filename = ['libL2RegularizedSTL_finalThetaVector_train_dataset_' datasetFolderName '_C_' num2str(C(regTest)) '_B_' num2str(B) '_level_' num2str(nL) '.mat'];
        fullFileName = fullfile(partialPath, filename);
        save(fullFileName, '-struct', 'model');
    end
    filename = ['libL2RegularizedSTL_accMicroMacroAvgTimeModelLearning_train_dataset_' datasetFolderName '_C_' num2str(C(regTest)) '_B_' num2str(B) '_level_' num2str(nL) '.txt'];
    fullFileName = fullfile(partialPath, filename);
    save(fullFileName, 'micMac', '-ascii');
end
