%% STEP 1: Data Preparation

% Clear workspace and close all figures
close all
clear all

% Load the training and testing sets along with the corresponding labels
load('Train_Images.mat');
load('Train_Labels.mat');
load('Test_Images.mat');
load('Test_Labels.mat');

% Create testing and training variables for each class/digit (0 to 9)
Train_Im_0 = Train_Images(:,Train_labels == 0); Train_Lb_0 = Train_labels(Train_labels == 0);
Train_Im_1 = Train_Images(:,Train_labels == 1); Train_Lb_1 = Train_labels(Train_labels == 1);
Train_Im_2 = Train_Images(:,Train_labels == 2); Train_Lb_2 = Train_labels(Train_labels == 2);
Train_Im_3 = Train_Images(:,Train_labels == 3); Train_Lb_3 = Train_labels(Train_labels == 3);
Train_Im_4 = Train_Images(:,Train_labels == 4); Train_Lb_4 = Train_labels(Train_labels == 4);
Train_Im_5 = Train_Images(:,Train_labels == 5); Train_Lb_5 = Train_labels(Train_labels == 5);
Train_Im_6 = Train_Images(:,Train_labels == 6); Train_Lb_6 = Train_labels(Train_labels == 6);
Train_Im_7 = Train_Images(:,Train_labels == 7); Train_Lb_7 = Train_labels(Train_labels == 7);
Train_Im_8 = Train_Images(:,Train_labels == 8); Train_Lb_8 = Train_labels(Train_labels == 8);
Train_Im_9 = Train_Images(:,Train_labels == 9); Train_Lb_9 = Train_labels(Train_labels == 9);

Test_Im_0 = Test_Images(:,Test_labels == 0); Test_Lb_0 = Test_labels(Test_labels == 0);
Test_Im_1 = Test_Images(:,Test_labels == 1); Test_Lb_1 = Test_labels(Test_labels == 1);
Test_Im_2 = Test_Images(:,Test_labels == 2); Test_Lb_2 = Test_labels(Test_labels == 2);
Test_Im_3 = Test_Images(:,Test_labels == 3); Test_Lb_3 = Test_labels(Test_labels == 3);
Test_Im_4 = Test_Images(:,Test_labels == 4); Test_Lb_4 = Test_labels(Test_labels == 4);
Test_Im_5 = Test_Images(:,Test_labels == 5); Test_Lb_5 = Test_labels(Test_labels == 5);
Test_Im_6 = Test_Images(:,Test_labels == 6); Test_Lb_6 = Test_labels(Test_labels == 6);
Test_Im_7 = Test_Images(:,Test_labels == 7); Test_Lb_7 = Test_labels(Test_labels == 7);
Test_Im_8 = Test_Images(:,Test_labels == 8); Test_Lb_8 = Test_labels(Test_labels == 8);
Test_Im_9 = Test_Images(:,Test_labels == 9); Test_Lb_9 = Test_labels(Test_labels == 9);

%% STEP 2: HOG Features calculation

HOG_Train = [];
HOG_Test  = [];

dim = [6 6];

N_Test = [];
N_Train = [];


for j = 0:9
    
    disp(['Digit: ',num2str(j)])
    
    if j == 0
        Test_In  = Test_Im_0;
        Train_In = Train_Im_0;
        n1 = size(Test_Im_0, 2);
        n2 = size(Train_Im_0, 2);
    elseif j == 1
        Test_In  = Test_Im_1;
        Train_In = Train_Im_1;
        n1 = size(Test_Im_1, 2);
        n2 = size(Train_Im_1, 2);
    elseif j == 2
        Test_In  = Test_Im_2;
        Train_In = Train_Im_2;
        n1 = size(Test_Im_2, 2);
        n2 = size(Train_Im_2, 2);
    elseif j == 3
        Test_In  = Test_Im_3;
        Train_In = Train_Im_3;
        n1 = size(Test_Im_3, 2);
        n2 = size(Train_Im_3, 2);
    elseif j == 4
        Test_In  = Test_Im_4;
        Train_In = Train_Im_4;
        n1 = size(Test_Im_4, 2);
        n2 = size(Train_Im_4, 2);
    elseif j == 5
        Test_In  = Test_Im_5;
        Train_In = Train_Im_5;
        n1 = size(Test_Im_5, 2);
        n2 = size(Train_Im_5, 2);
    elseif j == 6
        Test_In  = Test_Im_6;
        Train_In = Train_Im_6;
        n1 = size(Test_Im_6, 2);
        n2 = size(Train_Im_6, 2);
    elseif j == 7
        Test_In  = Test_Im_7;
        Train_In = Train_Im_7;
        n1 = size(Test_Im_7, 2);
        n2 = size(Train_Im_7, 2);
    elseif j == 8
        Test_In  = Test_Im_8;
        Train_In = Train_Im_8;
        n1 = size(Test_Im_8, 2);
        n2 = size(Train_Im_8, 2);
    elseif j == 9
        Test_In  = Test_Im_9;
        Train_In = Train_Im_9;
        n1 = size(Test_Im_9, 2);
        n2 = size(Train_Im_9, 2);
    end
    
    for i = 1:n1
        A = reshape(Test_In(:,i) , [28 28]);
        [feat_test,~] = extractHOGFeatures(A,'CellSize',dim);
        HOG_Test = [HOG_Test; feat_test];
        N_Test = [N_Test j];
        
    end
    
    for k = 1:n2
        B = reshape(Train_In(:,k), [28 28]);
        [feat_train,~] = extractHOGFeatures(B,'CellSize',dim);
        HOG_Train = [HOG_Train; feat_train];
        N_Train = [N_Train j];
    end
    
end



% Prepare HOG Features and Labels
HOG_Train_0 = HOG_Train(N_Train==0,:);
HOG_Train_1 = HOG_Train(N_Train==1,:);
HOG_Train_2 = HOG_Train(N_Train==2,:);
HOG_Train_3 = HOG_Train(N_Train==3,:);
HOG_Train_4 = HOG_Train(N_Train==4,:);
HOG_Train_5 = HOG_Train(N_Train==5,:);
HOG_Train_6 = HOG_Train(N_Train==6,:);
HOG_Train_7 = HOG_Train(N_Train==7,:);
HOG_Train_8 = HOG_Train(N_Train==8,:);
HOG_Train_9 = HOG_Train(N_Train==9,:);

HOG_Test_0 = HOG_Test(N_Test==0,:);
HOG_Test_1 = HOG_Test(N_Test==1,:);
HOG_Test_2 = HOG_Test(N_Test==2,:);
HOG_Test_3 = HOG_Test(N_Test==3,:);
HOG_Test_4 = HOG_Test(N_Test==4,:);
HOG_Test_5 = HOG_Test(N_Test==5,:);
HOG_Test_6 = HOG_Test(N_Test==6,:);
HOG_Test_7 = HOG_Test(N_Test==7,:);
HOG_Test_8 = HOG_Test(N_Test==8,:);
HOG_Test_9 = HOG_Test(N_Test==9,:);


HOG_TRAIN = [HOG_Train_0; HOG_Train_1; HOG_Train_2; HOG_Train_3; HOG_Train_4;...
    HOG_Train_5; HOG_Train_6; HOG_Train_7; HOG_Train_8; HOG_Train_9];
HOG_TEST  = [HOG_Test_0; HOG_Test_1; HOG_Test_2; HOG_Test_3; HOG_Test_4;...
    HOG_Test_5; HOG_Test_6; HOG_Test_7; HOG_Test_8; HOG_Test_9];


LABEL_TRAIN = N_Train';
LABEL_TEST  = N_Test';

%% STEP 3: Set up and run SVM

% one-vs-one SVM Linear kernel
SVMModel = fitcecoc(HOG_TRAIN, LABEL_TRAIN);

% Check SVM on testing sets
[Predicted_Digits , scores] = predict(SVMModel,HOG_TEST);

% Plot the confusion matrix 
confusionchart(LABEL_TEST',Predicted_Digits');

