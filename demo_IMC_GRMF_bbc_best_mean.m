% This is a demo, run in matlab 2015b
% For the other datasets or other incomplete group 'f', 
% turn parameters lambda1,lambda2,r,k for the best result.

% The code is written by Jie Wen, 
% if you have any problems, please don't hesitate to contact me via: jiewen_pr@126.com 
% If you find the code is useful, please cite the following references:
% Jie Wen, Zheng Zhang, Zhao Zhang, Lunke Fei, Meng Wang,
% Generalized Incomplete Multi-view Clustering With Flexible Locality
% Structure Diffusion[J], IEEE Transactions on Cybenetics, 2020.
% Jie Wen , Zheng Zhang, Yong Xu, Zuofeng Zhong, 
% Incomplete Multi-view Clustering via Graph Regularized Matrix Factorization [C], 
% European Conference on Computer Vision Workshops on Compact and Efficient Feature Representation and Learning in Computer Vision, 2018.
% homepage: https://sites.google.com/view/jerry-wen-hit/publications
clear;
clc

Dataname = 'bbcsport4vbigRnSp';

% percentDel = 0.1
% lambda1 = 100;
% lambda2 = 1;
% r = 7;
% k = 15;
% f = 3;

% percentDel = 0.3
% lambda1 = 10;
% lambda2 = 0.00001;
% r = 3;
% k = 7;
% f = 3;

percentDel = 0.5
lambda1 = 10;
lambda2 = 0.00001;
r = 3;
k = 7;
f = 2;


Datafold = [Dataname,'_percentDel_',num2str(percentDel),'.mat'];
load(Dataname);
load(Datafold);

num_view = length(X);
numClust = length(unique(truth));
numInst  = length(truth); 


ind_folds = folds{f};
truthF = truth;
load(Dataname);
for iv = 1:length(X)
    X1 = X{iv}';
    X1 = NormalizeFea(X1,1);
    ind_0 = find(ind_folds(:,iv) == 0);
    X1(ind_0,:) = [];       % 去掉 缺失样本
    Y{iv} = X1;
    % ------------- 构造缺失视角的索引矩阵 ----------- %
    W1 = eye(numInst);
    W1(ind_0,:) = [];
    G{iv} = W1;                            
end
clear X X1 W1 ind_0
X = Y;
clear Y   

% ---------- nearest neighbor graph of feature construction ------------ %
for iv = 1:length(X)
    options = [];
    options.NeighborMode = 'KNN';
    options.k = k;
    options.WeightMode = 'Binary';      % Binary  HeatKernel
    Z1 = full(constructW(X{iv},options));
    W{iv} = (Z1+Z1')/2;
    clear Z1;
end

opts.lambda1 = lambda1;
opts.lambda2 = lambda2;
opts.r       = r;
opts.nnClass = numClust;
opts.num_view= num_view;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
opts.max_iter= 120;
[Pc] = GIMC_FLSD(X,W,G,opts);
new_F    = Pc;
norm_mat = repmat(sqrt(sum(new_F.*new_F,2)),1,size(new_F,2));
%%avoid divide by zero
for i = 1:size(norm_mat,1)
    if (norm_mat(i,1)==0)
        norm_mat(i,:) = 1;
    end
end
new_F = new_F./norm_mat; 
repeat = 5;
for iter_c = 1:repeat
    pre_labels    = kmeans(new_F,numClust,'emptyaction','singleton','replicates',20,'display','off');
    result_LatLRR = ClusteringMeasure(truthF, pre_labels);       
    AC(iter_c)    = result_LatLRR(1)*100;
    MIhat(iter_c) = result_LatLRR(2)*100;
    Purity(iter_c)= result_LatLRR(3)*100;
end
mean_ACC = mean(AC)
mean_NMI = mean(MIhat)
mean_PUR = mean(Purity)







