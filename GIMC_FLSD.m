function [Pc] = GIMC_FLSD(X,W,G,opts)
% The code is written by Jie Wen, 
% if you have any problems, please don't hesitate to contact me via: jiewen_pr@126.com 
% If you find the code is useful, please cite the following references:
% Jie Wen, Zheng Zhang, Zhao Zhang, Lunke Fei, Meng Wang,
% Generalized Incomplete Multi-view Clustering With Flexible Locality
% Structure Diffusion[J], IEEE Transactions on Cybenetics, 2020.
% Jie Wen , Zheng Zhang, Yong Xu, Zuofeng Zhong, 
% Incomplete Multi-view Clustering via Graph Regularized Matrix Factorization [C], 
% European Conference on Computer Vision Workshop on Compact and Efficient Feature Representation and Learning in Computer Vision, 2018.
% homepage: https://sites.google.com/view/jerry-wen-hit/publications
num_view   = opts.num_view;
num_sample = size(G{1},2);
lambda1    = opts.lambda1;
lambda2    = opts.lambda2;
r          = opts.r;
nnClass    = opts.nnClass;
max_iter   = opts.max_iter;
alpha   = ones(1,num_view)/num_view;
alpha_r = alpha.^r;
for k = 1:num_view
    D{k} = diag(sum(W{k}));
    rand('seed',k*666);
    U{k} = rand(nnClass,size(X{k},2));
    if nnClass < size(X{k},2)
        U{k} = orth(U{k}')';
    else
        U{k} = orth(U{k});
    end    
    P{k} = X{k}*U{k}';
end
rand('seed',220);
Pc = rand(num_sample,nnClass);
for iter = 1:max_iter
    linshi_Pc = 0;
    for k = 1:num_view
        % --------------- Uk --------------- %
        [Gs,~,Vs] = svd(X{k}'*W{k}*P{k},'econ');
        Gs(isnan(Gs)) = 0;
        Vs(isnan(Vs)) = 0;
        U{k} = Vs*Gs';  
        clear Vs Gs
        % -------------- Pk -------------- % 
        M = D{k}+lambda1*eye(size(D{k},1));
        A = U{k}*X{k}'*W{k}+lambda1*Pc'*G{k}';
        C = (A*diag(1./(diag(M))))';
        linshi_P = [];
        for ip = 1:size(P{k},1)            
            temp1 = 0.5*lambda2/M(ip,ip);
            temp2 = C(ip,:);
            linshi_P(ip,:) = max(0,temp2-temp1) + min(0,temp2+temp1);
        end
        P{k} = linshi_P;                
    end
    % -----------------  Pc --------------- %
    linshi_Pc1 = 0;
    linshi_Pc2 = 0;
    for iv = 1:num_view
        linshi_Pc1 = linshi_Pc1+alpha_r(iv)*G{iv}'*G{iv};
        linshi_Pc2 = linshi_Pc2+alpha_r(iv)*G{iv}'*P{iv};
    end
    Pc = diag(1./diag(linshi_Pc1))*linshi_Pc2;
    % ------------- alpha -------------- %
    NormX = 0;
    for k = 1:length(X)
       % ------- obj reconstructed error ------------ %
       Rec_error(k) = trace(X{k}'*D{k}*X{k})+trace(P{k}'*D{k}*P{k})-2*trace(X{k}'*W{k}*P{k}*U{k})+lambda1*norm(P{k}-G{k}*Pc,'fro')^2+lambda2*sum(abs(P{k}(:)));
       NormX = NormX + norm(X{k},'fro')^2;
    end
    H = bsxfun(@power,Rec_error, 1/(1-r));     % h = h.^(1/(1-r));
    alpha = bsxfun(@rdivide,H,sum(H)); % alpha = H./sum(H);
    alpha_r = alpha.^r;
    % ------------- obj ------------- %
    obj(iter) = alpha_r*Rec_error';
    obj2(iter) = (alpha_r*Rec_error')/NormX;
    if iter > 10 && abs(obj2(iter)-obj2(iter-1))<1e-7
        iter
        break;
    end
end
end