function [Xhat,C,niter,Wndb,Cndb,W] = overlap_2stage_2reg(loss,Y,X,GroupInfo,lambda,gamma,opts)

% function to perform the overlapping SGL optimization, with both LS and LOGIT loss
% INPUTS
% loss    = type of loss function. 0 = least squares, 1 = logistic
% Y       = T X 1 cell array of observations . T  =number of tasks
% X       = T X 1 cell array of data
% Xo      = T X 1 cell array of replicated data matrices
% G       = cell array of groups
% group_arr = output from replication step
% lambda  = regularizer
% OUTPUTS
% Xhat    = debiased output
% C       = the bias term in the output
% 
% CODE REQUIRES MALSAR PACKAGE
%
% Nikhil Rao
% 3/17/13
G = GroupInfo.G;
RepIndex = GroupInfo.RepIndex;
group_arr = GroupInfo.group_arr;
groups = GroupInfo.groups;

    if loss == 1
        [W, C, ~,niter] = Logistic_L21_2stage_2reg(X, Y, lambda, gamma, RepIndex, group_arr, groups, opts);
    elseif loss == 0
        [W, ~] = Least_L21_2stage(Xo, Y, lambda, group_arr);
        C = [];
    else
        error('loss has to be 0 or 1 \n');
    end
    
    % W is the output matrix
    %we now need to combine overlapping groups
    
    for iii = 1:length(G)
        temp = G{iii};
        n(iii) = max(temp);
    end
    n = max(n);
    T = length(Y);
    Xhat = zeros(n,T);
    % identify whether a dummy variable exists and chuck it
    dummy = max(max(group_arr));
    mask = (group_arr == dummy);
    isdummy = 0;
    if sum(sum(mask))>1
        isdummy = 1;
    end
    for ii = 1:length(G)
        t = G{ii};
        s = group_arr(ii,:);
        if isdummy == 1
%            dummyind = find();
           s(s == dummy) = [];
        end
        Xhat(t,:) = Xhat(t,:) + W(s,:);
    end
  
    % The solution before debiasing.
    Wndb = Xhat;
    Cndb = C;

    %debias the solution using the X and Y and Xhat
    % Xnew = cell(T,1);
    inds = cell(T,1);
    temp = Xhat;
    Xhat = zeros(n,T);
    C = zeros(1,T);
    % totinds =[];
    for ii = 1:T
        z = temp(:,ii) ~= 0;
        if any(z);
            inds{ii} = find(z);
            Xdeb = X{ii}(:,z);
            Bhat = glmfit(Xdeb, Y{ii}>0, 'binomial');
            Xhat(z,ii) = Bhat(2:end);
            C(ii) = Bhat(1);
        end
    end
end
