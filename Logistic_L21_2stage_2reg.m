%% FUNCTION Logistic_TGL
%  L21 Joint Feature Learning with Logistic Loss.
%
%% OBJECTIVE
%   argmin_{W,C} { sum_i^t (- sum(log (1./ (1+ exp(-X{i}*W(:, i) - Y{i} .* C(i)))))/length(Y{i}))
%            + opts.rho_L2 * \|W\|_2^2 + lambda2 * \|W\|_{2,1} }
%
%% INPUT
%   X: {n * d} * t - input matrix
%   Y: {n * 1} * t - output matrix
%   lambda2: L2,1-norm group Lasso parameter.
%   OPTIONAL
%      opts.rho_L2: L2-norm parameter (default = 0).
%
%% OUTPUT
%   W: model: d * t
%   C: model: 1 * t
%   funcVal: function value vector.
%
%% LICENSE
%   This program is free software: you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation, either version 3 of the License, or
%   (at your option) any later version.
%
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
%
%   You should have received a copy of the GNU General Public License
%   along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
%   Copyright (C) 2011 - 2012 Jiayu Zhou, Jun Liu and Jieping Ye
%
%   You are suggested to first read the Manual.
%   For any problem, please contact with Jiayu Zhou via jiayu.zhou@asu.edu
%
%   Last modified on June 3, 2012.
%
%% RELATED PAPERS
%
%   [1] Evgeniou, A. and Pontil, M. Multi-task feature learning, NIPS 2007.
%   [2] Liu, J. and Ye, J. Efficient L1/Lq Norm Regularization, Technical
%       Report, 2010.
%
%% RELATED FUNCTIONS
%  Least_L21, init_opts

%% Code starts here
function [W, C, funcVal, niter] = Logistic_L21_2stage_2reg(X, Y, lambda2, lambda1, RepIndex,group_arr,groups,opts)

if nargin <3
    error('\n Inputs: X, Y, lambda2, should be specified!\n');
end

for ss = 1:length(X)
    X{ss} = X{ss}';
end

% % initialize options.
% opts=init_opts(opts);

if isfield(opts, 'rho_L2')
    rho_L2 = opts.rho_L2;
else
    rho_L2 = 0;
end

task_num  = length(X);
dimension = size(RepIndex,2);
funcVal = [];

%initialize a starting point
C0_prep = zeros(1, task_num);
for t_idx = 1: task_num
    m1 = nnz(Y{t_idx} == 1);
    m2 = nnz(Y{t_idx} == -1);
    if ( m1==0 || m2==0 )
        C0_prep(t_idx) = 0;
    else
        C0_prep(t_idx) = log(m1/m2);
    end
end

if opts.init==2
    W0 = zeros(dimension, task_num);
    C0 = zeros(1, task_num);
elseif opts.init== 0
    W0 = randn(dimension, task_num);
    C0 = C0_prep;
else
    if isfield(opts,'W0')
        W0=opts.W0;
        if (nnz(size(W0)-[dimension, task_num]))
            error('\n Check the input .W0');
        end
    else
        W0 = zeros(dimension, task_num);
    end
    if isfield(opts,'C0')
        C0=opts.C0;
    else
        C0=C0_prep;
    end
end



bFlag=0; % this flag tests whether the gradient step only changes a little


Wz= W0;
Cz= C0;
Wz_old = W0;
Cz_old = C0;

t = 1;
t_old = 0;
iter = 0;
gamma = 1;
gamma_inc = 2;
opts.maxIter = 1000;
opts.tol = 1e-8;
while iter < opts.maxIter
    alpha = (t_old - 1) /t;
    
    Ws = (1 + alpha) * Wz - alpha * Wz_old;
    Cs = (1 + alpha) * Cz - alpha * Cz_old;
    
    % compute function value and gradients of the search point
    [gWs, gCs, Fs ]  = gradVal_eval(Ws, Cs);
    
%     Fs
    
    % the Armijo Goldstein line search scheme
    while true
        Wzp = soslasso_shrink_logistic(Ws - gWs/gamma,group_arr,groups,lambda2/gamma,lambda1/gamma);

%         Wzp = FGLasso_projection_2stage(Ws - gWs/gamma, lambda2/gamma, group_arr);
        Czp = Cs - gCs/gamma;
        Fzp = funVal_eval  (Wzp, Czp);
        
        delta_Wzp = Wzp - Ws;
        delta_Czp = Czp - Cs;
        nrm_delta_Wzp = norm(delta_Wzp, 'fro')^2;
        nrm_delta_Czp = norm(delta_Czp, 'fro')^2;
        r_sum = (nrm_delta_Wzp+nrm_delta_Czp)/2;
        
        Fzp_gamma = Fs + sum(sum(delta_Wzp.* gWs))...
            + sum(sum(delta_Czp .* gCs))...
            + gamma/2 * nrm_delta_Wzp ...
            + gamma/2 * nrm_delta_Czp;
        
        if (r_sum <=1e-20)
            bFlag=1; % this shows that, the gradient step makes little improvement
            break;
        end
        
        if (Fzp <= Fzp_gamma)
            break;
        else
            gamma = gamma * gamma_inc;
        end
    end
    
    Wz_old = Wz;
    Cz_old = Cz;
    Wz = Wzp;
    Cz = Czp;
    
    funcVal = cat(1, funcVal, Fzp + nonsmooth_eval(Wz, lambda2));
    
    if (bFlag)
        % fprintf('\n The program terminates as the gradient step changes the solution very small.');
        break;
    end
    
    % test stop condition.
    switch(opts.tFlag)
        case 0
            if iter>=2
                if (abs( funcVal(end) - funcVal(end-1) ) <= opts.tol)
                    break;
                end
            end
        case 1
            if iter>=2
                if (abs( funcVal(end) - funcVal(end-1) ) <=...
                        opts.tol* funcVal(end-1))
                    break;
                end
            end
        case 2
            if ( funcVal(end)<= opts.tol)
                break;
            end
        case 3
            if iter>=opts.maxIter
                break;
            end
    end
    
    iter = iter + 1;
    t_old = t;
    t = 0.5 * (1 + (1+ 4 * t^2)^0.5);
    
end
niter = iter;
W = Wzp;
C = Czp;

% private functions

    function [grad_W, grad_C, funcVal] = gradVal_eval(W, C)
        grad_W = zeros(dimension, task_num);
        grad_C = zeros(1, task_num);
        lossValVect = zeros (1 , task_num);
        if opts.pFlag
            parfor i = 1:task_num
                [ grad_W(:, i), grad_C(:, i), lossValVect(:, i)] = unit_grad_eval( W(:, i), C(i), X{i}(RepIndex,:), Y{i});
            end
        else
            for i = 1:task_num
                [ grad_W(:, i), grad_C(:, i), lossValVect(:, i)] = unit_grad_eval( W(:, i), C(i), X{i}(RepIndex,:), Y{i});
            end
        end
        grad_W = grad_W + rho_L2 * 2 * W;
        % here when computing function value we do not include
        % l1 norm.
        funcVal = sum(lossValVect) + rho_L2 * norm(W,'fro')^2;
    end

    function [funcVal] = funVal_eval (W, C)
        funcVal = 0;
        if opts.pFlag
            parfor i = 1: task_num
                funcVal = funcVal + unit_funcVal_eval( W(:, i), C(i), X{i}(RepIndex,:), Y{i});
            end
        else
            for i = 1: task_num
                funcVal = funcVal + unit_funcVal_eval( W(:, i), C(i), X{i}(RepIndex,:), Y{i});
            end
        end
        % here when computing function value we do not include
        % l1 norm.
        funcVal = funcVal + rho_L2 * norm(W,'fro')^2;
    end

    function [non_smooth_value] = nonsmooth_eval(W, rho_1)
        non_smooth_value = 0;
        for i = 1 : size(W, 1)
            w = W(i, :);
            non_smooth_value = non_smooth_value ...
                + rho_1 * norm(w, 2);
        end
    end
end

% function seg_ind_set = gen_seg_idx(s_idx, seg_len, total_len)
% %generate segmentation indices.
% seg_ind_set = (s_idx-1) * seg_len + 1  : min(s_idx* seg_len, total_len);
% end

function [ grad_w, grad_c, funcVal ] = unit_grad_eval( w, c, x, y)
%gradient and logistic evaluation for each task
m = length(y);
weight = ones(m, 1)/m;
weighty = weight.* y;
aa = -y.*(x'*w + c);
bb = max( aa, 0);
funcVal = weight'* ( log( exp(-bb) +  exp(aa-bb) ) + bb );
pp = 1./ (1+exp(aa));
b = -weighty.*(1-pp);
grad_c = sum(b);
grad_w = x * b;
end

function [ funcVal ] = unit_funcVal_eval( w, c, x, y)
%function value evaluation for each task
m = length(y);
weight = ones(m, 1)/m;
aa = -y.*(x'*w + c);
bb = max( aa, 0);
funcVal = weight'* ( log( exp(-bb) +  exp(aa-bb) ) + bb );
end
