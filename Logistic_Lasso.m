%% FUNCTION Logistic_Lasso
%   Sparse Structure-Regularized Learning with Logistic Loss.
%
%% OBJECTIVE
%   argmin_{W,C} { sum_i^t (- sum(log (1./ (1+ exp(-X{i}*W(:, i) - Y{i} .* C(i)))))/length(Y{i}))
%            + rho1 * \|W\|_1 + opts.rho_L2 * \|W\|_F^2}
%
%% INPUT
%   X: {n * d} * t - input matrix
%   Y: {n * 1} * t - output matrix
%   rho1: sprasity controlling parameter
%   opts.rho_L2: L2-norm regularization parameter
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
%   Copyright (C) 2011 - 2012 Jiayu Zhou and Jieping Ye
%
%   You are suggested to first read the Manual.
%   For any problem, please contact with Jiayu Zhou via jiayu.zhou@asu.edu
%
%   Last modified on June 3, 2012.
%
%% Related papers
%
%   [1] Tibshirani, J. Regression shrinkage and selection via
%   the Lasso, Journal of the Royal Statistical Society. Series B 1996
%
%% Related functions
%   Least_Lasso, init_opts

%% Code starts here
function [W, C, funcVal] = Logistic_Lasso(X, Y, rho1, opts)

if nargin <3
    error('Inputs: X, Y, and rho1 should be specified!');
end
X = multi_transpose(X);

if nargin <4
    opts = [];
end

% initialize options.
opts=init_opts(opts);

if isfield(opts, 'rho_L2')
    rho_L2 = opts.rho_L2;
else
    rho_L2 = 0;
end

task_num  = length (X);
dimension = size(X{1}, 1);
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
    %C0 = zeros(1, task_num);
    C0 = C0_prep;
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

while iter < opts.maxIter
    alpha = (t_old - 1) /t;
    
    Ws = (1 + alpha) * Wz - alpha * Wz_old;
    Cs = (1 + alpha) * Cz - alpha * Cz_old;
    
    %Ws = Wz;
    %Cs = Cz;
    
    % compute function value and gradients of the search point
    [gWs, gCs, Fs ]  = gradVal_eval(Ws, Cs);
    
    % the Armijo Goldstein line search scheme
    while true
        [Wzp l1c_wzp] = l1_projection(Ws - gWs/gamma, 2 * rho1 / gamma);
        Czp = Cs - gCs/gamma;
        Fzp = funVal_eval  (Wzp, Czp);
        
        delta_Wzp = Wzp - Ws;
        delta_Czp = Czp - Cs;
        nrm_delta_Wzp = norm(delta_Wzp, 'fro')^2;
        nrm_delta_Czp = norm(delta_Czp, 'fro')^2;
        r_sum = (nrm_delta_Wzp+nrm_delta_Czp)/2;
        
        %         Fzp_gamma = Fs + trace(delta_Wzp' * gWs)...
        %             + trace(delta_Czp' * gCs)...
        %             + gamma/2 * nrm_delta_Wzp ...
        %             + gamma/2 * nrm_delta_Czp;
        
        Fzp_gamma = Fs + sum(sum(delta_Wzp .* gWs))...
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
    
    funcVal = cat(1, funcVal, Fzp + rho1 * l1c_wzp);
    
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

W = Wzp;
C = Czp;

% private functions

    function [z l1_comp_val] = l1_projection (v, beta)
        % this projection calculates
        % argmin_z = \|z-v\|_2^2 + beta \|z\|_1
        % z: solution
        % l1_comp_val: value of l1 component (\|z\|_1)
        
        z = zeros(size(v));
        vp = v - beta/2;
        z (v> beta/2)  = vp(v> beta/2);
        vn = v + beta/2;
        z (v< -beta/2) = vn(v< -beta/2);
        
        l1_comp_val = sum(sum(abs(z)));
    end


    function [grad_W, grad_C, funcVal] = gradVal_eval(W, C)
        grad_W = zeros(dimension, task_num);
        grad_C = zeros(1, task_num);
        lossValVect = zeros (1 , task_num);
        if opts.pFlag
            parfor i = 1:task_num
                [ grad_W(:, i), grad_C(:, i), lossValVect(:, i)] = unit_grad_eval( W(:, i), C(i), X{i}, Y{i});
            end
        else
            for i = 1:task_num
                [ grad_W(:, i), grad_C(:, i), lossValVect(:, i)] = unit_grad_eval( W(:, i), C(i), X{i}, Y{i});
            end
        end
        grad_W = grad_W + rho_L2 * 2 * W;
        % here when computing function value we do not include
        % l1 norm.
        funcVal = sum(lossValVect) + rho_L2 * norm(W, 'fro')^2;
    end



    function [funcVal] = funVal_eval (W, C)
        funcVal = 0;
        if opts.pFlag
            parfor i = 1: task_num
                funcVal = funcVal + unit_funcVal_eval( W(:, i), C(i), X{i}, Y{i});
            end
        else
            for i = 1: task_num
                funcVal = funcVal + unit_funcVal_eval( W(:, i), C(i), X{i}, Y{i});
            end
        end
        % here when computing function value we do not include
        % l1 norm.
        funcVal = funcVal + rho_L2 * norm(W, 'fro')^2;
    end


end


function [ grad_w, grad_c, funcVal ] = unit_grad_eval( w, c, x, y)
%gradient and logistic evaluation for each task
m = length(y);
weight = ones(m, 1)/m;
weighty = weight.* y;
aa = -y.*(x'*w + c);
bb = max( aa, 0);
%funcVal = weight' * log(exp(-bb) + exp(aa-bb) + bb); %bug!!!
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
%funcVal = weight' * log(exp(-bb) + exp(aa-bb) + bb); % BUG!!!!
funcVal = weight'* ( log( exp(-bb) +  exp(aa-bb) ) + bb );
end