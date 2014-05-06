%%% shrinkage on product norm

function y = soslasso_shrink_logistic(X,G,groups,lam,gam)

lam1 = gam*lam;
% step 1: perform soft thresholding
X_soft = sign(X).*max(abs(X) - lam1,0);

%step 2: perform group soft thresholding
M = size(G,1); % number of groups
X_soft = [X_soft; zeros(1,size(X_soft,2))]; % for the dummy
Xtemp = sum(X_soft.^2,2); %xtemp is now a vector
Xtemp = sum(Xtemp(G),2);
Xtemp = sqrt(Xtemp);
Xtemp = max(Xtemp - lam,0); % this is the multiplying factor
Xtemp = Xtemp./(Xtemp + lam);
if (size(Xtemp,1)~=M)
    error('something weird is happening with the group shrinkage \n');
end
Xtemp = Xtemp(groups);
Xtemp = repmat(Xtemp,1,size(X,2));
y = X_soft(1:end-1,:).*Xtemp;

end