function JLP_classify_lasso_condor()
    load('jlp_metadata.mat');
    load('jlp_params.mat');
    ss = load('subind.txt');
    f = load('WhichFinal.txt');
    
    LAMSET = params.LAMSET;
    nlam = length(LAMSET);
    TargetCategory = params.TargetCategory;
    Target = sprintf('True%s',TargetCategory);
    nsubj = length(metadata); % 1xnsubj structured array.
    
    %% load and z-score the data
    % NB. There are no outliers in this dataset after z-scoring.
    load(sprintf('jlp%02d.mat',ss),'X');
    X = bsxfun(@minus,X,mean(X)); %#ok<NODEF>
    X = bsxfun(@rdivide,X,std(X));
	
    %% Set up the CV filters
    CV = metadata(ss).CVBLOCKS;
    FINAL = CV(:,f);
    CV(:,f) = [];
    TRAIN = bsxfun(@and,~CV,~FINAL);
    ncv = size(CV,2);
            
    %% Define the appropriate y
    y = metadata(ss).(Target);
            
    [Bb,B] = deal(zeros(size(X,2),ncv*nlam));
    [Cb,C] = deal(zeros(1,ncv*nlam));
    
	for ll = 1:nlam
        LAMBDA = LAMSET(ll);
        for cc = 1:ncv
            %% Fit lasso model to the training set.
            a = ((ll-1)*ncv) + cc;
            train = TRAIN(:,cc);
            [Bb(:,a), Cb(1,a)] = Logistic_Lasso({X(train,:)}, ...
                                                  {y(train)}, ...
                                                  LAMBDA);

            %% Debias the weights
            v = abs(Bb(:,a))>0;
            b = glmfit(X(train,v),y(train)>0,'binomial'); % b{ss}(1) is the intercept.
            B(v,a) = b(2:end);
            C(1,a) = b(1);
        end
	end
	
    %% Compute "activation level" for each TR
	% Possitive activation levels: prediction of target category
    Yhb = bsxfun(@plus, X * Bb, Cb);
	Yh = bsxfun(@plus, X * B, C);
            
    %% Compute d-prime values: measure of selectivity/sensitivity
    % Evaluate on the training and test sets, separately.
    dpb  = reshape(dprimeCV(y,Yhb,CV   ),ncv,nlam); 
    dptb = reshape(dprimeCV(y,Yhb,TRAIN),ncv,nlam);
    dp   = reshape(dprimeCV(y,Yh, CV   ),ncv,nlam); 
    dpt  = reshape(dprimeCV(y,Yh, TRAIN),ncv,nlam);
    [~,BestLamdab] = max(mean(dpb)); %#ok<*NASGU>
    [~,BestLamda ] = max(mean(dp ));

    save(sprintf('lassoresult_cv%02d_ss%02d_%s.mat',f,ss,date), ...
            'Bb','Cb','B','C', ...
            'Yhb','Yh', ...
            'dpb','dptb','dp','dpt',...
            'metadata','params');
end
%         Results.BestLambda.debaised = BestLambda;
%         Results.BestLambda.baised = BestLambdab;
%         Results.DPrime.train.debaised = dpt;
%         Results.DPrime.train.baised = dptb;
%         Results.DPrime.test.debaised = dp;
%         Results.DPrime.test.baised = dpb;
%         Results.Yhat.debaised = Yh;
%         Results.Yhat.baised = Yhb;
%         Results.Betahat.debaised = B;
%         Results.Betahat.baised = Bb;
%         Results.C.debaised = C;
%         Results.C.baised = Cb;
        
        
    
    
