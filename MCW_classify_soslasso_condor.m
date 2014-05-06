function MCW_classify_soslasso_condor()
% Send the data in native space, generate on arrival.
% EVERY CV IS A SEPARATE JOB
    %% Load metadata and set up SUBJECTS
	load('mcw_soslasso_metadata.mat');
    SUBJECTS = metadata.subjects;
	nsub = length(SUBJECTS);
    
    %% Set up the CV filters
    f = load('WhichFinal.txt');
    c = load('WhichCV.txt');
	CV = metadata.CVBLOCKS(:,c);
	FINAL = metadata.CVBLOCKS(:,f);
	TRAIN = ~CV & ~FINAL;
    
    %% Loat params
	load('mcw_soslasso_params.mat');

	%% Set up lambda
	l = load('WhichLambda.txt');
	LAMBDA = params.lamset(l);

	%% Set up mu
	m = load('WhichMu.txt');
	MU = params.muset(m);
    
    %% Load opts
    load('mcw_soslasso_opts.mat');
    
	%% Load and z-score the data, while defining GroupInfo
	[X,GroupInfo] = mcw_prep_data_multiples(metadata,params);
	if params.MeanCenter == true && params.NormVariance == true
		for ss = 1:nsub
% 			X{ss} = zscore(X{ss});
		end
		fprintf('DATA CENTERED\n')
        fprintf('DATA SCALED\n')
	end

	%% Initialize output
    % B's and C's do not need initialization.
	[Yhb,Yh] = deal(zeros(size(X{1},1),nsub));
			
    %% Set up X and Y for soslasso
    [trainX, trainY] = deal(cell(nsub,1));
    for ss = 1:nsub
        trainX{ss} = X{ss}(TRAIN,:);
        trainY{ss} = ones(size(metadata.TrueAnimals(TRAIN)));
        trainY{ss}(metadata.TrueAnimals(TRAIN)) = -1;
    end

    %% Fit lasso model to the training set.
     [Bb,Cb,~,B,C] = overlap_2stage_2reg(1,trainY,trainX,GroupInfo,LAMBDA,MU,opts);
% ++++++++++++++++++++++++++++++++++++++++++
%            For debugging only
% ++++++++++++++++++++++++++++++++++++++++++
%	Bb = randn(size(X{1},2),nsub);
%	B = randn(size(X{1},2),nsub);
%	Cb = randn(1,nsub);
%	C = randn(1,nsub);
	
	%% Compute "activation level" for each TR
	% Possitive activation levels: prediction of target category
	for ss = 1:nsub
		Yhb(:,ss) = bsxfun(@plus, X{ss} * Bb(:,ss), Cb(1,ss));
		Yh(:,ss) = bsxfun(@plus, X{ss} * B(:,ss), C(1,ss));
	end
		   
	%% Compute d-prime values: measure of selectivity/sensitivity
	% Evaluate on the training and test sets, separately.
    y = metadata.TrueAnimals;
	dpb  = dprimeCV(y,Yhb,CV);  %#ok<*NASGU>
	dptb = dprimeCV(y,Yhb,TRAIN);
	dp   = dprimeCV(y,Yh, CV); 
	dpt  = dprimeCV(y,Yh, TRAIN);

    save(sprintf('soslassoresult_%02d%02d%02d%02d_%s.mat',f,c,m,l,datestr(date,'ddmmmyy')), ...
			'Bb','Cb','B','C', ...
			'Yhb','Yh', ...
			'dpb','dptb','dp','dpt');
%			'metadata','params','opts');
end
%		 Results.BestLambda.debaised = BestLambda;
%		 Results.BestLambda.baised = BestLambdab;
%		 Results.DPrime.train.debaised = dpt;
%		 Results.DPrime.train.baised = dptb;
%		 Results.DPrime.test.debaised = dp;
%		 Results.DPrime.test.baised = dpb;
%		 Results.Yhat.debaised = Yh;
%		 Results.Yhat.baised = Yhb;
%		 Results.Betahat.debaised = B;
%		 Results.Betahat.baised = Bb;
%		 Results.C.debaised = C;
%		 Results.C.baised = Cb;
