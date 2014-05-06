function [fitObj,fitObjRaw] = jlp_cvsoslasso_condor(X,Y,CV,LAMSET,MU,GroupInfo)
% X : Nx1 Cell array, where N is the number of subjects.
% Y : Nx1 Cell array, where N is the number of subjects.
    
    nsubj = length(X);
    nlam = length(LAMSET);
    ncv = size(CV{1},2);
            
    [Bb,B] = deal(zeros(size(X{1},2),ncv*nlam*nsubj));
    [Cb,C] = deal(zeros(1,ncv*nlam*nsubj));
    
	[Xtrain, Ytrain] = deal(cell(nsubj,1));
	

    %% Fit lasso model to the training set.
	if exist(fullfile(pwd, 'CHECKPOINT.mat'), 'file') == 2
		load('CHECKPOINT.mat');
		% 'Bb','Cb','B','C','cc','ll','opts'
		cv_start = cc;
		lam_start = ll;
		fprintf('Resuming from cv: %d, lam: %d.\n',cc,ll);
	else
    	opts = init_opts([]);
		cv_start = 1;
		lam_start = 1;
	end
	for cc = cv_start:ncv
		for ss = 1:nsubj
       		Xtrain{ss} = X{ss}(~CV{ss}(:,cc),:);
       		y = double(Y{ss}(~CV{ss}(:,cc)));
			y(y==0) = -1;
			Ytrain{ss} = y;
    	end
		for ll = lam_start:nlam
			fprintf('cv: %d, lam: %d\n', cc,ll);
        	LAMBDA = LAMSET(ll);
            %% Fit lasso model to the training set.
            a = sub2ind([nsubj,ncv,nlam],1,cc,ll);
            b = sub2ind([nsubj,ncv,nlam],nsubj,cc,ll);

			% For Debugging---START
			%disp([a,b,b-a+1]);
			%[Bb(1,a:b),Cb(a:b),B(1,a:b),C(a:b)];
			%c=((a-1)*27626);
			%Bb(c+randperm(27626*nsubj,2000)) = randn(2000,1); 
			%B(c+randperm(27626*nsubj,2000)) = randn(2000,1); 
			% For Debugging---END
		
			[Bb(:,a:b),Cb(a:b),~,B(:,a:b),C(a:b),W] = overlap_2stage_2reg(1,Ytrain,Xtrain,GroupInfo,LAMBDA,MU,opts);
			opts.W0=W; % Warm start
			opts.C0=C(a:b); % Warm start
			Bb = sparse(Bb);
			B = sparse(B);
			save('CHECKPOINT.mat','Bb','Cb','B','C','cc','ll','opts');
        end
		opts = rmfield(opts,'W0');
		opts = rmfield(opts,'C0');
		lam_start = 1;
	end
	
    %% Evaluate and store debiased solutions 
	Betas = cell(size(X));
	a0 = cell(size(X));
	Yh = cell(size(X));
    dp = cell(size(X));
	dpt = cell(size(X));
	SubjectLabels = mod(0:(b-1),nsubj)+1;
	for ss = 1:nsubj
		z = SubjectLabels == ss;
		Betas{ss} = Bb(:,z);
		a0{ss} = Cb(z);
		Yh{ss} = bsxfun(@plus, X{ss} * Betas{ss}, a0{ss});
    	dp{ss} = reshape(dprimeCV(Y{ss}>0,Yh{ss}>0, CV{ss}), ncv, nlam); 
    	dpt{ss} = reshape(dprimeCV(Y{ss},Yh{ss}, ~CV{ss}), ncv, nlam);
	end

	[~,bestLambdaInd] = max(mean(cell2mat(dp)));
	bestLambda = LAMSET(bestLambdaInd);

	fitObj.Betas = Betas;
	fitObj.a0 = a0;
	fitObj.lambda = LAMSET;
	fitObj.mu = MU;
	fitObj.bestLambda = bestLambda;
	fitObj.bestLambdaInd = bestLambdaInd;
	fitObj.dp = dp;
	fitObj.dpt = dpt;
	fitObj.Yh = Yh;
	fitObj.Y = Y;
	fitObj.CV = CV;
	fitObj.method='soslasso';

    %% Evaluate and store raw solutions 
	Betas = cell(size(X));
	a0 = cell(size(X));
	Yh = cell(size(X));
    dp = cell(size(X));
	dpt = cell(size(X));
	SubjectLabels = mod(0:(b-1),nsubj)+1;
	for ss = 1:nsubj
		z = SubjectLabels == ss;
		Betas{ss} = B(:,z);
		a0{ss} = C(z);
		Yh{ss} = bsxfun(@plus, X{ss} * Betas{ss}, a0{ss});
    	dp{ss} = reshape(dprimeCV(Y{ss}>0,Yh{ss}>0, CV{ss}), ncv, nlam); 
    	dpt{ss} = reshape(dprimeCV(Y{ss},Yh{ss}, ~CV{ss}), ncv, nlam);
	end

	[~,bestLambdaInd] = max(mean(cell2mat(dp)));
	bestLambda = LAMSET(bestLambdaInd);

	fitObjRaw.Betas = Betas;
	fitObjRaw.a0 = a0;
	fitObjRaw.lambda = LAMSET;
	fitObjRaw.mu = MU;
	fitObjRaw.bestLambda = bestLambda;
	fitObjRaw.bestLambdaInd = bestLambdaInd;
	fitObjRaw.dp = dp;
	fitObjRaw.dpt = dpt;
	fitObjRaw.Yh = Yh;
	fitObjRaw.Y = Y;
	fitObjRaw.CV = CV;
	fitObjRaw.method='soslasso';
	delete(fullfile(pwd,'CHECKPOINT.mat'));
end
