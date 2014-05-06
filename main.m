function [fitObj,fitObjRaw] = main()
	load('jlp_metadata.mat');
	OMIT = load('WhichCV.txt');
	WhichMu = load('WhichMu.txt');

	params.Debias = load('Debias.txt');
	params.GroupShift = load('GroupShift.txt');
	params.GroupSize = load('GroupSize.txt');
	params.muset = load('GroupSparseVals.txt');
	params.MeanCenter = logical(load('MeanCenter.txt'));
	params.NormVariance = logical(load('NormVariance.txt'));
	params.RecoveryMode = load('RecoveryMode.txt');
	params.Save = load('Save.txt');
	params.SharedSpaceVoxelSize = load('SharedSpaceVoxelSize.txt');
	params.lamset = load('SparseVals.txt');
	params.TargetCategory = loadstr('TargetCategory.txt');

	mu = params.muset(WhichMu);
	
	nSubjects = length(metadata);	
	%% Set up Y
	switch params.TargetCategory
	case 'Face'
		Y = {metadata.TrueFaces}';
	case 'Place'
		Y = {metadata.TruePlaces}';
	case 'Thing'
		Y = {metadata.TrueThings}';
	end

	%% Set up X
	X = cell(nSubjects,1);
	for i = 1:nSubjects
		s=load(sprintf('jlp%02d.mat',i),'X');
        if params.MeanCenter == true
			X{i} = bsxfun(@minus,s.X,mean(s.X));
        end
        if params.NormVariance == true
			X{i} = bsxfun(@rdivide,s.X,std(s.X));
        end
	end
    clear s;

	%% Project subjects' data in to Shared Space and create GroupInfo.
	[X,GroupInfo] = jlp_prep_data(X,metadata,params);
	
	%% CV selections
	test = cellfun(@(x) x(:,OMIT), {metadata.CVBLOCKS}, 'unif', 0);
	Xtrain = cell(size(X));
	Ytrain = cell(size(Y));
	CV = cell(size(Y));
	for i = 1:nSubjects
		Xtrain{i} = X{i}(~test{i},:);
		Ytrain{i} = Y{i}(~test{i});
		CV{i} = metadata(i).CVBLOCKS(~test{i},(1:10)~=OMIT);
	end

	[fitObj, fitObjRaw] = jlp_cvsoslasso_condor(Xtrain,Ytrain,CV,params.lamset,mu,GroupInfo);
    save(sprintf('jlp+soslasso_mu%02d_cv_%02d.mat',WhichMu,OMIT),'fitObj','fitObjRaw','OMIT','WhichMu');
end

function [string] = loadstr(filename)
    try
        h=fopen(filename,'r');
    catch ME
        rethrow ME
    end
    string = fscanf(h,'%s');
    fclose(h);
    string = strrep(string,'''','');
    string = strrep(string,'"','');
end
