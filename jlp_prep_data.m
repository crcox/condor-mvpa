function [X,GroupInfo] = jlp_prep_data(X,metadata,params)
    %% Parameters
    nSubjects  = length(X);
    GroupSize  = params.GroupSize;
    GroupShift = params.GroupShift; 
	XYZ_tlrc   = {metadata.xyz_tlrc}';

	temp       = cellfun(@(x) unique(roundto(x,params.SharedSpaceVoxelSize),'rows'), XYZ_tlrc, 'unif', 0);
    for ss = 1:length(temp)
        if length(temp{ss}) < length(XYZ_tlrc{ss});
            disp(ss)
            error('Collision!')
        end
    end
    XYZ_tlrc = temp;
    clear temp;
    
    %% Concatenate all coordinates
    XYZ_tlrc_mat = cell2mat(XYZ_tlrc);
    
    %% Find the range of each dimension that contains all coordinates.
    minXYZ = min(XYZ_tlrc_mat);
    maxXYZ = max(XYZ_tlrc_mat);
    maxIJK = ((maxXYZ - minXYZ)/params.SharedSpaceVoxelSize)+ 1;
    I = maxIJK(1);
    J = maxIJK(2);
    K = maxIJK(3);
    clear maxXYZ maxIJK;
    
    %% Inflate slightly, as needed, to accomodate group size (and GroupShift).
    I = I + (GroupShift-mod(I-GroupSize,GroupShift));
    J = J + (GroupShift-mod(J-GroupSize,GroupShift));
    K = K + (GroupShift-mod(K-GroupSize,GroupShift));
    
    GroupInfo.ijk_tlrc_range = [I J K];

    %% MAPPING FROM ijk ---> n = i + I(j-1) + IJ(k-1)
    IJK_TLRC = ((bsxfun(@minus,XYZ_tlrc_mat,minXYZ)/params.SharedSpaceVoxelSize)+1);
    IND_TLRC = uint32(unique(sub2ind([I,J,K],IJK_TLRC(:,1),IJK_TLRC(:,2),IJK_TLRC(:,3))));
    IND_TLRC = sort(IND_TLRC); % In Nikhil's it seems to be sorted...
    clear IJK_TLRC;
    % Breakdown:
    % 1. Take coordinates for all subjects as one large matrix, subtract the
    % minimums, and add 1. Transformation to 1 based IJK indexes in TLRC space.
    % 2. Transform those IJK indexes to literal column indexes. Retain only the
    % unique ones, and store as an array of unsigned 32 bit integers.

    NZ_TLRC = false(1,I*J*K);
    NZ_TLRC(IND_TLRC) = true;
    GroupInfo.NZ_TLRC = NZ_TLRC;
    
    % MAKE NEW DATA MATRICES 
    fprintf('\n');
    for ss = 1:nSubjects
        fprintf('% 2d',ss)
        xyz_tlrc = XYZ_tlrc{ss};
        ijk_tlrc = (bsxfun(@minus,xyz_tlrc,minXYZ)/params.SharedSpaceVoxelSize) + 1;
        ind_tlrc = sub2ind([I J K],ijk_tlrc(:,1),ijk_tlrc(:,2),ijk_tlrc(:,3));
        funcData_tlrc = zeros(size(X{ss},1),I*J*K);
        funcData_tlrc(:,ind_tlrc) = X{ss};
        funcData_tlrc = sparse(funcData_tlrc(:,NZ_TLRC));
		X{ss} = funcData_tlrc;
        clear ijk xyz_tlrc;
    end
    fprintf('\n')

    %% Define Range 
    irange = 1:GroupShift:(I-GroupSize);
    jrange = 1:GroupShift:(J-GroupSize);
    krange = 1:GroupShift:(K-GroupSize);

    %% Generate Grid of Corners
    [KG,JG,IG] = ndgrid(krange,jrange,irange);
    IJK_corners = [IG(:), JG(:), KG(:)]; % Grid of group corners.

    %% Generate Grid-patch to make groups
    [IG,JG,KG]= ndgrid(0:(GroupSize-1),0:(GroupSize-1),0:(GroupSize-1));
    GroupExtent = [IG(:), JG(:), KG(:)]; % Grid of distances from corner.

    %% Make G (full)
    M = length(IJK_corners);
    G = cell(M,1);
    for i = 1:(M)
        G_ijk = bsxfun(@plus, IJK_corners(i,:), GroupExtent);
        temp = sub2ind([I,J,K], G_ijk(:,1),G_ijk(:,2),G_ijk(:,3))';
        G{i} = uint32(temp(NZ_TLRC(temp)));
    end
    temp = setdiff(IND_TLRC,cell2mat(G'));
    G{M+1} = uint32(temp(NZ_TLRC(temp)));
    clear IND_TLRC;
    %% Remove Empty Groups
    G = G(~cellfun('isempty',G));

    %% Remove Repeated Groups 
    G = uniquecell(G);
    G = G';

    %% Map to small space
    GroupInfo.G_oldspace = G;
    M = length(G);
    G = cell(M,1);

    nth_NZ_TLRC = repmat(uint32(0),1,length(NZ_TLRC));
    nth_NZ_TLRC(NZ_TLRC) = 1:sum(NZ_TLRC);
    for ii = 1:M
        G{ii} = nth_NZ_TLRC(GroupInfo.G_oldspace{ii});
    end
    GroupInfo.G = G;
    clear G;

    %% Define the replicated, non-overlapping space.
    [GroupInfo.RepIndex, GroupInfo.groups, GroupInfo.group_arr] = define_rep_space(GroupInfo.G);
end

function [RepIndex, groups, group_arr] = define_rep_space(G)
% DEFINE_REP_SPACE Generate all necessary indexes for working with data in a
% replicated space where voxels in multiple groups can be considered as a
% member of each group, and treated like several voxels. Ultimately, the
% weights on several instances of the same voxel in different groups will
% be aggregated.
%
% USAGE:
% [RepIndex, groups, group_arr] = DEFINE_REP_SPACE(G)
%
% INPUTS:
% G:      A Kx1 cell array that contains the column (i.e. voxel) indexes
% in the unreplicated data that belong to each group, where K is the number
% of groups.
%
% OUTPUTS:
% REPINDEX:    A 1xN vector, where N is the number of columns in the
% replicated matrix.  The indexes refer to columns in the unreplicated
% matrix, and when used to select from the unreplicated matrix will produce
% the replicated matrix.  Because the replicated matrix is very large, care
% should be taken to not modify it, otherwise a full copy will be made in
% memory.  If REPINDEX is merely used to index into the unreplicated
% matrix, replication can be achieved by redundant pointers, and not by
% redundent copies.  With large datasets, this is essential to keep the
% problem tractable.
%
% GROUPS:     A 1xN vector, where N has the same meaning as above.  Each
% element in GROUPS is a group label in the replicated space.  The
% replicated matrix has columns ordered by group, so GROUPS will be
% [repmat(1,1,length(G{1})), repmat(2,1,length(G{2})), ... ,
% repmat(K,1,length(G{K}))], where K is the number of groups.
%
% GROUP_ARR: A KxMaxGroupSize matrix, where K means the same as above, and
% MaxGroupSize = max(cellfun(@length,G)).  Rows correspond to groups, and
% elements are indexes into the replicated matrix that would select a
% single group.  If groups are unequal, rows are padded out with N+1, an
% index just outside of the column range of the replicated matrix. This
% dummy index will be handled appropriately at later steps.
%
% DEFINE_REP_SPACE is a memory efficient replacement for makeA_multitask.
%
% Chris Cox | July 24, 2013
    if isrow(G);
        G = G';
    end
    K = length(G);
    MaxGroupSize = uint32(max(cellfun(@length,G)));
    RepSpaceSize = uint32(sum(cellfun(@length,G)));
%% Create RepIndex
    RepIndex = cell2mat(G');

%% Create groups
    temp = G;
    for i = 1:K
        temp{i}(:) = i;
    end
    groups = cell2mat(temp');

%% Create group_arr
    group_arr = repmat(RepSpaceSize+1,K,MaxGroupSize);
    a = 1;
    for i = 1:K
        glen = length(G{i});
        b = a+glen-1;
        group_arr(i,1:glen) = a:b;
        a = b+1;
    end
end
