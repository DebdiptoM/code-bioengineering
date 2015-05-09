function RES = splitbysomething (X,L,tr,te,smtg,splitby,opts)
%function [FORFOLDS,FORSPLITS,DETAILS,COMBINED] = splitbysomething (X,L,tr,te,smtg,splitby,opts)

  % SPLITBYSOMETHING Creates different classifiers with same folds and conditional on a partition of a different vector
  % RES = splitbysomething (X,L,tr,te,smtg,splitby,opts)
  % RES has fields 
  %  .folds
  %  .splits
  %  .combined
  %  .details
  %
  % X is N x D matrix representing N D-dimensional data points
  % L is N x 1 matrix of integer labels
  % tr and te are cell arrays of the same length (=:numfolds), with indices of 
  %   train and test examples in the i-th fold (you can make them with makefolds.m)
  % smtg is a N x 1 matrix of values
  % splitby is a S x 1 cell array such that different classifiers are created from tr 
  %   and te conditional on splitting by values in splitby. IT SHOULD PARTITION SMTG!
  % 
  % Parameters of opts
  %  .name (default = 'tmp') : template of intermediate files
  %  .doscale (default = 0) : if 1, scales inputs. if 0, doesnt.
  %  .doweight (default = 0) : if 0, no reweighting. 
  %                           : If x (where x is a positive real number), then a reweighting of
  %                             1/(x+prob(i)) is used for the i-th class, where prob(i) is the fraction of 
  %                               examples in the training set that are of class i.
  %                           : If a k-element vector, where k is the number of classes, then
  %                             those are the weights
  %                           : If 'exp' then a reweighting of exp(-prob(i)) is used
  %  .libsvmdir (default='.'): directory, relative to current directory Matlab is running in (default is current)
  %  .kerneltype : 0 linear, 1 poly, 2 rbf (default: not specified, therefore uses libsvm default RBF)
  %  .kernelparam : if .kerneltype is 1, this is the degree (default: not specified, therefore uses
  %                    libsvm default, currently 3)
  %                 If .kerneltype is 2, this is gamma (default: not specified, therefore uses libsvm
  %                    defaults, currently 1/k
  %  .religion : your religious affiliation (not used in current version, maybe later)
  %  .libsvmflags : string with any other flags for libsvm, e.g. '-m 1000' for setting up a
  %      1000Mb cache
  %  .doprobest (default = 0) : if 1, get probability estimates for each class using LIBSVM's -b option
  %  
  % 
  % 
  %
  % pred is a N x 1 matrix of predicted labels
  % wts is {} unless you're using a linear kernel. In that case it is a n+2 cell array, with
  %    wts{i} having a D-dimensional vector of primal weights for the i-th fold, i=1:n.
  %    wts{n+1} has the average primal weight
  %    wts{n+2} has the number of folds where the weights have the same sign
  % nc is a single number with the total number of correct labels   
  % cm is a k x k confusion matrix
  % accuracy = nc/N
  % probest is a N x k matrix with probest(i,j) = probability that data point i is classified as j (in
  %    the fold where i is part of the test set). It is only computed if opts.doprobest==1 and a
  %    linear kernel is used i.e. opts.kerneltype is 0.
  %
  % Copyright (c) by Dinoj Surendran  (2005)
  % $Revision: 0.2 $ $Date: 2005/04/17  $
  % mailto:dinoj@cs.uchicago.edu
  % 
  % This program is released unter the GNU General Public License.

if nargin<5,
  opts;
end

if ~isfield(opts,'name'), opts.name = 'tmp'; end
if ~isfield(opts,'doscale'), opts.doscale = 0; end
if ~isfield(opts,'doweight'), opts.doweight = 0; end
if ~isfield(opts,'libsvmdir'), opts.libsvmdir = '.'; end
if ~isfield(opts,'kerneltype'), opts.kerneltype = -1; end
if ~isfield(opts,'kernelparam'), opts.kernelparam = -1; end % later -1 becomes 2
if ~isfield(opts,'libsvmflags'), opts.libsvmflags = ''; end 
if ~isfield(opts,'doprobest'), opts.doprobest = 0; end

nfolds=length(tr);
if (nfolds ~= length(te))
  error('tr and te should be cell arrays of the same size');
end
for n=1:nfolds
  if (length(intersect(tr{n},te{n})))
    error (sprintf('%d-th fold has overlap between training and test sets',n));
  end
end

numclasses = length(unique(L));
N = length(L);
% [pred,wts,nc,cm,accuracy,probest] = splitbysomething (X,L,tr,te,smtg,splitby,opts)

if ~ispartition(smtg,splitby)
  error('splitby should be a partition of smtg');
end


indicesofsplit = {};
for j=1:length(splitby)
  indicesofsplit{j} = [];
  for k=1:length(splitby{j})
    f = find(smtg==splitby{j}(k));
    if (size(f,1)==1)
      f=f';
    end
    indicesofsplit{j} = vertcat(indicesofsplit{j},f);
  end
  % now indicesofsplit{j} has the indices of all examples (test and training) that have a value
  % in the j-th split, i.e. value equal to something in smtg(split{j})

  for n=1:nfolds
    [x,xtrn,xiosj] = intersect( tr{n}, indicesofsplit{j} );
    trainindicesofsplit{j}{n} = xiosj;
    [x,xten,xiosj] = intersect( te{n}, indicesofsplit{j} );
    testindicesofsplit{j}{n} = xiosj;
  end
  % We must have   X(indicesofsplit{j}( trainindicesofsplit{j}{n} ),:) = X(  intersect(tr{n},indicesofsplit{j})  ,:)
  % Similarly,     X(indicesofsplit{j}( testindicesofsplit{j}{n} ),:)  = X(  intersect(te{n},indicesofsplit{j})  ,:)
  
end

%  save intermediate.mat

DETAILS=cell(length(splitby),1);

for j=1:length(splitby)
  j
  [DETAILS{j}.pred, DETAILS{j}.wts, DETAILS{j}.nc, DETAILS{j}.cm, DETAILS{j}.ac, DETAILS{j}.pe, DETAILS{j}.optsused ] = batchtest (X(indicesofsplit{j},:), L(indicesofsplit{j}), trainindicesofsplit{j}, testindicesofsplit{j}, opts);
%  save intermediate.mat
end

for n=1:nfolds
  FORFOLDS.nc(n) = 0;
  FORFOLDS.pred{n} = zeros(length(te{n}),1);
  FORFOLDS.pe{n} = zeros(length(te{n}),numclasses);
  FORFOLDS.cm{n} = zeros(numclasses);
  FORFOLDS.wts{n} = [];
  for j=1:length(splitby)
    [x,xten,xiosj] = intersect(te{n}, indicesofsplit{j});   
    % FORFOLDS.pred{n} should correspond to the PRED{n} you would get from batchtest(X,L,tr,te,opts) 
    % Since PRED{n}(q,:) is the prediction for X(te{n}(q),:) we have that the
    % DESIRED OUTCOME: FORFOLDS.pred{n}(q,:) should be the prediction for X(te{n}(q),:)
    % FACT 1: DETAILS{j}.pred{n}(r,:) is the prediction for X(indicesofsplit{j}(testindicesofsplit{j}{n}(r)),:) 
    %                                                   = X( x(r), :)         // where x := intersect( te{n}, indicesofsplit{j} )
    % FACT 2: x = te{n}(xten)
    % FACT 1+2  means that DETAILS{j}.pred{n}(:,:) is prediction for X(x(:),:) = X(te{n}(xten),:)
    % If we set FORFOLDS.pred{n}(xten,:) to DETAILS{j}.pred{n} then it will be for the prediction X(te{n}(xten),:), which is correct
     FORFOLDS.pred{n}(xten) = DETAILS{j}.pred{n};
     if (size(FORFOLDS.pe{n}(xten,:),2) > size(DETAILS{j}.pe{n},2))
       a = sort(unique(FORFOLDS.pe{n}(xten,:)));
       b = sort(unique(DETAILS{j}.pe{n}));
       for kk=1:length(a)
	 fba = find(b == a(kk));
	 if length(fba)
	   FORFOLDS.pe{n}(xten,fba) = DETAILS{j}.pe{n}(:,kk);
	 end
       end
     elseif (size(FORFOLDS.pe{n}(xten,:),2) == size(DETAILS{j}.pe{n},2))
       FORFOLDS.pe{n}(xten,:) = DETAILS{j}.pe{n};
     end
     
%     if (length(DETAILS{j}.wts)>=n) && (iscell(DETAILS{j}.wts))
%       FORFOLDS.wts{n}(xiosj,:) = DETAILS{j}.wts{n};
%     end
     FORFOLDS.nc(n) = FORFOLDS.nc(n) + DETAILS{j}.nc(n);
     FORFOLDS.cm{n} = FORFOLDS.cm{n} + DETAILS{j}.cm{n};
  end

  FORFOLDS.ac(n) = FORFOLDS.nc(n) / length(te{n});

%  save intermediate.mat

end



for j=1:length(splitby)
  FORSPLITS.nc(j) = 0;
  FORSPLITS.pred{j} = zeros(length(indicesofsplit{j}),1);
  FORSPLITS.pe{j} = zeros(length(indicesofsplit{j}),numclasses);
  FORSPLITS.cm{j} = zeros(numclasses);
  FORSPLITS.wts{j} = [];

  for n=1:nfolds
    [x,xten,xiosj] = intersect(te{n}, indicesofsplit{j});   

    FORSPLITS.pred{j}(xiosj) = DETAILS{j}.pred{n};

%     save DETAILS.mat DETAILS j n xten FORFOLDS FORSPLITS x xiosj te tr *indicesofsplit
     
     if ((max(xiosj) <= size(FORSPLITS.pe{j},1)) && (size(FORSPLITS.pe{j}(xiosj,:),2) > size(DETAILS{j}.pe{n},2)))
       a = sort(unique(FORSPLITS.pe{j}(xiosj,:)));
       b = sort(unique(DETAILS{j}.pe{n}));
       for kk=1:length(a)
	 fba = find(b == a(kk));
	 if length(fba)
	   FORSPLITS.pe{j}(xiosj,fba) = DETAILS{j}.pe{n}(:,kk);
	 end
       end
     else
       FORSPLITS.pe{j}(xiosj,:) = DETAILS{j}.pe{n};
     end
    

%     if (length(DETAILS{j}.wts))
%       FORSPLITS.wts{j}(xiosj,:) = DETAILS{j}.wts{n};
%     end
     FORSPLITS.nc(j) = FORSPLITS.nc(j) + DETAILS{j}.nc(n);
     FORSPLITS.cm{j} = FORSPLITS.cm{j} + DETAILS{j}.cm{n};
  end

  FORSPLITS.ac(j) = FORSPLITS.nc(j) / length(indicesofsplit{j});

%  save intermediate.mat

end


% if testindices are a partition of [1:N] then calculate combined stats

if ispartition([1:N],te)   
  COMBINED.cm = zeros(numclasses);
  COMBINED.pred = zeros(N,1);
  COMBINED.pe = zeros(N,numclasses);
  for n=1:nfolds
    COMBINED.cm = COMBINED.cm + FORFOLDS.cm{n};
    COMBINED.pred(te{n}) = FORFOLDS.pred{n};
    COMBINED.pe(te{n},:) = FORFOLDS.pe{n};
  end
  COMBINED.nc = sum(FORFOLDS.nc);
  COMBINED.ac = COMBINED.nc / N;
else
  COMBINED = [];
end

%  save intermediate.mat

RES.splits=FORSPLITS;    
RES.folds=FORFOLDS;
RES.details=DETAILS;
RES.combined=COMBINED;
    
