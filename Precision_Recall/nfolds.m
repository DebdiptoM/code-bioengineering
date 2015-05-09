function [pred,nc,cm,acc,probest,details] = nfolds (X,L,numfolds,opts)

  % NFOLDS performs n-fold train+test crossvalidation using LIBSVM (Chang & Lin 2000)
  % [pred,nc,cm,accuracy,probest,details] = nfolds (X,L,numfolds,[opts])
  % 
  % X is N x D matrix representing N D-dimensional data points
  % L is N x 1 matrix of integer labels
  % numfolds is the number of folds you wish to split the set into (or a cell array) 
  %
  % Runs LIBSVM on numfolds train and test sets, where the test sets are created such that
  %   each of the N data points occurs in exactly one test set. 
  % Note that numfolds can also be a cell array, with each numfolds{i} being a subvector of [1:N], as long as 
  %  it is then exactly a partition of [1:N]. In this case, for each fold the training 
  %  indices are taken to be all indices from 1 to N other than the test indices.

  % The output of intermediate programs (makefolds and batchtest) are placed in details.
  % 
  % pred is a N x 1 matrix of predicted labels
  % nc is a single number with the total number of correct labels   
  % cm is a k x k confusion matrix
  % accuracy = nc/N
  % probest is a N x k matrix with probest(i,j) = probability that data point i is classified as j (in
  %    the fold where i is part of the test set). It is only computed if opts.doprobest==1 and a
  %    linear kernel is used i.e. opts.kerneltype is 0.
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

  % Copyright (c) by Dinoj Surendran (dinoj@cs.uchicago.edu)
  % Date: 2005/06/12
  % See http://people.cs.uchicago.edu/~dinoj/matlab for more code and updated versions
  % 
  % This program is released unter the GNU General Public License.

if nargin<4,
  opts.name='tmp';
end
N=length(L);

trainfolds = {};
testfolds = {};

if isnumeric(numfolds)
  [trainfolds,testfolds] = makefolds ([1:length(X)],numfolds);
elseif iscell(numfolds)
  if isvalid(numfolds,N)
    testfolds = numfolds;
    for i=1:length(testfolds)
      trainfolds{i} = setdiff ([1:N],testfolds{i});
    end
  else
    error ('the cell array you supplied for numfolds isnt a partition of the indices');
  end
  numfolds = length(testfolds);
end

[pred_,wts_,nc_,cm_,acc_,pe_,ou_] = batchtest (X,L,trainfolds,testfolds,opts);

probest=[];
for n=1:numfolds
  pred(testfolds{n}) = pred_{n};
  if (length(pe_)>=n) & (opts.doprobest)
    probest(testfolds{n},:) = pe_{n};
  end
end
[cm,nc]=getcm(L,pred,sort(unique(L)));
acc=nc/length(L); 

details.predicted = pred_;
details.weights = wts_;
details.numcorrect = nc_;
details.confusionmatrices = cm_;
details.accuracy = acc_;
details.probestimate = pe_;
details.optsused = ou_;
details.trainfolds = trainfolds;
details.testfolds = testfolds;


function yes = isvalid(folds,N)

a = [];
yes = 0;
for i=1:length(folds)
  if (1==size(folds{i},2)),folds{i}=folds{i}';end  % make it a row
  a = [a folds{i}];
  sa = sort(unique(a));
  if ((length(a)==length(sa)) & (min(sa)==1) & (max(sa)==N))
    yes = 1;
  end
end