function [pred,wts,nc,cm,acc,probest,optsused] = batchtest (X,L,tr,te,opts)

% BATCHTEST gets predictions & other stats using LIBSVM (Chang & Lin 2000)
%  [pred,wts,nc,cm,acc,probest,optsused] = batchtest (X,L,tr,te,opts);
% 
% X is N x D matrix representing N D-dimensional data points
% L is N x 1 matrix of integer labels
% tr and te are cell arrays of the same length, call it T, with indices of 
%   train and test examples in the i-th fold (you can make them with makefolds.m)
%
% The outputs are  T-element cell arrays or vectors with results of training with the T train-test
%    pairs in tr and te. We will denote by Li the number of test examples in the i-th test set.
% pred{i} is a a Li x 1 vector of predicted labels
% wts{i} is a D-element vector of primal weights computed on the training set (only for linear kernel)
% nc(i) is a scalar with the number of correct labels
% acc(i) is a scalar with the accuracy on the i-th test set
% cm{i} is a k x k confusion matrix whose(a,b)-th entry is the number of examples in the i-th test 
%    set that were class labs(a) and classified as class labs(b)
% probest(i) is a Li x k matrix. whose (a,b)-th entry is the probability that the a-th data point in
%    the i-th test set is classified as class labs(j). Only computed if opts.doprobest==1.
% labs is opts.labels if that was supplied, or unique(L)
%
% Runs LIBSVM on training examples of tr{i} and test examples of te{i}, for i=1:N.
% 
% 
% Places intermediate models and predictions (and optionally, weights & probability
%   estimates) in files of the form tmp_i for i=1:T (to change name from 'tmp' to 'blah',
%   set opts.name = 'blah')
%
% Parameters of opts
%  .name (default = 'tmp') : template of intermediate files
%  .doscale (default = 0) : if 1, scales inputs. if 0, doesnt.
%  .getweights (default = 1) : if 0, no weights are found
%  .doweight (default = 0.001) : if 0, no reweighting. 
%                           : If x (where x is a positive real number), then a reweighting of
%                             1/(x+prob(i)) is used for the i-th class, where prob(i) is the fraction of 
%                               examples in the training set that are of class i.
%                           : If a k-element vector, where k is the number of classes, then
%                             those are the weights
%                           : If 'exp' then a reweighting of exp(-prob(i)) is used
%  .libsvmdir (default='.'): directory, relative to current directory Matlab is running in (default is current)
%  .nfolds (default=2) : number of folds to use in crossvalidation (only for rbf kernel with
%             kernelparam set to 'find' or 'findeach'
%  .kerneltype : 0 linear, 1 poly, 2 rbf (default: not specified, therefore uses libsvm default RBF)
%  .kernelparam : if .kerneltype is 1, this is the degree (default: not specified, therefore uses
%                    libsvm default, currently 3)
%
%      : if .kerneltype is 2 (rbf kernel), there are two parameters required
%       (gamma, c). These can be specified by setting opts.kernelparam
%        to any of the following
%
%        (0) Nothing at all. In this case, default LIBSVM ones are
%        used.
%
%	(1) a string of libsvm flags e.g. '-g 0.123 -c 10' for gamma=0.123, c=10
%
%	(2) 'find' . In this case, different gamma and c are found
%	using grid.py and then applied. (The gamma and c found are
%	returned in optsused.libsvmflags)
%
%	(3) 'findeach' : Same as 'find', but gamma and c are found
%         with grid.py for each of the T parts separately. (Recommended)
%
%  .religion : your religious affiliation (not used in current version, maybe later)
%  .libsvmflags : string with any other flags for libsvm, e.g. '-m 1000' for setting up a
%      1000Mb cache
%  .doprobest (default = 0) : if 1, get probability estimates for each class using LIBSVM's -b option
%  .labels (default = unique(L)) : vector of class labels. Should include unique(L)
%  .log2c : 3-element vector with [start step end] 
%  .log2g : 3-element vector with [start step end]
% 
% Copyright (c) by Dinoj Surendran  (2005)
% $Revision: 0.2 $ $Date: 2005/06/02  $
% mailto:dinoj@cs.uchicago.edu 
 % 
% This program is released unter the GNU General Public License.

if nargin<5,
  opts;
end

if ~isfield(opts,'name'), opts.name = 'tmp'; end
if ~isfield(opts,'doscale'), opts.doscale = 0; end
if ~isfield(opts,'doweight'), opts.doweight = 0.001; end
if ~isfield(opts,'doweightN'), opts.doweightN = 1; end
if ~isfield(opts,'getweights'), opts.getweights = 1; end
if ~isfield(opts,'libsvmdir'), opts.libsvmdir = '.'; end
if ~isfield(opts,'kerneltype'), opts.kerneltype = -1; end
if ~isfield(opts,'kernelparam'), opts.kernelparam = ''; end
if ~isfield(opts,'libsvmflags'), opts.libsvmflags = ''; end
if ~isfield(opts,'doprobest'), opts.doprobest = 0; end
if ~isfield(opts,'nfolds'), opts.nfolds = 3; end
if ~isfield(opts,'labels'), opts.labels = sort(unique(L)); end
if ~isfield(opts,'log2c'),opts.log2c = [-3 1 3]; end

labs = opts.labels;
k = length(labs);

if ~isfield(opts,'log2g'),
  loggest = round(log(1/k)/log(2));         % -5;
  opts.log2g = [loggest-3 1 loggest+2]; 
end



[N,D] = size(X);    
L = reshape(L,length(L),1);
pred = {};
scal = '';% string to add to project name 
wts={};
probest={};
cm = {};
acc = [];
nc = [];

T = length(tr);% should equal length(te)

if (T ~= length(te))
  error ('tr and te should have the same number of folds');
end  
for i=1:T
  if (length(intersect(tr{i},te{i})))
    warning(sprintf('the training and test examples in the %d-th fold have something in common',i));
  end
end  
if ~exist([opts.libsvmdir '/svm-train']) 
  error ([opts.libsvmdir '/svm-train not found']);
end
if ~exist([opts.libsvmdir '/svm-predict']) 
  error ([opts.libsvmdir '/svm-predict not found']);
end
if (opts.doscale) & (~exist([opts.libsvmdir '/svm-predict']))
  error ([opts.libsvmdir '/svm-scale not found']);
end
  
wtflags = '';


if (opts.doweight)
  probs = hist(L,labs);
  probs = probs / sum(probs);
  
  if ischar(opts.doweight) & (strcmpi(opts.doweight,'exp'))
    opts.doweight = exp(1).^(-probs);
  end
  if isnumeric(opts.doweight)
    m = length(opts.doweight);
    if (m==k)
      probwts = opts.doweight.^(-opts.doweightN);
    elseif (m==1)
      probwts = (opts.doweight + probs).^(-opts.doweightN);
    else
      warning(sprintf('opts.doweight has %d entries but there are %d clases',m,k));  
    end
    for i=1:length(probwts)
      wtflags = sprintf('%s -w%d %0.10f ',wtflags,labs(i),probwts(i));
    end
  else
    warning('not sure what your weights are. Ignoring them.');
  end
  opts.libsvmflags = [opts.libsvmflags ' ' wtflags];
end

if (opts.kerneltype ~= -1)
  opts.libsvmflags = [opts.libsvmflags ' -t ' num2str(opts.kerneltype)];
else
  opts.kerneltype = 2;
end 
USEGRIDPY = 0;      % set to 2 if want to use  grid.py for each of the T parts, to 1 if just for 1 part

if (opts.kerneltype == 1)
  if (opts.kernelparam ~= -1) 
    opts.libsvmflags = [opts.libsvmflags ' -d ' num2str(opts.kernelparam)];
  end
elseif (opts.kerneltype == 2)
  if length(opts.kernelparam) 
    if strcmpi ('find',opts.kernelparam)
      USEGRIDPY = 1;
    elseif strcmpi('findeach',opts.kernelparam)
      USEGRIDPY = 2;
    elseif ischar(opts.kernelparam) % had better be libsvm flags
      opts.libsvmflags = [opts.libsvmflags opts.kernelparam];
    end
  end
end

for i=1:T
  svmlwrite([opts.name 'tr' num2str(i)],X(tr{i},:),L(tr{i}));
  svmlwrite([opts.name 'te' num2str(i)],X(te{i},:),L(te{i}));
  if opts.doscale
    cmd = sprintf('! %s/svm-scale -s scalparam%d %str%d > %str%d_scaled',opts.libsvmdir,i,opts.name,i,opts.name,i);
    eval(cmd);
    cmd = sprintf('! %s/svm-scale -r scalparam%d %ste%d > %ste%d_scaled',opts.libsvmdir,i,opts.name,i,opts.name,i);
    eval(cmd);
    scal = '_scaled';
  end
end

probestflag = '';
if (opts.doprobest)
  probestflag = '-b 1';
end
opts.libsvmflags = [opts.libsvmflags ' ' probestflag ' '];

gridpy = '';
if ((USEGRIDPY>=1) & (opts.kerneltype==2))
  gridpyloc = {[opts.libsvmdir '/grid.py'], [opts.libsvmdir '/tools/grid.py'],['./grid.py']};
  for i=1:length(gridpyloc)
    if exist(gridpyloc{i})
      gridpy = gridpyloc{i};
      break;
    end
  end
  if ~length(gridpy)
    warning ('cant find grid.py, not estimating any RBF parameters');
  end
end

allotherflags=opts.libsvmflags;

lastpartwhereestparams = 1; 
if (USEGRIDPY==2)
  lastpartwhereestparams=T;
end

if (USEGRIDPY>=1)
  opts.libsvmflags = {};
  opts.libsvmflags{1} = allotherflags;
  for i=1:lastpartwhereestparams
  
    cmd = sprintf('! %s -v %d -svmtrain %s/svm-train -log2c %d,%d,%d  -log2g %d,%d,%d -out %s_paramrbf %s %str%d%s', gridpy,  opts.nfolds, opts.libsvmdir,   opts.log2c(1),opts.log2c(3),opts.log2c(2),opts.log2g(1),opts.log2g(3),opts.log2g(2),    opts.name, allotherflags, opts.name,i,scal);
    
%    cmd = sprintf('! %s -v %d -svmtrain %s/svm-train -log2c 0,6,1  -log2g %d,%d,1 -out %s_paramrbf %s %str%d%s', gridpy, opts.nfolds, opts.libsvmdir, loggest-3,loggest+2,  opts.name, allotherflags, opts.name,i,scal);

    %    cmd = sprintf('! %s -v %d -svmtrain %s/svm-train -log2c -3,3,1  -log2g %d,%d,1 -out %s_paramrbf %s %str%d%s', gridpy, opts.nfolds, opts.libsvmdir, loggest-2,loggest+2,  opts.name, allotherflags, opts.name,i,scal);
    cmd
    eval(cmd);
    tmp = load(sprintf('%s_paramrbf',opts.name));
    [m_,mw_]=max(tmp(:,3));
    c_ = 2^tmp(mw_,1);
    gamma_ = 2^tmp(mw_,2);
    opts.libsvmflags{i} = sprintf('%s -c %f -g %f ',allotherflags, c_, gamma_);
  end
end


for i=1:T  
  libsvmflags = '';
  if (USEGRIDPY>=1)
    libsvmflags = opts.libsvmflags{min(i,lastpartwhereestparams)};
  else
    libsvmflags = opts.libsvmflags;
  end

  cmd = sprintf('! %s/svm-train %s %str%d%s %str%d.model',opts.libsvmdir,libsvmflags,opts.name,i,scal,opts.name,i)
  eval(cmd);
  cmd=sprintf('! %s/svm-predict %s %ste%d%s %str%d.model %ste%d.pred',opts.libsvmdir,probestflag,opts.name,i,scal,opts.name,i,opts.name,i)
  eval(cmd);

  predfile = sprintf('%ste%d.pred',opts.name,i);
  
  if (opts.doprobest)
    [probest{i},pred{i}] = readprobest(predfile);
  else
    pred{i} = load(predfile);
  end

  [cm{i},nc(i)]=getcm(L(te{i}),pred{i},labs);
  acc(i) = nc(i)/length(te{i});
  
  if ((opts.getweights) & (~opts.kerneltype)) % svm-weight only works for linear kernel
    if exist([opts.libsvmdir '/svm-weight' ])
      wtsfile = sprintf('%str%d_weights',opts.name,i);
      cmd = sprintf('! %s/svm-weight -f %d %str%d.model > %s',opts.libsvmdir,D,opts.name,i,wtsfile);
      eval(cmd);
      wts{i} = readwts(wtsfile);
    else
      ; % nothing - just cant get weights, that's all
    end
  end
end

optsused = opts;