function [P,pred,labels] = readprobest (filename)
  
  % READPROBEST - read probability estimates in format output by libsvm
  % [P,pred,labels] = readprobest (filename)
  %
  % P is a N x L matrix, where N is the number of objects and L is the number of labels
  % P(n,k) is the probability that the n-th object has label labels(k)
  % labels is a L-element vector, with sorted class labels
  % pred(n) has the most likely class predicted
    

  a = textread (filename, '%s', 'commentstyle', 'matlab', 'headerlines',0, 'delimiter', '\n' );
  tmp = strread (a{1},'%s','delimiter', ' ');

  P = [];
  pred=[];
  labels = [];
  for i=2:length(tmp)
    labels(i-1) = str2num(tmp{i});
  end
  labelssorted = sort(labels);
  L = length(labels);
  for j=1:L
    lsi(j) = find(labelssorted==labels(j));
  end
  N = length(a)-1;
  for i=1:N
    tmp = strread(a{i+1},'%f','delimiter',' ');
    % tmp is a (1+L)-dimensional vector
    % tmp(1+j) has the probability that this (the i-th example) is in class labels(j), so
    % place it in P(i,find(labelssorted==labels(j))) = P(i,lsi(j))
    pred(i,1) = tmp(1);
    for j=1:L
      P(i,lsi(j)) = tmp(j+1);
    end
  end    