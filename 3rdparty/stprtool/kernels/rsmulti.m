function ReducedModel = rsmulti(Model, newNumSV, maxDev)
% RSMULTI Reduced set method for multi-class kernel rule. 
%
% Synopsis:
%   ReducedModel = rsmulti( Model, newNumSV )
%   ReducedModel = rsmulti( Model, newNumSV, maxDev )
%
% Description:
%  Given K kernel expansions
%   Phi1_k =  sum   Model.Alpha(k,i)*Phi(model.SV(:,i))    k=1:K
%           i=1:nSV
%  the goal is to find REDUCED_MODEL such that 
%     max   F(k) := ||Phi1_k - Phi2_k||
%    k=1:K
%  is minimal where 
%   Phi2_k = sum ReducedModel.Alpha(k,i)*Phi(ReducedModel.SV(:,i))  k=1:K
%         i=1:new_nSV
%  are reduced kernel expansions and Phi is given implictily by kernel 
%    Phi(x)'*Phi(y) = kernel(x,y,model.kernelName,Model.kernelArgs).
%  
%  The implemented method is a kind of greedy algorithm which finds a 
%  local optimum of the problem. 
%  
%  The method requires that PREIMAGE is implemeted for given kernel. 
%

%% process inputs
if nargin < 3,
    max_dev = 1e-7;
end

[K,nSV] = size(Model.Alpha);
nDim = size(Model.SV,1);

%% compute norms of input expansions and inital objective value
Phi1_normsq = [];
Ka = kernel( Model.SV, Model.kernelName,Model.kernelArgs);
for k=1:K
    Phi1_normsq(k) = Model.Alpha(k,:)*Ka*Model.Alpha(k,:)';
    Fk(k) = sqrt(Phi1_normsq(k));
end
[F,worst_k] = max(Fk);

%% initial solution
ReducedModel.Alpha = [];
ReducedModel.bias = Model.bias;
ReducedModel.SV = [];
ReducedModel.kernelName = Model.kernelName;
ReducedModel.kernelArgs = Model.kernelArgs;
ReducedModel.eval = @kernelclassif;

%% start optimization loop
fprintf('new_nSV  max_k F(k)      worst k\n');
fprintf('   0     %.10f    %d\n', F,worst_k);
i = 0;
while F > max_dev && i < newNumSV
   i = i + 1;

   if i == 1
       alpha = Model.Alpha(worst_k,:);
       X = Model.SV;
   else
       alpha = [Model.Alpha(worst_k,:) -ReducedModel.Alpha(worst_k,:)];
       X = [Model.SV ReducedModel.SV];
   end
   
   z = preimage(X,alpha,Model.kernelName,Model.kernelArgs);
   
   ReducedModel.SV = [ReducedModel.SV z];
   ReducedModel.Alpha = zeros(K,i);
   for k=1:K
      [beta,Fk(k)] = comp_beta(Model.Alpha(k,:),Model.SV,...
                        ReducedModel.SV,Phi1_normsq(k),...
                        Model.kernelName,Model.kernelArgs);
                  
      ReducedModel.Alpha(k,:) = beta(:)';
   end
   [F,worst_k] = max(Fk);
    
   fprintf('%4d     %.10f    %d\n', i, F, worst_k);   
end

return;

function [beta,res] = comp_beta(alpha,A,B,normsqA,kernelName,kernelArgs)

Kb = kernel(B,kernelName,kernelArgs);
Kba = kernel(B,A,kernelName,kernelArgs);
beta = Kb\(Kba*alpha(:));    

res = sqrt(beta(:)'*Kb*beta(:) -2*beta(:)'*Kba*alpha(:) + normsqA); 

return;

