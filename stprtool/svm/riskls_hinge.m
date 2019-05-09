function [R,subgrad,Fbsf,Wnew] = riskls_svm( Data, Wd, Wbsf, lambda )
% RISKLS_SVM L1-hinge loss for two-class linear classifier.
% 
% Synopsis:
%  [R,subgrad,Fbsf,Wnew] = riskls_svm( Data )
%  [R,subgrad,Fbsf,Wnew] = riskls_svm( Data, Wd, Wbsf, lambda )
%
% Description:
%   F(W) = 0.5*lambda*norm(W) + sum( C.* max(0,1-(X'*W).*Y) )
%
%   where lambda [1x1] is a constant, C [M x 1] are cost factors, 
%   X [N x M] features and Y [M x 1] are labels (+1/-1).
%
%  
%
    MU = 0.1;   % for MU = 1 OCA reduces to BMRM 

    [nDim, nExamples] = size( Data.X );

    if nargin < 2, 
        
        % cutting plane at zero
        Wnew       = zeros(nDim,1); 
        score   = ones( 1, length( Data.Y));
        idx     = find( score > 0);                
        R       = sum( score(idx).*Data.C(idx)' ) ;
        subgrad = -Data.X(:,idx)*(Data.Y(idx).*Data.C(idx)) ;
        Fbsf    = R;
    else
        
        % line search min_{k>=0} F( Wbsf*(1-k) + k*Wd )
        A = 0.5*lambda*norm( Wd - Wbsf )^2;
        B = lambda * Wbsf' * (Wd - Wbsf );
        C = zeros( nExamples, 1);
        D = zeros( nExamples, 1);
        E = (Data.Y').* ((Wbsf-Wd)'*Data.X ).*(Data.C');
        F = (1 - (Data.Y').* (Wbsf' * Data.X )) .* (Data.C');
        
        [k, fval] = svm_linesearch( A, B, C, D, E, F );

        % compute best so far primal solution W from old Wbsf
        Fbsf = fval + 0.5*lambda* norm( Wbsf )^2;
        Wnew = Wbsf*(1-k) + Wd*k;

        % compute cutting plane parametrized by (subgrad,R) defined as
        % R(W) >= R(Wcp) + subgrad'*(W-Wcp) so that R = R(Wcp) -subgrad'*Wcp
        Wcp  = Wnew*(1-MU) + Wd*MU; 
        
        score   = 1 - (Wcp'*Data.X).*Data.Y';    
        idx     = find( score > 0);                
        R       = sum( score(idx).*Data.C(idx)' );
        subgrad = -Data.X(:,idx)*(Data.Y(idx).*Data.C(idx));
        R       = R - subgrad'*Wcp;
    end
end
% EOF