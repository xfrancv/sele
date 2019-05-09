function h = plot_ellipse_around_points( X, style )
%
%  h = plot_ellipse_around_points( X, style )
%
% Example:
%   X = randn(2,10);
%   figure;
%   ppatterns(X); hold on;
%   plot_ellipse_around_points( X, 'r' );
%


if nargin < 2, style = 'b'; end

%%
N = size(X,2);

if N == 1
    h = plot(X(1),X(2),'ok');
elseif N == 2
    h = plot(X(1,:),X(2,:),style);
else


    %%
    beta = ones(1,N);
    for i = 1:100
        alpha     = [1;1]*beta/sum(beta);
        Model     = gmm_ml( X, alpha, 'full', 1e-3 );
        P         = gmm_logpxy( X, Model); 
        [~,idx]   = min( P(1,:) );
        beta(idx) = beta(idx) + 1;  
    end

    C  =  Model.U(:,:,1)*diag(Model.D(:,1))*Model.U(:,:,1)';
    A  = inv(C);
    mu = Model.Mean(:,1);

    md = zeros(N,1);
    for i = 1 : N, md(i) = (X(:,i)-mu)'*A*(X(:,i)-mu); end

    %%
    [x,y]=ellips(mu,A,sqrt(max(md)), 100); 
    h = plot(x,y,style);
end