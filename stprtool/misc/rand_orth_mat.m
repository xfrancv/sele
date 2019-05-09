function R = rand_orth_mat( N )
% R = rand_orth_mat( N ) randomly generates orthogonal matrix. 
%
% Synopsis:
%   R = rand_orth_mat( N )
%
% Input:
%   N [1x1] matrix dimension.
%
% Output:
%   R [NxN] orthogonal matrix.
% 

% Addopted from Ofek Shilon:
% http://www.mathworks.com/matlabcentral/fileexchange/authors/23951

R = zeros( N ); 
        
u = randn( N, 1);  

R(:,1) = u/norm(u);
    
for i=2:N
    normu = 0;
	while normu < 1e-6
        u = randn(N,1);
        u = u -  R(:,1:i-1)  * ( R(:,1:i-1).' * u )  ;
        normu = norm(u);
	 end
	 R(:,i) = u ./ normu;
end

% EOF