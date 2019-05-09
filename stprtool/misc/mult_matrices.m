function C = mult_matrices( A, B)
% MULT_MATRICES multiplies matrices A and B which can be of different types.
% 
% Synopsis:
%   C = mult_matrices( A, B)
% 
% Description:
%  Computes
%    C = A*B
%  
% Implementation of this function is necessary because Matlab for some
% unknown reason does not support multiplication of matrices of
% all types. Eg. if A is an array of int32 it refuces to work.
%
% Currently supported multiplications:
%   A [n x m]       B [m x p]
%   int32           double
%   int32           int32
%   double          int8
%   double          sparse logical
%
% The resulting matric C is always an array of doubles.
%

if isa(A,'double') & isa(B,'double')
    C = A*B;
elseif isa(A,'int32') & isa(B,'double') & ~issparse(A) & ~issparse(B)
    C = mult_matrices_int32xdouble_mex(A,B);
elseif isa(A,'int32') & isa(B,'int32') & ~issparse(A) & ~issparse(B)
    C = mult_matrices_int32xint32_mex(A,B);
elseif isa(A,'double') & isa(B,'int8') & ~issparse(A) & ~issparse(B)
    C = mult_matrices_doublexint8_mex(A,B);
elseif isa(A,'double') & isa(B,'logical') & ~issparse(A) & issparse(B)
    C = mult_matrices_doublexsparselogical_mex(A,B);
else
    error('Unsupported types of A and B');
end