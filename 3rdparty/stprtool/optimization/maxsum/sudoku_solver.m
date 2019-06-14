function [Solution,nSol] = sudoku_solver(S)
% SUDOKU_SOLVER solves sudoku puzzle.
% 
% Synopsis:
%  [Solution,nSol] = sudoku_solver(Puzzle)
%
% Example:
% 
% sudoku_solver([0 0 2 9 0 5 0 0 8;
%               0 1 5 0 0 7 2 0 0;
%               0 9 0 0 2 0 1 5 0;
%               0 5 0 0 6 1 0 0 3;
%               0 6 1 0 0 0 9 7 0;
%               8 0 0 2 9 0 0 6 0;
%               0 2 3 0 0 0 0 8 0;
%               0 0 6 8 0 0 4 0 0;
%               0 0 0 1 0 2 3 0 0])
%  
% S = [7 0 0 3 0 0 1 0 8;
%      0 0 8 0 5 0 0 4 2;
%      0 0 0 9 0 0 0 0 0;
%      0 0 7 0 0 0 4 8 0;
%      9 0 0 0 2 0 0 0 7;
%      0 5 4 0 0 0 3 0 0;
%      0 0 0 0 0 6 0 0 0;
%      4 7 0 0 8 0 9 0 0;
%      1 0 0 0 0 5 0 0 4];
%
    
% Modifications:
% 25-aug-2007, VF
    
% Transforms sudoku puzzle to max-sum labeling problem.
Inx = zeros(9,9);
model.nT = 0;
for i=1:9,
  for j=1:9,
     if S(i,j) == 0, 
         model.nT = model.nT + 1;
         Inx(i,j) = model.nT;
     end
  end
end

model.G = zeros(9,9);
model.G(1:10:end) = -inf;
model.nY = 9;
model.Q = zeros(1,9,model.nT);
model.E = uint32(zeros(3,0));
model.q = uint32(1:model.nT);

for i=1:9,
  for j=1:9,
     if S(i,j) == 0, 
         t1 = Inx(i,j);
         
         % i-ty radek
         for k=setdiff(1:9,j)
            if S(i,k) == 0,
               t2 = Inx(i,k);
               if ~any(model.E(1,:)==t1 & model.E(2,:)==t2) && ...
                  ~any( model.E(2,:)==t1 & model.E(1,:)==t2),
                 model.E = [model.E [t1; t2; 1]];
               end
            else
               model.Q(1,S(i,k),t1) = -inf;
            end
         end
         
         % j-ty sloupec
         for k=setdiff(1:9,i)
            if S(k,j) == 0,
               t2 = Inx(k,j);
               if ~any(model.E(1,:)==t1 & model.E(2,:)==t2) && ~any(model.E(2,:)==t1 & model.E(1,:)==t2),
                 model.E = [model.E [t1; t2; 1]];
               end
            else
               model.Q(1,S(k,j),t1) = -inf;
            end
         end
         
         % sector 
         i0 = 3*(ceil(i/3)-1); j0=3*(ceil(j/3)-1);
         for ii=i0+[1:3],
           for jj=j0+[1:3],
              if ii~= i || jj ~= j,
                if S(ii,jj) == 0,
                   t2 = Inx(ii,jj);
                   if ~any(model.E(1,:)==t1 & model.E(2,:)==t2) && ...
                      ~any(model.E(2,:)==t1 & model.E(1,:)==t2),
                      model.E = [model.E [t1; t2; 1]];
                   end
                else
                   model.Q(1,S(ii,jj),t1) = -inf;
                end
              end
           end
         end
         
         
     end
  end
end

[Y,F] = maxsum_feasible(uint32(ones(1,model.nT)),model);

nSol = size(Y,1);

if nSol == 0,
  Solution =[];
elseif nSol == 1,
 Solution = S;
 for i=1:9,
   for j=1:9,
      if S(i,j) == 0,
         Solution(i,j) = Y(1,Inx(i,j));
      end
   end
 end
else

 for k=1:nSol,
   Sol = S;
   for i=1:9,
     for j=1:9,
      if S(i,j) == 0,
         Sol(i,j) = Y(k,Inx(i,j));
      end
    end
   end
   
   Solution{k} = Sol;
 end

end

return;
% EOF