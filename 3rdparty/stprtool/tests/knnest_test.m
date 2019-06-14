function result=knnest_test()
% KNNEST_TEST tests functionality of KNNEST.

output_fname = 'knnest_test_output.mat';

load('riply_dataset','trn_X','trn_y','tst_X');

K = 5;
out  = knnest(tst_X,trn_X,trn_y,K);

[posterior,pred_y] = max(out);
posterior = posterior/K;


if exist(output_fname,'file')
    reference =  load(output_fname);
    if sum(abs(posterior-reference.posterior))==0 && ...
       sum(abs(pred_y-reference.pred_y)) == 0
        fprintf('KNNEST_TEST: Test passed OK.\n');
        result = 1;
    else
        fprintf('KNNEST_TEST: Error occured.\n');
        result = 0;
    end
else
    save(output_fname,'posterior','pred_y');
    fprintf('KNNEST_TEST: Reference output created.');
    result = 2;
end

return;
% EOF
