function result = bmrm_svm_test()
% BMRM_SVM_TEST tests functionality of BMRM solver on SVM learning problem.

output_fname = 'bmrm_svm_test_output.mat';

load('riply_dataset','trn_X','trn_y');
trn_y(find(trn_y~=1)) = -1;

data.X = trn_X;
data.X0 = 1;
data.y = trn_y;

lambda = 0.001;
options.verb = 0;

[W,stat]= bmrm(data,@risk_svm,lambda,options);

if exist(output_fname)
    reference =  load(output_fname);
    if sum(abs(reference.W-W)) == 0
        fprintf('BMRM_TEST: Test passed OK.\n');
        result = 1;
    else
        fprintf('BMRM_TEST: Error occured.\n');
        result = 0;
    end
else
    save(output_fname,'W','stat');
    fprintf('BMRM_TEST: Reference output created.');
    result = 2;
end
