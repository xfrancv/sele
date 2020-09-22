run('../selclassif_setpath.m');

dataSet = {'california1', 'abalone1','bank1','cpu1'};
       

for i = 1 : numel( dataSet )
    run_train_svorimc( dataSet{i}, 'zmuv' );
end


if ~strcmpi( 'HAL', hostname )
    exit();
end

