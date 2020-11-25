run('../selclassif_setpath.m');

dataSet = {'california1', 'abalone1','bank1','cpu1','msd1','facebook1',...
           'bikeshare1','ccpp1','superconduct1','gpu1','metro1' };


for i = 1 : numel( dataSet )
    run_train_svorimc( dataSet{i}, 'zmuv' );
end


if ~strcmpi( 'HAL', hostname )
    exit();
end

