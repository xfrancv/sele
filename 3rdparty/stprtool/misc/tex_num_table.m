function tex_num_table(outFile, data, colNames, colFormat )
% TEX_NUM_TABLE creates file with TeX table.
% 
% tex_num_table(outFile, data, colNames, colFormat )
%
% Example:
%   data = rand(5,3);
%   colNames = {'col1','col2','col3'};
%   colFormat = {'%.1f','%.2f','%.3f'};
%   tex_table('test.tex', data, colNames, colFormat );
%

[nRows,nCols] = size(data);

if nargin < 4, colFormat{1} = '%f'; end
if numel(colFormat) < nCols
    tmp{1} = colFormat;
    colFormat = tmp;
    for i=1:nCols
        colFormat{i}=colFormat{1};
    end
end

%if exist(outFile), error('Output file already exist. Erase it first'); end
if numel(colNames)~= nCols
    error('The number of columns in data does not correspond to colNames'); 
end


fid = fopen(outFile,'w+');
fprintf(fid,'\\begin{tabular}{|');
for i=1:nCols, fprintf(fid,'c|'); end
fprintf(fid,'}\n');
fprintf(fid,'\\hline\n');

if ~isempty(colNames)
    for j=1:nCols
        fprintf(fid,'%s', colNames{j});
        if j < nCols, fprintf(fid,' & ' ); else fprintf(fid,'\\\\ \n'); end
    end    
end
fprintf(fid,'\\hline\n');

for i=1:nRows
    for j=1:nCols
        fprintf(fid,colFormat{j},data(i,j));
        if j < nCols, fprintf(fid,' & ' ); else fprintf(fid,'\\\\ \n'); end
    end
end

fprintf(fid,'\\hline\n');
fprintf(fid,'\\end{tabular}\n');
fclose(fid);


% EOF

