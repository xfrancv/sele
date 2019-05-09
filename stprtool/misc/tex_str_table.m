function tex_str_table(outFile, table, align )
% TEX_STR_TABLE creates file with TeX table.
% 
% tex_str_table(outFile, table, align )
%
% Example:
%   tab = [{'','jim','fredy','john'}; {'points','1','2','3'};]
%   tex_str_table('test.tex', tab );
%

if nargin < 3, align = 'c'; end

[nRows,nCols] = size(table);

fid = fopen(outFile,'w+');
fprintf(fid,'\\begin{tabular}{|');
for i=1:nCols, fprintf(fid,'%c|', align); end
fprintf(fid,'}\n');
fprintf(fid,'\\hline\n');

for i=1:nRows
    for j=1:nCols
        fprintf(fid,'%s ', table{i,j} );
        if j==nCols, fprintf(fid,'\\\\ \n'); else fprintf(fid,' & '); end
    end
end

fprintf(fid,'\\hline\n');
fprintf(fid,'\\end{tabular}\n');
fclose(fid);


% EOF

