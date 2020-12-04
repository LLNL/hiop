%% specify filename
filename = 'kkt_mat_0.iajaaa';
f = fopen(filename,'r');
A = fscanf(f, '%f');
fclose(f);

%% data loading
m=A(1); % matrix size 
nnz=A(2); % number of nnz of the upper triangle
rowp = A(3:m+3); % row pointers in colidx and vals (see below)
colidx=A(m+4:nnz+m+3); % column indexes for each nz
vals=A(nnz+m+4:nnz+nnz+m+3); % values for each nz
rhs=A(nnz+nnz+m+4:nnz+nnz+m+m+3); % rhs 
sol=A(nnz+nnz+m+m+4:nnz+nnz+m+m+m+3); % solution
fprintf('Loaded a matrix of size %d with %d nonzeros from "%s"\n', ...
    m, nnz, filename);

%% convert to triplet needed by Matlab's 'sparse'
rowidx=zeros(nnz,1);
for ii=2:m+1
    for jj=rowp(ii-1):(rowp(ii)-1)
        rowidx(jj) = ii-1;
    end
end

%% form the matrix and fill lower triangle
M = sparse(rowidx, colidx, vals, m, m);
M=M+transpose(triu(M,1));

%% solve using Matlab and check the residuals of the Matlab solution 
%% and of the solution from .iajaaa file
sol2 = M\rhs;
fprintf('residuals norm: Matlab sol %.5e   sol from file: %.5e\n', ...
    norm(M* sol2 - rhs), norm(M* sol - rhs));
