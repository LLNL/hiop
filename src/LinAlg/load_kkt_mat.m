%% specify filename
filename = 'kkt_linsys_0.iajaaa';
f = fopen(filename,'r');
A = fscanf(f, '%f');
fclose(f);

%% data loading
m=A(1); % matrix size 
nx=A(2); % number of (primal) variables of the underlying NLP
meq=A(3); % number of equalities of the underlying NLP
nineq=A(4); % number of inequalities of the underlying NLP
nnz=A(5); % number of nnz of matrix (or of the upper triangle for symmetric matrices)
rowp = A(6:m+6); % row pointers in colidx and vals (see below)
colidx=A(m+7:nnz+m+6); % column indexes for each nz
vals=A(nnz+m+7:nnz+nnz+m+6); % values for each nz
rhs=A(nnz+nnz+m+7:nnz+nnz+m+m+6); % rhs 
sol=A(nnz+nnz+m+m+7:nnz+nnz+m+m+m+6); % solution
fprintf('Loaded a matrix of size %d with %d nonzeros from "%s"\n', ...
    m, nnz, filename);

%% convert to triplet needed by Matlab's 'sparse'
rowidx=zeros(nnz,1);
for ii=2:m+1
    for jj=rowp(ii-1):(rowp(ii)-1)
        rowidx(jj) = ii-1;
    end
end

%% form the matrix
M = sparse(rowidx, colidx, vals, m, m);
% if the matrix is known to be symmetric fill, you also need to know whether
%  A. the lower triangle was saved, in which case use
% M=M+transpose(tril(M,-1));
% or
% B. the upper triangle was saved, in which case use
% M=M+transpose(triu(M,1));

%% solve using Matlab and check the residuals of the Matlab solution 
%% and of the solution from .iajaaa file
sol2 = M\rhs;
fprintf('residuals norm: Matlab sol %.5e   sol from file: %.5e\n', ...
    norm(M* sol2 - rhs), norm(M* sol - rhs));
