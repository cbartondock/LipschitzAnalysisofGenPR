module EigenApprox

using LinearAlgebra, RandomMatrices

export computeEigenError, testEigenGrad

# Just confirming that \lambda_k(A+\epsilon B)-\lambda_k(A)-\epsilon e_k(A)^T B e_k(A) = O(\epsilon^2)
function computeEigenError(epsilon::Float64, n::Int)
  cholesky_factor::Array{Float64,2} = rand(n,n);
  posmatrix = cholesky_factor'*cholesky_factor;
  pert = rand(n,n);
  original_eigs, original_eigvecs = LinearAlgebra.eigen(posmatrix);
  true_eigs = LinearAlgebra.eigen(posmatrix+epsilon*pert).values;
  approx_eigs = [original_eigs[i]+epsilon*original_eigvecs[:,i]'*pert*original_eigvecs[:,i] for i in 1:n];
  return approx_eigs-true_eigs
end

function testEigenGrad(epsilon::Float64, n::Int)
  cholesky_factor::Array{Float64,2} = rand(n,n);
  posmatrix = cholesky_factor'*cholesky_factor;
  original_eigs, original_eigvecs = LinearAlgebra.eigen(posmatrix);
  eigmin = original_eigs[1];
  eigvecmin = original_eigvecs[:,1];
  new_eigmin = LinearAlgebra.eigmin(posmatrix+epsilon*eigvecmin*eigvecmin')
  return new_eigmin-eigmin
end

end
