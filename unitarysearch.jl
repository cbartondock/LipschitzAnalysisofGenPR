module UnitarySearch

using LinearAlgebra, RandomMatrices, Plotly, LaTeXStrings
export mu, tau, j_morph, diag, offdiag, mu_i, tau_i, diag_i, outer, outerdiff, unitaryproj, upgradeToF
export genFrame, genU, genU2
export frameeigs, minimizeLambdaQ, minimizeLambdaQGlobal, minimizeLambdaQWithHistory, a0
export R_hat, T_hat, A2_hat, a2_hat
export Q, lambdaQ, gradLambdaQ
export testGrad, computeGradError, testU2
export plot1, plot2, plot3, plot4, plot5, plot6, plot7, plot7alt, plot8, plot8alt, plot9, plot9alt

####################
# Internal Functions
####################

# Extract diagonal of square matrix as a vector
function diag(A::Array{Float64,2})::Array{Float64,1}
  return A[CartesianIndex.(axes(A,1),axes(A,2))];
end

# Pseudo inverse of diag
function diag_i(v::Array{Float64,1})::Array{Float64,2}
  result = zeros(Float64,size(v,1),size(v,1));
  result[CartesianIndex.(axes(result,1),axes(result,2))]=v;
  return result;
end

# Extract off-diagonal of square matrix as a vector
function offdiag(A::Array{Float64,2})::Array{Float64,1}
  return A[CartesianIndex.((m,n) for n in 2:size(A,2) for m in 1:(n-1))];
end

# Isometrically vectorize a complex matrix
function mu(A::Array{<:Number,2})::Array{Float64,1}
  return reshape(hcat(real(A), imag(A)),2*size(A,2)*size(A,1));
end

# Isometrically vectorize a symmtric matrix
function tau(A::Array{<:Number,2})::Array{Float64,1}
  return vcat(diag(real(A)), sqrt(2)*offdiag(real(A)), sqrt(2)*offdiag(imag(A)));
end

# Injective homomorphism between \C^{n\times r} and \R^{2 n\times 2 r}
function j_morph(A::Array{<:Number,2})::Array{Float64,2}
  return [real(A) -imag(A); imag(A) real(A)];
end

# Take frame up to real outer frame
function upgradeToF(f::Array{Array{Complex{Float64},2},1}, k::Int)::Array{Array{Float64,2},1}
  return [kron(Matrix{Float64}(I,k,k),j_morph(f[j])) for j=1:size(f,1)];
end

# z to \hat{z} such that [\hat{z}|0]V= z
function fullRankPart(z::Array{Complex{Float64},2},tol::Float64=10^-5)::Array{Complex{Float64},2}
  U, s, V = svd(z);
  Indices = collect(1:size(s,1))
  nonZero = Indices[s[Indices].>tol];
  return U*diag_i(s)*V[:, nonZero];
end

# Inverse of mu
function mu_i(v::Array{<:Number,1}, nrow::Int, ncol::Int)::Array{Complex{Float64},2}
  return (x->x[:,1:ncol]+im*x[:,(ncol+1):2*ncol])(reshape(v,nrow,2*ncol));
end

# Inverse of tau
function tau_i(v::Array{<:Number,1},n::Int)::Array{Complex{Float64},2}
  result::Array{Complex{Float64},2} = zeros(Float64,n,n);
  split::Int = div(n*(n+1),2);
  result[CartesianIndex.(axes(result,1),axes(result,2))] = v[1:n];
  result[CartesianIndex.((m,n) for n in 2:size(result,2) for m in 1:(n-1))] = sqrt(2)*(v[(n+1):split]+im*v[split+1:n^2]);
  return .5*(result+result');
end

# Outer Product
function outer(z::Union{Array{<:Number,2},Array{<:Number,1}})::Array{<:Number,2}
  return z*z';
end

# Outer Differential
function outerdiff(z::Union{Array{<:Number,2},Array{<:Number,1}},w::Union{Array{<:Number,2},Array{<:Number,1}})::Array{<:Number,2}
  return z*w'+w*z';
end

function unitaryproj(s::Array{Complex{Float64},2})
  return (x->x.U*x.V')(svd(s));
end

############
# Generators
############

# Create random unitary
function genU(n::Int)::Array{Complex{Float64},2}
  dist::Haar = Haar(2);
  return rand(dist::Haar,n);
end

# Create random U2 so that [U1|U2] is unitary
function genU2(U1::Array{Complex{Float64},2},n::Int)::Array{Complex{Float64},2}
  R = rand(n,n-size(U1,2))+im*rand(n,n-size(U1,2));
  return qr(R - U1*U1'*R).Q[:,1:(n-size(U1,2))];
end

# Create random matrix frame of rank r
function genFrame(n::Int,r::Int)::Array{Array{Complex{Float64},2},1}
  m::Int = 4*n*r-4*r^2;
  return [outer((rand(n,r)+rand(n,r)*im)/sqrt(2)) for j=1:m]
end

##########################
# T and R Matrix Functions
##########################

function R_hat(z::Array{Complex{Float64},2},
           f::Array{Array{Complex{Float64},2},1})::Array{Float64,2}
  Indices = collect(1:size(f,1));
  prods = [f[j]*z for j=1:size(f,1)];
  I_z = Indices[prods[Indices] .!= 0.0];
  I0_z = setdiff(Indices, I_z);
  F = upgradeToF(f, rank(z));
  if size(I0_z,1) > 0
    return sum([F[j] for j=I0_z]);
  else
    return similar(F[1], Float64);
  end
end

function T_hat(z::Array{Complex{Float64},2},
               f::Array{Array{Complex{Float64},2},1})::Array{Float64,2}
  Indices = collect(1:size(f,1));
  prods = [f[j]*z for j=1:size(f,1)];
  I_z = Indices[prods[Indices] .!= 0.0];
  I0_z = setdiff(Indices, I_z);
  F = upgradeToF(f, rank(z));
  zhat = fullRankPart(z);
  if size(I_z,1) > 0
    return sum([(1/(mu(zhat)'*F[j]*mu(zhat)))*F[j]*mu(zhat)*mu(zhat)'*F[j] for j=I_z]);
  else
    return similar(F[1], Float64);
  end
end

function A2_hat(z::Array{Complex{Float64},2},
      f::Array{Array{Complex{Float64},2},1}, n::Int64)::Float64
  k = rank(z);
  U, s, V = svd(T_hat(z,f));
  return s[2*n*k-k^2];
end

#######################
# Qhat Matrix Functions
#######################
function Q_hat(z::Array{Complex{Float64},2},
           f::Array{Array{Complex{Float64},2},1})::Array{Complex{Float64},2}
  zhat = fullRankPart(z);
  F = upgradeToF(f, rank(z));
  return sum([(1/(mu(zhat)'*F[j]*mu(zhat)))*F[j]*mu(zhat)*mu(zhat)'*F[j] for j=1:size(f,1)]);
end

function a2_hat(z::Array{Complex{Float64},2},
      f::Array{Array{Complex{Float64},2},1}, n::Int64)::Float64
  k = rank(z);
  U, s, V = svd(Q_hat(z,f));
  return s[2*n*k-k^2];
end

####################
# Q Matrix Functions
####################

# Build Q Matrix
function Q(U::Array{Complex{Float64},2},
           f::Array{Array{Complex{Float64},2},1},
           k::Int)::Array{Complex{Float64},2}
  U1::Array{Complex{Float64},2} = U[:,1:k];
  U2::Array{Complex{Float64},2} = U[:,(k+1):size(U,2)];
  return Q(U1,U2,f)
end

function Q(U1::Array{Complex{Float64},2},
           U2::Array{Complex{Float64},2},
           f::Array{Array{Complex{Float64},2},1})::Array{Complex{Float64},2}
  return real(sum(outer(vcat(tau(U1'*f[j]*U1), mu(U2'*f[j]*U1))) for j=1:size(f,1)));
end

# Smallest eigenvalue of Q Matrix
function lambdaQ(U::Array{Complex{Float64},2},
                 f::Array{Array{Complex{Float64},2},1},
                 k::Int) ::Float64
  return eigmin(Q(U,f,k));
end

function lambdaQ(U1::Array{Complex{Float64},2},
                 U2::Array{Complex{Float64},2},
                 f::Array{Array{Complex{Float64},2},1}) ::Float64
  return eigmin(Q(U1,U2,f));
end

function lambdaQ(f::Array{Array{Complex{Float64},2},1}, n::Int, k::Int) ::Float64
  U = genU(n);
  return eigmin(Q(U, f, k));
end

# Unitary gradient of lambdaQ
function gradLambdaQ(U::Array{Complex{Float64},2},
                     f::Array{Array{Complex{Float64},2},1},
                     n::Int, k::Int)
  e::Array{Float64,1}=eigvecs(Q(U,f,k))[:,1];
  E1::Array{Complex{Float64},2} = tau_i(e[1:k^2],k);
  E2::Array{Complex{Float64},2} = mu_i(e[k^2+1:(2*n*k-k^2)],n-k,k);
  U1::Array{Complex{Float64},2} = U[:,1:k];
  U2::Array{Complex{Float64},2} = U[:,(k+1):n];
  grad = 2*sum(real(tr((E1*U1'+E2'*U2')*f[j]*U1))*f[j] for j=1:size(f,1))*hcat(2*U1*E1+U2*E2, U1*E2')
  return (grad-grad')/2;
end

################
# Test Functions
################

# Test for correct directionality of unitary gradient of lambdaQ
function testGrad(f::Array{Array{Complex{Float64},2},1},n::Int,k::Int,epsilon::Float64)::Float64
  U::Array{Complex{Float64},2}=genU(n);
  Delta::Array{Complex{Float64},2}=gradLambdaQ(U,f,n,k);
  pert::Array{Complex{Float64},2}=U-epsilon*Delta;
  pertproj::Array{Complex{Float64},2}=(x->x.U*x.V')(svd(pert));
  return lambdaQ(pertproj,f,k)-lambdaQ(U,f,k);
end

function testGrad(n::Int,r::Int,k::Int,l::Int,epsilon::Float64)::Array{Float64,1}
  f::Array{Array{Complex{Float64},2},1}=genFrame(n,r);
  return [testGrad(f,n,k,epsilon) for j=1:l]
end

# Test for quadratic error of unitary gradient of lambdaQ
function computeGradError(n::Int,r::Int,k::Int,epsilon::Float64) ::Dict{String, Float64}
  U::Array{Complex{Float64},2} = genU(n);
  f::Array{Array{Complex{Float64},2},1} = genFrame(n,r);
  original = lambdaQ(U,f,k);
  pert = (x->(x-x')/2)(rand(n,n));
  actual = lambdaQ(U+epsilon*pert,f,k);
  approx = original+epsilon*real(tr(gradLambdaQ(U,f,n,k)*pert'));
  return Dict("original"=>original,"actual"=>actual,"approx"=>approx,"error"=>approx-actual);
end

# Play with U2 only to see if result depends on it, U2 -> U2 * U(n-k\times n-k)
function testU2(n::Int, r::Int, k::Int, l::Int)
  f::Array{Array{Complex{Float64},2},1}=genFrame(n,r);
  U=genU(n);
  U1 = U[:,1:k];
  return [lambdaQ(U1, genU2(U1,n),f) for j=1:l];
end

##########################
# Unitary Gradient Descent
##########################

function minimizeLambdaQ(U0::Array{Complex{Float64},2},f::Array{Array{Complex{Float64},2},1},
                         n::Int, k::Int,epsilon0::Float64=.1,factor::Float64=10.)::Float64
  U::Array{Complex{Float64},2}=U0;
  lambda::Float64 = lambdaQ(U0,f,k);
  next::Float64 = lambda;
  epsilon::Float64 = epsilon0;
  machinePrecision::Float64 = 2.0^(-precision(Float64));
  while epsilon>machinePrecision
    U = unitaryproj(U-epsilon*gradLambdaQ(U,f,n,k));
    next = lambdaQ(U,f,k);
    if(next<lambda)
      lambda = next;
    else
      epsilon /= factor;
    end
  end
  return lambda
end

function minimizeLambdaQWithHistory(U0::Array{Complex{Float64},2},f::Array{Array{Complex{Float64},2},1},
                                    n::Int, k::Int,epsilon0::Float64=.1,factor::Float64=10.)::Array{Float64,1}
  U::Array{Complex{Float64},2}=U0;
  lambda::Float64 = lambdaQ(U0,f,k);
  next::Float64 = 0.;
  epsilon::Float64 = epsilon0;
  machinePrecision::Float64 = 2.0^(-precision(Float64));
  history::Array{Float64,1} = [];
  while epsilon> machinePrecision
    push!(history, lambda);
    U = unitaryproj(U-epsilon*gradLambdaQ(U,f,n,k));
    next = lambdaQ(U,f,k);
    if(next<lambda)
      lambda = next;
    else
      epsilon/=factor;
    end
  end
  return history;
end

function minimizeLambdaQ(f::Array{Array{Complex{Float64},2},1},
                         n::Int, k::Int,epsilon0::Float64=.1,factor::Float64=10.)::Float64
  U0::Array{Complex{Float64},2} = genU(n);
  return minimizeLambdaQ(U0,f,n,k,epsilon0,factor);
end

function minimizeLambdaQGlobal(f::Array{Array{Complex{Float64},2},1},
                               n::Int, k::Int,l::Int,epsilon0::Float64=.1,factor::Float64=10.) ::Dict{String, Any}
  data::Array{Float64,1} = zeros(Float64, l);
  Threads.@threads for i=1:l
    data[i] = minimizeLambdaQ(f,n,k,epsilon0,factor);
  end
  return Dict("data"=>data,"min"=>minimum(data));
end

function a0(f::Array{Array{Complex{Float64},2},1}, n::Int, r::Int, l::Int)::Dict{String,<:Number}
  kresults::Array{Float64,1} = [minimizeLambdaQGlobal(f,n,k,l)["min"] for k=1:r];
  return Dict("k"=>argmin(kresults),"a0"=>minimum(kresults));
end

############################
# Smallest Frame Eigenvalues
############################

function frameeigs(n::Int, r::Int)::Array{Float64,1}
  U::Array{Complex{Float64},2} = genU(n)::Array{Complex{Float64},2}
  return frameeigs(r, genFrame(n,r), U);
end

function frameeigs(n::Int,
                   r::Int,
                   f::Array{<:Array{<:Number,2},1})::Array{Float64,1}
  U::Array{Complex{Float64},2} = genU(n)::Array{Complex{Float64},2}
  return frameeigs(r, f, U);
end

function frameeigs(n::Int,
                   r::Int,
                   U::Array{<:Number,2})::Array{Float64,1}
  return frameeigs(r, genFrame(n,r), U);

end

function frameeigs(r::Int,
                   f::Array{<:Array{<:Number,2},1},
                   U::Array{<:Number,2})::Array{Float64,1}
  return [eigmin(Q(U,f,k)) for k=1:r];
end

####################
# Plotting Functions
####################

# Box plot of lambda_{2nk-k^2}(Q_z) for each k=1...r
# Statistics over both frame and unitary
function plot1(n::Int, r::Int, l::Int)
  data::Array{Float64,2} = zeros(Float64,l, r);
  Threads.@threads for i=1:l
    data[i,:] = frameeigs(n,r);
  end
  #data = vcat([frameeigs(n,r)' for i=1:l]...);
  p=Plotly.plot([Plotly.box(y=data[:,i],name="k=$(i)") for i=1:r],Layout(title="lambda_{2nk-k^2}(Q_z) for n=$(n), r=$(r), and l=$(l)"));
  Plotly.post(p,fileopt="overwrite",filename="Lambda Min with n=$(n) and r=$(r)",world_readable=true)
end

# Histogram of the minimizer in k of lambda_{2nk-k^2}(Q_z)
# Statistics over both frame and unitary
function plot2(n::Int, r::Int, l::Int)
  data::Array{UInt8,1} = zeros(UInt8,l);
  Threads.@threads for i=1:l
    data[i] = argmin(frameeigs(n,r));
  end
  #data = [argmin(frameeigs(n,r)) for i=1:l];
  p=Plotly.plot(Plotly.histogram(x=data),Layout(title="Minimizer k for n=$(n), r=$(r), and l=$(l)"));
  Plotly.post(p,fileopt="overwrite",filename="Minimizer in k with n=$(n) and r=$(r)",world_readable=true)
end

# Box plot lambda_{2nk-k^2}(Q_z) for each k=1...r
# Statistics is over the unitary
# Frame is fixed at outset
function plot3(n::Int, r::Int, l::Int)
  f::Array{Array{Complex{Float64},2},1} = genFrame(n,r);
  data::Array{Float64,2} = zeros(Float64, l, r);
  Threads.@threads for i=1:l
    data[i,:] = frameeigs(n, r, f);
  end
  p=Plotly.plot([Plotly.box(y=data[:,i],name="k=$(i)") for i=1:r], Layout(title=L"\lambda_{2nk-k^2}(Q_z)\mbox{ for a fixed frame with } n=%$(n), r=%$(r), and l=%$(l)"));
  Plotly.post(p,fileopt="overwrite",filename="Lambda Min for Fixed Frame with n=$(n) and r=$(r)",world_readable=true)
end

# Histogram of the minimizer in k of lambda_{2nk-k^2}(Q_z)
# Statistics over the unitary
# Frame is fixed at the outset
function plot4(n::Int, r::Int, l::Int)
  f::Array{Array{Complex{Float64},2},1} = genFrame(n,r);
  data::Array{UInt8,1} = zeros(UInt8,l);
  Threads.@threads for i=1:l
    data[i] = argmin(frameeigs(n,r));
  end
  p=Plotly.plot(Plotly.histogram(x=data),Layout(title="Minimizer k for a fixed frame with n=$(n), r=$(r), and l=$(l)"));
  Plotly.post(p,fileopt="overwrite",filename="Minimizer in k for Fixed Frame with n=$(n) and r=$(r)",world_readable=true)
end

# Histogram of the minimizer in k of lambda_{2nk-k^2}(Q_z)
# Unitary minimization is performed first (gradient descent at l2 starting locations)
# Statistics is over frame only (l1 of them)
function plot5(n::Int, r::Int, l1::Int,l2::Int)
  histdata::Array{Int,1} =zeros(l1);
  for i in 1:l1
    f::Array{Array{Complex{Float64},2},1} = genFrame(n,r);
    histdata[i]=a0(f,n,r,l2)["k"];
  end
  p = Plotly.plot(Plotly.histogram(x=histdata),Layout(title="Minimizer k after unitary minimization for n=$(n), r=$(r), and l=$(l1)"))
end

# Track unitary gradient descent from l random points
function plot6(n::Int, r::Int, k::Int, l::Int,epsilon0::Float64=.1,factor::Float64=10.)
  f=genFrame(n,r);
  plots::Array{GenericTrace{Dict{Symbol,Any}},1}=[];
  for i in 1:l
    U0=genU(n);
    history=minimizeLambdaQWithHistory(U0,f,n,k,epsilon0,factor);
    push!(plots,Plotly.scatter(x=[i/size(history,1) for i in 1:size(history,1)],y=history));
  end
  p = Plotly.plot(plots, Layout(title="Unitary gradient descent with n=$(n), r=$(r), k=$(k), and l=$(l)"));
  Plotly.post(p,fileopt="overwrite",filename="Unitary gradient descent with n=$(n), r=$(r), and k=$(k)")
end

# Histogram of A2_hat(z) over random z for a fixed n\times r frame
# Overlay plots
function plot7(n::Int, r::Int, l1::Int, logp::Bool=false, post::Bool=true,bsize::Float64=0.01)
  plotname=logp ? L"\log(1+\hat{A}_2(z))" : L"\hat{A}_2(z)";
  f = genFrame(n,r);
  plots::Array{GenericTrace{Dict{Symbol,Any}},1}=[];
  maxVal::Float64 = 0.;
  for k=1:r
    histdata::Array{Float64, 1} = [A2_hat(rand(n,k)+im*rand(n,k),f[1:(4*n*k-4*k^2)],n) for j=1:l1];
    if logp
      histdata = log.(1 .+ histdata);
    end
    push!(plots,Plotly.histogram(x=histdata,opacity=0.6, xbins_size=bsize, name=L"\mbox{rank}(z)=%$(k)"));
    newMax::Float64 = max(histdata...);
    if newMax > maxVal
      maxVal = newMax;
    end
  end

  p = Plotly.plot(plots, Layout(barmode="overlay", title=replace(plotname*L"\mbox{ for }n=%$(n)\mbox{, }r=%$(r)\mbox{, and }l=%$(l1)\mbox{ random z}","\$\$"=>""), xaxis_title=plotname,yaxis_title=L"n",xaxis_range=[0,maxVal]));
  if post
    Plotly.post(p,fileopt="overwrite",filename="A2_hat(z) with n=$(n) and r=$(r)",world_readable=true)
  else
    p
  end
end

# Histogram of A2_hat(z) over random z for a fixed n\times r frame
# Grid plots
function plot7alt(n::Int, r::Int, l1::Int, logp::Bool=false, post::Bool=true)
  if r%2 == 1
    println("r must be even for this type of plot")
    return
  end
  f = genFrame(n,r);
  global plots = [];
  for k=1:r
    histdata::Array{Float64,1} =[A2_hat(rand(n,k)+im*rand(n,k),f[1:(4*n*k-4*k^2)],n) for j=1:l1];
    if logp
      histdata = log.(1 .+ histdata);
    end
    if k%2 == 1 && k < r-1
      push!(plots, Plotly.plot(Plotly.histogram(x=histdata, name=L"\mbox{rank}(z)=%$(k)"), Layout(yaxis_title=L"n",xaxis_range=[0,max(histdata...)])));
    elseif k%2 == 0 && k < r-1
      push!(plots, Plotly.plot(Plotly.histogram(x=histdata, name=L"\mbox{rank}(z)=%$(k)"), Layout(xaxis_range=[0,max(histdata...)])));
    elseif k%2 == 1 && k >= r-1
      push!(plots, Plotly.plot(Plotly.histogram(x=histdata, name=L"\mbox{rank}(z)=%$(k)"), Layout(xaxis_title=L"\hat{A}_2(z)", yaxis_title=L"n",xaxis_range=[0,max(histdata...)])));
    elseif k%2 ==0 && k >= r-1
      push!(plots, Plotly.plot(Plotly.histogram(x=histdata, name=L"\mbox{rank}(z)=%$(k)"), Layout(xaxis_title=L"\hat{A}_2(z)",xaxis_range=[0,max(histdata...)])));
    end
  end
  sa = ["["]
  for k=1:r
    push!(sa, " plots[$(k)]");
    if k % 2 == 0 && k < r
      push!(sa,";");
    end
  end
  push!(sa," ]")
  s = join(sa);
  p = eval(Meta.parse(s));

  Plotly.relayout!(p, height=500, width=700, title_text=L"\hat{A}_2(z)\mbox{ for }n=%$(n)\mbox{, }r=%$(r)\mbox{, and }l=%$(l1)\mbox{ random z}")
  if post
    Plotly.post(p,fileopt="overwrite", filename="A2_hat(z) with n=$(n) and r=$(r) [grid]",world_readable=true)
  else
    p
  end
end

# Histogram of a2_hat(z) over random z for a fixed n\times r frame
# Overlay plots
function plot8(n::Int, r::Int, l1::Int, logp::Bool=false, post::Bool=true,bsize::Float64=0.01)
  plotname=logp ? L"\log(1+\hat{a}_2(z))" : L"\hat{a}_2(z)";
  f = genFrame(n,r);
  plots::Array{GenericTrace{Dict{Symbol,Any}},1}=[];
  maxVal::Float64 = 0.;
  for k=1:r
    histdata::Array{Float64, 1} = [a2_hat(rand(n,k)+im*rand(n,k),f[1:(4*n*k-4*k^2)],n) for j=1:l1];
    if logp
      histdata = log.(1 .+ histdata);
    end
    push!(plots,Plotly.histogram(x=histdata,opacity=0.6, xbins_size=bsize, name=L"\mbox{rank}(z)=%$(k)"));
    newMax::Float64 = max(histdata...);
    if newMax > maxVal
      maxVal = newMax;
    end
  end
  p = Plotly.plot(plots, Layout(barmode="overlay", title=replace(plotname*L"\mbox{ for }n=%$(n)\mbox{, }r=%$(r)\mbox{, and }l=%$(l1)\mbox{ random z}","\$\$"=>""), xaxis_title=plotname,yaxis_title=L"n",xaxis_range=[0,maxVal]))
  if post
  Plotly.post(p,fileopt="overwrite",filename="littlea2_hat(z) with n=$(n) and r=$(r)",world_readable=true)
  else
    p
  end
end

# Histogram of a2_hat(z) over random z for a fixed n\times r frame
# Grid plots
function plot8alt(n::Int, r::Int, l1::Int, logp::Bool=false, post::Bool=true)
  if r%2 ==1
    println("r must be even for this type of plot")
    return
  end
  f = genFrame(n,r);
  global plots = [];
  for k=1:r
    histdata::Array{Float64,1} =[a2_hat(rand(n,k)+im*rand(n,k),f[1:(4*n*k-4*k^2)],n) for j=1:l1];
    if logp
      histdata = log.(1 .+ histdata);
    end
    if k%2 == 1 && k < r-1
      push!(plots, Plotly.plot(Plotly.histogram(x=histdata, name=L"\mbox{rank}(z)=%$(k)"), Layout(yaxis_title=L"n",xaxis_range=[0,max(histdata...)])));
    elseif k%2 == 0 && k < r-1
      push!(plots, Plotly.plot(Plotly.histogram(x=histdata, name=L"\mbox{rank}(z)=%$(k)"), Layout(xaxis_range=[0,max(histdata...)])));
    elseif k%2 == 1 && k >= r-1
      push!(plots, Plotly.plot(Plotly.histogram(x=histdata, name=L"\mbox{rank}(z)=%$(k)"), Layout(xaxis_title=L"\hat{a}_2(z)", yaxis_title=L"n",xaxis_range=[0,max(histdata...)])));
    elseif k%2 ==0 && k >= r-1
      push!(plots, Plotly.plot(Plotly.histogram(x=histdata, name=L"\mbox{rank}(z)=%$(k)"), Layout(xaxis_title=L"\hat{a}_2(z)",xaxis_range=[0,max(histdata...)])));
    end
  end
  sa = ["["]
  for k=1:r
    push!(sa, " plots[$(k)]");
    if k % 2 == 0 && k < r
      push!(sa,";");
    end
  end
  push!(sa," ]")
  s = join(sa);
  p = eval(Meta.parse(s));

  Plotly.relayout!(p, height=500, width=700, title_text=L"\hat{a}_2(z)\mbox{ for }n=%$(n)\mbox{, }r=%$(r)\mbox{, and }l=%$(l1)\mbox{ random z}")
  if post
    Plotly.post(p,fileopt="overwrite", filename="littlea2_hat(z) with n=$(n) and r=$(r) [grid]",world_readable=true)
  else
    p
  end
end

# Histogram of a(z) over random z for a fixed n\times r frame
# Overlay plots
function plot9(n::Int, r::Int, l1::Int, logp::Bool=false, post::Bool=true,bsize::Float64=0.01)
  plotname=logp ? L"\log(1+a(z))" : L"a(z)";
  f = genFrame(n,r);
  plots::Array{GenericTrace{Dict{Symbol,Any}},1}=[];
  maxVal::Float64 = 0.;
  for k=1:r
    histdata::Array{Float64, 1} = [lambdaQ(f[1:(4*n*k-4*k^2)], n, k) for j=1:l1];
    if logp
      histdata = log.(1 .+ histdata);
    end
    push!(plots,Plotly.histogram(x=histdata,opacity=0.6, xbins_size=bsize, name=L"\mbox{rank}(z)=%$(k)"));
    newMax::Float64 = max(histdata...);
    if newMax > maxVal
      maxVal = newMax;
    end
  end
  p = Plotly.plot(plots, Layout(barmode="overlay", title=replace(plotname*L"\mbox{ for }n=%$(n)\mbox{, }r=%$(r)\mbox{, and }l=%$(l1)\mbox{ random z}","\$\$"=>""), xaxis_title=plotname,yaxis_title=L"n",xaxis_range=[0,maxVal]))
  if post
  Plotly.post(p,fileopt="overwrite",filename="a(z) with n=$(n) and r=$(r)",world_readable=true)
  else
    p
  end
end

# Histogram of a(z) over random z for a fixed n\times r frame
# Grid plots
function plot9alt(n::Int, r::Int, l1::Int, logp::Bool=false, post::Bool=true)
  if r%2 ==1
    println("r must be even for this type of plot")
    return
  end
  f = genFrame(n,r);
  global plots = [];
  for k=1:r
    histdata::Array{Float64,1} =[lambdaQ(f[1:(4*n*k-4*k^2)], n, k) for j=1:l1];
    if logp
      histdata = log.(1 .+ histdata);
    end
    if k%2 == 1 && k < r-1
      push!(plots, Plotly.plot(Plotly.histogram(x=histdata, name=L"\mbox{rank}(z)=%$(k)"), Layout(yaxis_title=L"n",xaxis_range=[0,max(histdata...)])));
    elseif k%2 == 0 && k < r-1
      push!(plots, Plotly.plot(Plotly.histogram(x=histdata, name=L"\mbox{rank}(z)=%$(k)"), Layout(xaxis_range=[0,max(histdata...)])));
    elseif k%2 == 1 && k >= r-1
      push!(plots, Plotly.plot(Plotly.histogram(x=histdata, name=L"\mbox{rank}(z)=%$(k)"), Layout(xaxis_title=L"a(z)", yaxis_title=L"n",xaxis_range=[0,max(histdata...)])));
    elseif k%2 ==0 && k >= r-1
      push!(plots, Plotly.plot(Plotly.histogram(x=histdata, name=L"\mbox{rank}(z)=%$(k)"), Layout(xaxis_title=L"a(z)",xaxis_range=[0,max(histdata...)])));
    end
  end
  sa = ["["]
  for k=1:r
    push!(sa, " plots[$(k)]");
    if k % 2 == 0 && k < r
      push!(sa,";");
    end
  end
  push!(sa," ]")
  s = join(sa);
  p = eval(Meta.parse(s));

  Plotly.relayout!(p, height=500, width=700, title_text=L"a(z)\mbox{ for }n=%$(n)\mbox{, }r=%$(r)\mbox{, and }l=%$(l1)\mbox{ random z}")
  if post
  Plotly.post(p,fileopt="overwrite", filename="a(z) with n=$(n) and r=$(r) [grid]",world_readable=true)
  else
    p
  end
end

end
