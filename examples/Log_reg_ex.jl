ENV["JULIA_CUDA_SILENT"] = true

using Flux
using Flux: Data
using Flux: mse
using Flux.Data: DataLoader
using Random
using Printf
using IterTools
using ProgressMeter: @showprogress
using Flux: @epochs

using Plots
using LinearAlgebra

include("PolyFuns.jl")
include("NTKFuns.jl")


# Placing training and test points in grids
# Logistic regression classifier by pulling out the NTK and CK
# from a trained DNN
# Placing kernel optimization inside a function
# Can mislabel some points
# Using the Ktilde's for kernel optimization
# Evolution of errors with training epochs


function LogKer(K_trtr,K_tetr,len1,len0,Its,alp0)


Ns = Int(len1 + len0);

Ktilde = [K_trtr[1:len1,:] ; -K_trtr[len1+1:end,:]];

LossF(alpha) = sum(log.(1 .+ exp.(-Ktilde*alpha)));
Fun(alpha) = -Ktilde'*(1 ./(1 .+ exp.(Ktilde*alpha)));
DFun(alpha) = Ktilde'.*(exp.(Ktilde*alpha)./(1 .+ exp.(Ktilde*alpha)).^2)'*Ktilde;

Loss_vals = zeros(3,Its+1);

alpha0 = (2*rand(Ns,1) .- 1)*alp0;
Loss_vals[1,1] = maximum(abs.(Fun(alpha0)));

l_stop = Its;

for l = 1:Its
    

	if mod(l,1000) == 0
		println("$l")
	end
	
	# alpha1 = alpha0 - DFun(alpha0)\Fun(alpha0);
	alpha1 = alpha0 - pinv(DFun(alpha0))*Fun(alpha0);

	diff_F = abs.(Fun(alpha1));
	diff_alp = abs.(alpha1-alpha0); 
	diff_Kalp = abs.(Ktilde*(alpha1-alpha0));
	
	Loss_vals[:,l+1] = [maximum(diff_F); maximum(diff_alp) ;  maximum(diff_Kalp)];

	if (maximum(diff_F) < 1.0e-12) || (maximum(diff_alp) < 1.0e-12) || (maximum(diff_Kalp) < 1.0e-12) 
		l_stop = l;
		break
	end
	
	alpha0 = alpha1;


end


Kpreds_tr = 1 ./(1 .+ exp.(-K_trtr*alpha0));
Kpreds_te = 1 ./(1 .+ exp.(-K_tetr*alpha0));

return Kpreds_tr, Kpreds_te, Loss_vals, l_stop

end

L = [-1 1 ; -1 1];
Its = 12;

D = 3; # number of layers in the NN
Wid = 2^7; # width of each layer in the NN

Ns_vals = [11 7];
tau_vals = [2 3];

Ns_test_vals = Int.(Ns_vals.*tau_vals);

Ns_total = Int(prod(Ns_vals .+ 1));
Ns_test_total = Int(prod(Ns_test_vals .+ 1));
bsz = Ns_total; # batchsize

runs = 10;
stages = 100;
num_epochs = 4000;
showind = Int(num_epochs/stages);

gams = [1 3 -4]/5;
# gams = [-0.1 0.97 0.3 -1]*2;

mlbl = 0;
train_thresh = 0.99 - 3*mlbl/Ns_total;

NTKa0 = 1.0e-3;
CKa0 = 1.0e-3;

# Choice of activation function and learning rates
# sig(x) = relu.(x); sig_p(x) = (sign.(x) .+ 1.0)/2; lea_rt = 1e-5;
sig(x) = tanh.(x); sig_p(x) = (sech.(x)).^2; lea_rt = 1e-5;

ErrVals = zeros(6,stages,runs);
AccVals = zeros(6,stages,runs);


xv_tr_1 = range(L[1,1], L[1,2] , Ns_vals[1]+1);
xv_tr_2 = range(L[2,1], L[2,2] , Ns_vals[2]+1);

xv_tr = zeros(Ns_total,2);
for ii = 1:Ns_vals[1]+1
	xv_tr[(ii-1)*(Ns_vals[2]+1)+1:(ii)*(Ns_vals[2]+1),:] = [xv_tr_1[ii]*ones(Ns_vals[2]+1,1) xv_tr_2];
end

xv_te_1 = range(L[1,1], L[1,2] , Ns_test_vals[1]+1);
xv_te_2 = range(L[2,1], L[2,2] , Ns_test_vals[2]+1);

xv_te = zeros(Ns_test_total,2);
for ii = 1:Ns_test_vals[1]+1
	xv_te[(ii-1)*(Ns_test_vals[2]+1)+1:(ii)*(Ns_test_vals[2]+1),:] = [xv_te_1[ii]*ones(Ns_test_vals[2]+1,1) xv_te_2];
end


yv_tr = (sign.(xv_tr[:,2] - Horner(xv_tr[:,1],gams)) .+ 1)/2;
yv_te = (sign.(xv_te[:,2] - Horner(xv_te[:,1],gams)) .+ 1)/2;


yv_tr[1:mlbl] = 1 .- yv_tr[1:mlbl];

xv1 = xv_tr[iszero.(1 .- yv_tr),:];
xv0 = xv_tr[iszero.(yv_tr),:];

len1 = size(xv1)[1]; len0 = size(xv0)[1];

Xc = [xv1' xv0'];
Yc = [ones(1,len1) zeros(1,len0) ; zeros(1,len1) ones(1,len0)];

zv_tr = [yv_tr' ; 1 .- yv_tr'];
zv_te = [yv_te' ; 1 .- yv_te'];

opt = ADAM(lea_rt);

# Setting up the NN
neurons = cat((2,Wid),[(Wid,Wid) for i in 1:D-2],(Wid,2),dims = (1,1));
actvs = cat([sig for i in 1:D-1],identity,dims = (1,1));



function loss0(x, y, Model)
	y_mo = Model(x);
	Flux.crossentropy(y_mo, y);
end

loss(x,y,Model) = Flux.mse(Model(x),y) # Loss function

train_loader = DataLoader((xv_tr',zv_tr), batchsize=bsz); # Placing the training data into batches

@showprogress 1 "Training the model..." for rr = 1:runs

layers = [Dense(neurons[i][1],neurons[i][2],actvs[i]) for i in 1:D];
Mo = Chain(layers...,softmax)|>f64;
par = Flux.params(Mo);


for ii = 1:num_epochs
	
for (x,y) in train_loader
	Flux.train!((x, y) -> loss(x, y, Mo), par, [(x,y)], opt)
end

	
train_acc0 = sum((Mo(xv_tr') .>= 0.5).*zv_tr)/Ns_total;
test_acc0 = sum((Mo(xv_te') .>= 0.5).*zv_te)/Ns_test_total;

train_CE0 = Flux.crossentropy(Mo(xv_tr') , zv_tr);
test_CE0 = Flux.crossentropy(Mo(xv_te') , zv_te);

if train_acc0 >= train_thresh
	println("\nHit training accuracy threshold at epoch $ii")
	break
end


if ii%showind == 0

ll = Int(ii/showind);

println("\nrr = $rr out of $runs, ii = $ii out of $num_epochs")
				
println("\nTrain Err $train_CE0 \t Test Err $test_CE0")
println("Train Acc $train_acc0 \t Test Acc $test_acc0")


pred_train = Mo(xv_tr');
train_acc = sum((Mo(xv_tr') .>= 0.5).*zv_tr)/Ns_total;
train_CE = Flux.crossentropy(pred_train , zv_tr);


pred_test = Mo(xv_te');
test_acc = sum((Mo(xv_te') .>= 0.5).*zv_te)/Ns_test_total;
test_CE = Flux.crossentropy(pred_test , zv_te);

println("\nNN Train Err $train_CE")
println("NN Test Err $test_CE")

println("\nNN Train Acc $train_acc")
println("NN Test Acc $test_acc")

# Same NN as Mo except for the last layer
Mo1 = Chain(layers[1:D-1])|>f64;;
par1 = Flux.params(Mo1);
for j = 1:2*(D-1)
	par1[j] .= par[j];
end

# Output of last hidden layer at training inputs, with a row of ones -- involved in CK
A_tr = [Mo1(Xc) ; ones(1,Ns_total)];
A_te = [Mo1(xv_te') ; ones(1,Ns_test_total)];

CK_trtr = A_tr'*A_tr;
CK_tetr = A_te'*A_tr;


par_red = vcat(par[1:2*(D-1)] , [(par[2*D-1][1,:] - par[2*D-1][2,:])'] ,  [par[2*D][1] - par[2*D][2]] );

NTK_trtr =  NTKCal(sig,sig_p,par_red,[Xc , Xc],false);
NTK_tetr = NTKCal(sig,sig_p,par_red,[xv_te' , Xc],false);

println("\nNTK iterations")
NTKpreds_tr, NTKpreds_te, NTK_Loss_vals, NTK_l_stop = LogKer(NTK_trtr,NTK_tetr,len1,len0,Its,NTKa0);

NTKtrain_acc = (sum((NTKpreds_tr .>= 0.5).*Yc[1,:]) + sum((NTKpreds_tr .< 0.5).*Yc[2,:]))/Ns_total;
NTKtrain_CE = Flux.crossentropy([NTKpreds_tr (1 .- NTKpreds_tr)]' , Yc);

NTKtest_acc = (sum((NTKpreds_te .>= 0.5).*zv_te[1,:]) + sum((NTKpreds_te .< 0.5).*zv_te[2,:]))/Ns_test_total;
NTKtest_CE = Flux.crossentropy([NTKpreds_te (1 .- NTKpreds_te)]' , zv_te);

println("\nNTK Train Err $NTKtrain_CE")
println("NTK Test Err $NTKtest_CE")
println("\nNTK Train Acc $NTKtrain_acc")
println("NTK Test Acc $NTKtest_acc")


println("\nCK iterations")
CKpreds_tr, CKpreds_te, CK_Loss_vals, CK_l_stop = LogKer(CK_trtr,CK_tetr,len1,len0,Its,CKa0);

CKtrain_acc = (sum((CKpreds_tr .>= 0.5).*Yc[1,:]) + sum((CKpreds_tr .< 0.5).*Yc[2,:]))/Ns_total;
CKtrain_CE = Flux.crossentropy([CKpreds_tr (1 .- CKpreds_tr)]' , Yc);

CKtest_acc = (sum((CKpreds_te .>= 0.5).*zv_te[1,:]) + sum((CKpreds_te .< 0.5).*zv_te[2,:]))/Ns_test_total;
CKtest_CE = Flux.crossentropy([CKpreds_te (1 .- CKpreds_te)]' , zv_te);



println("\nCK Train Err $CKtrain_CE")
println("CK Test Err $CKtest_CE")
println("\nCK Train Acc $CKtrain_acc")
println("CK Test Acc $CKtest_acc")


ErrVals[:,ll,rr] = [train_CE ; test_CE ; CKtrain_CE ; NTKtrain_CE ; CKtest_CE ; NTKtest_CE];
AccVals[:,ll,rr] = [train_acc ; test_acc ; CKtrain_acc ; NTKtrain_acc ; CKtest_acc ; NTKtest_acc];


end


end

end

Errs = sum(ErrVals,dims=3)/runs;
Accs = sum(AccVals,dims=3)/runs;


plot(range(0,num_epochs,stages+1)[2:end],Errs[2,:],label = "NN"); plot!(range(0,num_epochs,stages+1)[2:end],Errs[5,:],label = "CK"); plot!(range(0,num_epochs,stages+1)[2:end],Errs[6,:],yaxis =:log, label = "NTK", xlab = "epochs", ylab = "test CE")


# plot(range(0,num_epochs,stages+1)[2:end],Accs[2,:],label = "NN"); plot!(range(0,num_epochs,stages+1)[2:end],Accs[5,:],label = "CK"); plot!(range(0,num_epochs,stages+1)[2:end],Accs[6,:],label = "NTK", xlab = "epochs", ylab = "test accuracy")



