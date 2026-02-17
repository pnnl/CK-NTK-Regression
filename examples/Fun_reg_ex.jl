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
using LaTeXStrings
using LinearAlgebra


include("PolyFuns.jl")
include("NTKFuns.jl")


# Building NNs for function regression and comparing with NTK/CK performance
# Evolution of errors with training epochs

f(x) = exp.(sin.(2*pi*x));
# f(x) = exp.(3*x);
# f(x) = cos.(exp.(3*x));


# f(x) = sin.(cos.(pi*x));
# f(x) = sin.(pi*x);
# f(x) = x.^3;
# f(x) = x.^2
# f(x) = 2*x;
# f(x) = cos.(x.^2);
# f(x) = cos.(x.^2);
# f(x) = exp.(-x.^2);
# f(x) = exp.(3*x);
# f(x) = cos.(exp.(3*x));



D = 4; 
Wid = 2^7;
Ns = 200;
Ns_test = Int(3*Ns);

lam = 0.0;

bsz = Int(Ns);

stages = 40;
num_epochs = 1200;
showind = Int(num_epochs/stages);

ErrVals = zeros(10,stages);

sig(x) = relu.(x); sig_p(x) = (sign.(x) .+ 1.0)/2; lea_rt = 1e-3
# sig(x) = tanh.(x); sig_p(x) = (sech.(x)).^2; lea_rt = 5e-3;

opt = ADAM(lea_rt);


mag = false;

neurons = cat((1,Wid),[(Wid,Wid) for i in 1:D-2],(Wid,1),dims = (1,1));
actvs = cat([sig for i in 1:D-1],identity,dims = (1,1));

 
layers = [Dense(neurons[i][1],neurons[i][2],actvs[i]) for i in 1:D];

Mo = Chain(layers...)|>f64;

# Xtrain = range(-1,stop = 1, length = Ns+1); 
# Xtest = range(-1,stop = 1, length = Ns_test+1);

Xtrain = range(-1,stop = 1-2/Ns, length = Ns); 
Xtest = range(-1,stop = 1-2/Ns_test, length = Ns_test);

# Xtrain,~ = gauss_grid(Ns);
# Xtest,W2 = gauss_grid(Ns_test);

len_tr = length(Xtrain); len_te = length(Xtest);
Xtrain = reshape(Xtrain,1,len_tr); Xtest = reshape(Xtest,1,len_te); 

Ytrain = f(Xtrain); Ytest = f(Xtest);

Y0 = Mo(Xtrain);

loss(x,y,Model) = Flux.mse(Model(x),y)
par = Flux.params(Mo);

train_loader = DataLoader((Xtrain,Ytrain), batchsize=bsz);

@showprogress 1 "Training the model..." for ii = 1:num_epochs
	for (x,y) in train_loader
		Flux.train!((x, y) -> loss(x, y, Mo), par, [(x,y)], opt)

	end

	if ii%showind == 0

		ll = Int(ii/showind);
		
		diff_train = Mo(Xtrain) - Ytrain;
		diff_test = Mo(Xtest) - Ytest;

		train_MSE = sqrt(sum(diff_train.^2)/Ns);
		test_MSE = sqrt(sum(diff_test.^2)/Ns_test);

		println("\nTrain MSE $train_MSE")
		println("Test MSE $test_MSE")
		
		
		diff0 = sqrt(sum((Y0 - Mo(Xtrain)).^2)/Ns);


		Mo1 = Chain(layers[1:D-1])|>f64;;
		par1 = Flux.params(Mo1);
		for j = 1:2*(D-1)
			par1[j] .= par[j];
		end

		A = [Mo1(Xtrain) ; ones(1,len_tr)];
		r = rank(A);
		U,S,V = svd(A);
		U = U[:,1:r]; V = V[:,1:r];

		Mo2 = Chain(layers[1:D-2])|>f64;;
		par2 = Flux.params(Mo2);
		for j = 1:2*(D-2)
			par2[j] .= par[j];
		end


		A_test = Mo1(Xtest);
		CK_tete = [A_test ; ones(1,len_te)]'*[A_test ; ones(1,len_te)];

		Y_CKNN = [par[2*D-1] par[2*D]]*(U*U')*[A_test ; ones(1,len_te)];
		Y_CKNN0 = Mo(Xtrain)*pinv(A'*A)*(A'*[A_test ; ones(1,len_te)]);
		Y_CK = Ytrain*pinv(A'*A)*(A'*[A_test ; ones(1,len_te)]);

		# println("$(maximum(abs.(Y_CKNN - Y_CKNN0))) \t $(maximum(abs.(Y_CKNN - Y_CK)))")


		Km =  NTKCal(sig,sig_p,par,[Xtest , Xtest],true);
		K_tete = Km[1];

		Kmid = K_tete[1:3:end,1:3:end];
		K_te = K_tete[1:3:end,:];


		KC_tete = Km[D-1] - Km[D];
		KC = KC_tete[1:3:end,1:3:end];
		KC_te = KC_tete[1:3:end,:];


		# Y_NTK_NN =  Mo(Xtrain)*pinv(Kmid)*K_te; Y_NTK = Ytrain*pinv(Kmid)*K_te;
		Y_NTK_NN =  Mo(Xtrain)*pinv(Kmid+lam*I)*K_te; 
		Y_NTK = Ytrain*pinv(Kmid+lam*I)*K_te;

		Y_KC = Ytrain*pinv(KC+lam*I)*KC_te;

		r2 = rank(Kmid);
		Wst,Dv,W = svd(Kmid);
		W = W[:,1:r2];


		r4 = rank(KC);
		Wst4,Dv4,W4 = svd(KC);
		W4 = W4[:,1:r4];



		diff_CKNN = Mo(Xtest) - Y_CKNN0;
		diff_CK_train = Ytrain - Ytrain*pinv(A'*A)*(A'*A); #Ytrain*(V*V');
		diff_CK = Ytest - Y_CK;  

		diff_CK_train_P = diff_CK_train*pinv(Kmid)*Kmid;

		CKNN_MSE = sqrt(sum(diff_CKNN.^2)/Ns_test);
		CK_train_MSE = sqrt(sum(diff_CK_train.^2)/Ns);
		CK_MSE = sqrt(sum(diff_CK.^2)/Ns_test);

		diff_NTK_NN = Mo(Xtest) - Y_NTK_NN;
		diff_NTK_train = Ytrain - Ytrain*(W*W');
		diff_NTK = Ytest - Y_NTK;  

		NTK_NN_MSE = sqrt(sum(diff_NTK_NN.^2)/Ns_test);
		NTK_train_MSE = sqrt(sum(diff_NTK_train.^2)/Ns);
		NTK_MSE = sqrt(sum(diff_NTK.^2)/Ns_test);

		diff_KC = Ytest - Y_KC;  
		KC_MSE = sqrt(sum(diff_KC.^2)/Ns_test);


		V_te = U'*[A_test ; ones(1,len_te)]./S[1:r]; V_te = V_te';
		W_te = W'*K_te./Dv[1:r2]; W_te = W_te';
		W4_te = W4'*KC_te./Dv4[1:r4]; W4_te = W4_te';

		println("$CK_MSE \t $NTK_MSE")


		ErrVals[1:8,ll] = [train_MSE ; test_MSE ; CKNN_MSE ; NTK_NN_MSE ; CK_train_MSE ; NTK_train_MSE ; CK_MSE ; NTK_MSE];
	ErrVals[9:10,ll] = [norm(V_te*V'*transpose(diff_train)) ; norm(W_te*W'*transpose(diff_train))]/sqrt(Ns_test);


	end 
end


plot(range(0,num_epochs,stages+1)[2:end],ErrVals[2,:],label = "NN"); 
plot!(range(0,num_epochs,stages+1)[2:end],ErrVals[7,:],label = "CK"); 
plot!(range(0,num_epochs,stages+1)[2:end],ErrVals[8,:],label = "NTK",yaxis=:log, xlab = "epochs", ylab = "test errors")





