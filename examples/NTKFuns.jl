using LinearAlgebra

# Computes the X[1]--X[2] NTK given a NN with activation function sig, its derivative sig_p, parameters par, and inputs [X[1],X[2]]

function NTKCal(sig,sig_p,par,X,fl)

L = Int(length(par)/2);
s1 = size(X[1])[2]; s2 = size(X[2])[2];

Yvs = Array{Any,2}(undef,2,L);
Xvs = Array{Any}(undef,L);


Xvs[1] = X[1]'*X[2];
Y = [(par[1]*X[1] .+ par[2]) , (par[1]*X[2] .+ par[2])]; 

for k = 1:2
	Yvs[k,1] = sig_p.(Y[k]);
end


for l = 2:L
	Xvs[l] = sig.(Y[1])'*sig.(Y[2]);
    
 	for k = 1:2
		Y[k] = par[2*l-1]*(sig.(Y[k])) .+ par[2*l];
		Yvs[k,l] = sig_p.(Y[k]);
	end

end

Zs1 = Array{Any}(undef,s1); Zs2 = Array{Any}(undef,s2);

if fl
	Kvs = Array{Matrix{Float64}}(undef,L);
end

K = Xvs[L] .+ 1;
for ii = 1:s1
	Zs1[ii] = 1.0;
end

for jj = 1:s2
	Zs2[jj] = 1.0;
end

if fl
	Kvs[L] = 0 .+ K;
end

for l = L:-1:2
	for ii = 1:s1
		Zs1[ii] = Zs1[ii]*(par[2*l-1].*Yvs[1,l-1][:,ii]');
	end

	for jj = 1:s2
		Zs2[jj] = Zs2[jj]*(par[2*l-1].*Yvs[2,l-1][:,jj]');

		for ii = 1:s1
			K[ii,jj] = K[ii,jj] + (Xvs[l-1][ii,jj] + 1)*sum(Zs1[ii].*Zs2[jj]);
		end
	end

	if fl
		Kvs[l-1] = 0 .+ K;
	end
		
end

if fl
	return Kvs
else
	return K
end



end



