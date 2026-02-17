using LinearAlgebra


function gauss_grid(Nq)

A = zeros(Nq,Nq);
for i = 1:Nq-1
    bv = i^2/(4*i^2 - 1);
    A[i,i+1] = sqrt(bv);
    A[i+1,i] = A[i,i+1];
end

X , V = eigen(A);

W = V[1,:]; 
W = 2*W.^2;

return X,W

end




function Horner(z,A)

n = length(A) - 1;
p = 0*z .+ A[n+1];

for i = n:-1:1
    p = A[i] .+ z.*p;
end

return p

end




