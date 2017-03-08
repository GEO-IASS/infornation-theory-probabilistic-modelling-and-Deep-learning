function res = evid_prob(n, m, model, z, sigma2, alpha2, x, y)

% return the log marginal likelihood wrt z2
smallNum = 1e-6;
Psi = zeros(n,m);
for i=1:n

    Psi(i,:) = model(x(i), z)';    
end
A = 1/alpha2*eye(m)+1/sigma2*Psi'*Psi;
R = chol(A+smallNum*eye(m));
% logdet(A)
logdetA = 0;
for i =1:m
    logdetA = logdetA + 2*log(R(i,i));
end

%m_N
mN = 1/sigma2*solve_chol(R, Psi'*y);
%E(m_N)
EmN = 1/(2*sigma2)*norm(y-Psi*mN)^2 +1/(2*sigma2)*mN'*mN;
% result of log(y|alpha^2, sigma^2)
res = -n/2*log(2*pi) - m/2*log(alpha2) - n/2*log(sigma2) - 1/2*logdetA-EmN;
end