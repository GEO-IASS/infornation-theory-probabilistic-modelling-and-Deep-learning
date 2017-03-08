function [res, dres] = logp_alpha(loga, n, m, model, z, logs, x, y)
% return the log marginal likelihood wrt alpha^2
% to use unconstrained optimization we will assume alpha2=exp(loga), where
% loga is a new parameter
smallNum = 1e-6;
A = a_func(n, m, model, z, x, logs, loga);
R = chol(A+smallNum*eye(m));
% logdet(A)
logdetA = 0;
for i =1:m
    logdetA = logdetA + 2*log(R(i,i));
end

Psi = zeros(n,m);
for i = 1:n
    Psi(i,:) = model(x(i), z)';
end
%m_N
mN = 1/exp(logs)* solve_chol(R, Psi'*y);
%E(m_N)
EmN = 1/(2*exp(logs))*norm(y-Psi*mN)^2 +1/(2*exp(logs))*mN'*mN;
% result of log(y|alpha^2, sigma^2)
res = -n/2*log(2*pi) - m/2*loga - n/2*logs - 1/2*logdetA - EmN;
% derivative of mN
dmN = 1/exp(logs+2*loga)*solve_chol(R, solve_chol(R, Psi'*y));

% derivative of log(y|alpha^2, sigma^2)
dres = (-m/2/exp(loga)-1/2*trace(-1/exp(2*loga)*solve_chol(R, eye(m)))...
    - 1/exp(logs)*(-Psi*dmN)'*(y-Psi*mN)...
    - 1/2/exp(logs)*(dmN'*mN+mN'*dmN))*exp(loga);

end