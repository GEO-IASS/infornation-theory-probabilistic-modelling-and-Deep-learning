function [res, dres] = logp_sigma(logs, n, m, model, z, loga, x, y)
% return the log marginal likelihood wrt sigma^2
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
mN = 1/exp(logs)*solve_chol(R, Psi'*y);
%E(m_N)
EmN = 1/(2*exp(logs))*norm(y-Psi*mN)^2 +1/(2*exp(logs))*mN'*mN;
% result of log(y|alpha^2, sigma^2)
res = -n/2*log(2*pi) - m/2*loga - n/2*logs - 1/2*logdetA - EmN;
% derivative of mN
dmN = -1/exp(2*logs)*solve_chol(R, Psi'*y) + 1/exp(3*logs)*solve_chol(R, Psi'*Psi* solve_chol(R, Psi'*y));

% derivative of log(y|alpha^2, sigma^2)
dres = (    -n/2/exp(logs) + 1/2/exp(2*logs)*trace(solve_chol(R, Psi'*Psi)) +...
    1/2/exp(2*logs)*norm(y-Psi*mN)^2 ...
    - 1/exp(logs)*(-Psi*dmN)'*(y-Psi*mN) +...
    +1/2/exp(2*logs)*mN'*mN - 1/2/exp(logs)*(dmN'*dmN+mN'*dmN)     )*exp(logs);

end