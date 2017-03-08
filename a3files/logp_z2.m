function [res, dres] = logp_z2(z2, n, m, model, z1, logs, loga, x, y)
% return the log marginal likelihood wrt z2
smallNum = 1e-6;
z = [z1, z2];
A = a_func(n, m, model, z, x, logs, loga);
R = chol(A+smallNum*eye(m));
% logdet(A)
logdetA = 0;
for i =1:m
    logdetA = logdetA + 2*log(R(i,i));
end

Dpsi = zeros(n,m);
Psi = zeros(n,m);
for i=1:n
    % derivative of Psi wrt z2
    Dpsi(i,2) = exp(-(x(i)-5)^2/z(2)^2)*(-(x(i)-5)^2)*(-2)/z(2)^3;
    Psi(i,:) = model(x(i), z)';    
end
%m_N
mN = 1/exp(logs)*solve_chol(R, Psi'*y);
%E(m_N)
EmN = 1/(2*exp(logs))*norm(y-Psi*mN)^2 +1/(2*exp(logs))*mN'*mN;
% result of log(y|alpha^2, sigma^2)
res = -n/2*log(2*pi) - m/2*loga - n/2*logs - 1/2*logdetA - EmN;

% derivative of mN
dmN = 1/exp(logs)*(-1/exp(logs)*solve_chol(R, Dpsi'*Psi+Psi'*Dpsi)*solve_chol(R, Psi')+solve_chol(R, Dpsi'))*y;
% derivative of log(y|alpha^2, sigma^2)
dres = -1/2/exp(logs)*trace(solve_chol(R, Dpsi'*Psi+Psi'*Dpsi)) - ...
    1/exp(logs)*(-Dpsi*mN-Psi*dmN)'*(y-Psi*mN) -...
    1/2/exp(logs)*(dmN'*mN+mN'*dmN);

end