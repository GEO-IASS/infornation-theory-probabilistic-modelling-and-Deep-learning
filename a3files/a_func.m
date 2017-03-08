function psipsit = a_func(n, m, model, z, x, logs, loga)
% calculation of covariance matrix
% to use unconstrained optimization we will assume alpha2=exp(loga), where
% loga is a new parameter
Psi = zeros(n,m);
for i = 1:n
    Psi(i,:) = model(x(i), z)';
end
psipsit = 1/exp(loga)*eye(m) + 1/exp(logs) * Psi'*Psi;
end