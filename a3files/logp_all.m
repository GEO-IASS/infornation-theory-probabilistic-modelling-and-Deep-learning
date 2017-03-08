function [res, dres] = logp_all(par, n, m, model_num, x, y)
% return the log marginal likelihood wrt alpha^2

if model_num==1
    model = @m1;
elseif model_num==2
    model = @m2;
elseif model_num==3
    model = @m3;
else
    warning('wrong model number');
end

loga = par(1);
logs = par(2);
z1 = par(3);
z2 = par(4);
z = [z1,z2];

[a, dalpha] = logp_alpha(loga, n, m, model, z, logs, x, y);
[a, dsigma] = logp_sigma(logs, n, m, model, z, loga, x, y);
dz1 = 0;
dz2 = 0;
if isequal(model, @m2)
    [a, dz1] = logp_z1(z1, n, m, model, z2, logs, loga, x, y);
    [a, dz2] = logp_z2(z2, n, m, model, z1, logs, loga, x, y);
end
res = -a;
dres = [-dalpha; -dsigma; -dz1; -dz2];
%--------------------functions models---------------------------------
    function y = m1(x, z)
        %model 1
        %m = 6
        y = [1; x; x^2; x^3; x^4; x^5];
    end

    function y = m2(x, z)
        % model 2
        %m = 2
        y = [exp(-(x-1)^2/z(1)^2); exp(-(x-5)^2/z(2)^2)];
    end

    function y = m3(x,z)
        % model 3
        %m = 2;
        y = [x; cos(2*x)];
    end

end