function [evid,par] = HW3_part3(model_num, file_num)
%return the evidence of a model

% model_num is a number of the model
% file_num is a number of the dataset
% evid is the evidence

%file_num=1;
% choose the model
%model_num = 1;

if file_num==1
    file = matfile('occam1.mat');
else
    file = matfile('occam2.mat');
end
x = file.x;
%x = x(1:1000);
y = file.y;
%y =y(1:1000);
%n = 1000;%file2.N;
n = file.N;
smallNum = 1e-6;


% model is a presise function
% m is a number of psi_i
switch model_num
    case 1
        m = 6;
        model = @m1;
    case 2
        m = 2;
        model = @m2;
    case 3
        m = 2;
        model = @m3;
    otherwise
        warning('wrong number of model');
end

% exp(loga) = alpha2 = alpha^2; exp(logs) = sigma2 = sigma^2;
loga = 11;
logs = 10;
z1 = 10;
z2 = 10;
z = [z1, z2];

% check the derivatives
[d_alpha dy dh] = checkgrad('logp_alpha', loga, smallNum, n, m, model, z, logs, x, y);
[d_sigma dy dh] = checkgrad('logp_sigma', logs, smallNum, n, m, model, z, loga, x, y);

d_z1 = 0;
d_z2 = 0;
if isequal(model, @m2)
    [d_z1 dy dh] = checkgrad('logp_z1', z1, smallNum, n, m, model, z2, logs, loga, x, y);
    [d_z2 dy dh] = checkgrad('logp_z2', z2, smallNum, n, m, model, z1, logs, loga, x, y);
end

%check derivatives
if d_alpha>0.001 || d_sigma>0.001 || d_z1>0.001 || d_z2>0.001
    warning('derivatives do not coincide');
end
% gradient descent
[X, fX, i] = minimize([-5; -4; 1; 1], 'logp_all', 100, n, m, model_num, x, y );

alpha2 = exp(X(1))
sigma2 = exp(X(2))
z1 = X(3);
z2 = X(4);
if isequal(model, @m2)
    par = [alpha2;sigma2;z1;z2];
else
    par = [alpha2;sigma2;0;0];
end
evid = evid_prob(n, m, model, z, sigma2, alpha2, x, y);
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