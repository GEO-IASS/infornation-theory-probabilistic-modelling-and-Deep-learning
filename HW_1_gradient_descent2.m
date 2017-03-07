function HW_1_gradient_descent2()

par_prev = [2;2];

% training data
x = [2, 7, 5, 11, 14];
y = [19, 62, 37, 94, 120];

%sample size
n = 5;
% step

step = 0.00001;
par_new = [1;1]; %initialization x=[sigma^2; gamma^2]
while norm(par_prev - par_new)>0.00001
    
    par_prev = par_new;
    
    jacob = GF (n, x, y, par_prev);
    
    DF = jacob' * [G_1(n, x, y, par_prev); G_2(n, x, y, par_prev)];
    
    par_new = par_prev - step*DF

end

%%%%%%%%%%%%%%%%%%%%%%functions%%%%%%%%%%%%%%%%%%%%%

    function res = G_1(n, x, y, par)
        sum = 0;
        for i=1:n
            sum = sum + 1 / ( (par(1)) + x(i)^2*(par(2)) ) - y(i)^2 / ( (par(1)) + x(i)^2*(par(2)) )^2;
        end
        res = sum;
    end

    function res = G_2(n, x, y, par)
        sum = 0;
        for i=1:n
            sum = sum + x(i)^2 / ( (par(1)) + x(i)^2*(par(2)) ) - y(i)^2 * x(i)^2 / ( (par(1)) + x(i)^2*(par(2)) )^2;
        end
        res = sum;
    end

    function res = GF (n, x, y, par)
        %Jacobian matrix
        res = zeros(2);
        for i=1:n
            res(1,1) = res(1,1) - 1 / ( (par(1)) + x(i)^2*(par(2)) )^2 ...
                + 2 * y(i)^2   / ( (par(1)) + x(i)^2*(par(2)) )^3;
            
            res(1,2) = res(1,2) - 1 * x(i)^2 / ( (par(1)) + x(i)^2*(par(2)) )^2 ...
                + 2 * y(i)^2 * x(i)^2  / ( (par(1)) + x(i)^2*(par(2)) )^3;
            
            res(2,1) = res(2,1) -  x(i)^2 / ( (par(1)) + x(i)^2*(par(2)) )^2 ...
                + 2 * y(i)^2 * x(i)^2 * (par(1)) / ( (par(1)) + x(i)^2*(par(2)) )^3;
            
            res(2,2) = res(2,2) -  x(i)^4 / ( (par(1)) + x(i)^2*(par(2)) )^2 ...
                + 2 * y(i)^2 * x(i)^4 * (par(2)) / ( (par(1)) + x(i)^2*(par(2)) )^3;
            
        end
    end

end