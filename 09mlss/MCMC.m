function MCMC()
% HW#3; question 4
close('all');
data = load('astro_data');
x = data.xx;
y = data.vv;


%Initialization
% inital position
m = mean(x);
% initial frequency
log_omega = log(mean(abs(y)) / mean(x));
%weight of normal distribussions in the mix
weight = rand;
% parameters
% of the first distribution
log_var1 = 0;
mean1 = log(m*rand);
% of the 2nd distribution
log_var2 = 0;
mean2 = log(m*rand);

array = [log_omega; m; weight; mean1; mean2; log_var1; log_var2];

% function that takes inputs 'array'
wrapper = @(args) log_pstar(args{:}, x, y);
logdist = @(x) wrapper(num2cell(x));

% # of itarations
iter = 1e4;
% Metropolis’s step-size parameter
step = 0.001;
%metropolis algorithm
[samples, accepts] = dumb_metropolis(array, logdist, iter, step);
%# of the initial samples that we will not count
free_going = 1000;
% % of rejection
accepts

%Answers
omega_s = exp(samples(1,free_going:end));
omega = mean(omega_s)
m = mean(samples(2,free_going:end))
%errors
error_of_omega = std(omega_s)
error_of_m = std(samples(2,free_going:end))

% we plot the samples from begining but we've burn first 1000 samples
figure
hist(omega_s, 50);
title('approximation to the marginal posterior of omega');
figure
hist(samples(2,free_going:end), 50);
title('approximation to the marginal posterior of m');
figure
plot(samples(1,:), samples(2,:));
title('the time series m against omega');
figure
plot(samples(1,:));
title('the time series of m');
    function [samples, arate] = dumb_metropolis(init, log_ptilde, iters, sigma, varargin)
        %DUMB_METROPOLIS explore an unnormalized distribution. Eg code for tutorial
        % Iain Murray, September 2009
        
        D = numel(init);
        samples = zeros(D, iters);
        
        arate = 0;
        state = init;
        Lp_state = log_ptilde(state, varargin{:});
        for ss = 1:iters
            % Propose
            prop = state + sigma*randn(size(state));
            Lp_prop = log_ptilde(prop, varargin{:});
            if log(rand) < (Lp_prop - Lp_state)
                % Accept
                arate = arate + 1;
                state = prop;
                Lp_state = Lp_prop;
            end
            samples(:, ss) = state(:);
        end
        arate = arate/iters;
    end
end
