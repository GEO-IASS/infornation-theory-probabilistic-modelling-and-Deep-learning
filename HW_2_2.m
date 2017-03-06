function HW_2_2()
%-----initialization----------
% # of variables
N = 8;
% # of factors
K = 7;

% the graph matrix
G = zeros(N,K);
G(1,1) = 1;
G(2,1) = 1; G(2,3) = 1; G(2,5) = 1;
G(3,2) = 1; G(3,3) = 1; G(3,4) = 1;
G(4,2) = 1;
G(5,5) = 1;
G(6,4) = 1;

G(7,6) = 1;
G(1,6) = 1;
G(8,7) = 1;
G(4,7) = 1;
% the values of factors
F = ones(K,2,2);
F(1,2,2) = 5;
F(2,1,2) = 0.5;
F(3,1,1) = 0;
F(4,1,1) = 2;
F(6,2,1) = 0;
F(6,2,2) = 0;
F(7,1,1) = 0;
F(7,1,2) = 0;
%------------main part-----------------------
B = computeMarginals(G, F)

B = bruteForce(G, F)


%-------functions--------------

    function B = computeMarginals(G, F)
        %performs factor graph propagation to compute the marginal probabilities of all variables in the graph
        
        %---------initialization--------------------
        %marginal probabilities
        B = ones(N,2);
        %the messages from variables to factors
        VF = ones(N, K, 2);
        %the messages from factors to variables
        FV = ones(K, N, 2);
        % messages have been sent from variables to factors
        Vsent = zeros(N, K);
        % messages have been sent from factors to variables
        Fsent = zeros(K, N);
        
        %find leafs
        leafs=[];
        %row = leaf , connected factor
        for i=1:N
            if nnz(G(i,:))==1
                leafs = [leafs; [i,find(G(i,:))]];
                %since it's a leaf, we can send the message to connected
                %factor straight away
                Vsent(leafs(end,1), leafs(end,2)) = 1;
            end
        end
        
        
        
        
        %------------main part-----------------------
        
        %Let the last leaf in the array 'leafs' is a root and other 'leafs' are the leafs
        
        %Let find messages from the leafs to the root
        for l=1:length(leafs(1:end-1,1))
            message_from_var_to_factor(leafs(l,1),leafs(l,2));
        end
        %Let find messages from the root to the leafs
        message_from_var_to_factor(leafs(end,1),leafs(end,2));
        
        % marginal distribution is a product of the incoming messages
        for n=1:N
            k_n = find(G(n,:));
            for k = k_n
                B(n,:) = (B(n,:)' .* squeeze(FV(k, n, :)));
            end
            %normalization
            B(n,:) = B(n,:) / sum(B(n,:));
        end
        
        
        
        %%%%%%%%%%-----------function for computeMarginals---------%%%%%%%%%%%%%%%%%%%%
        function message_from_var_to_factor(n,k)
            % compute message from the variable n to factor k
            
            k_neigh = find(G(n,:));
            k_neigh = k_neigh(k_neigh~=k);
            prod = 1;
            for i = k_neigh
                prod = prod * Fsent(i,n);
            end
            if prod
                if ~isempty(k_neigh)
                    for i = k_neigh
                        VF(n,k,:) = VF(n,k,:) .* FV(i, n, :);
                    end
                    
                end
                Vsent(n,k) = 1;
                var_neigh = find(G(:,k))';
                var_neigh = var_neigh(var_neigh~=n);
                
                for i=var_neigh
                    message_from_factor_to_var(k,i);
                end
            end
            
        end
        
        
        
        
        function message_from_factor_to_var(k,n)
            % compute message from factor k to variable n
            
            prev = find(G(:,k))';
            prev = prev(prev~=n);
            if prev < n
                FV(k,n,:) = squeeze(F(k,:,:))' * squeeze(VF(prev,k,:));
            else
                FV(k,n,:) = squeeze(F(k,:,:)) * squeeze(VF(prev,k,:));
            end
            Fsent(k,n) = 1;
            
            next = find(G(n,:));
            next = next(next~=k);
            for i = next
                message_from_var_to_factor(n,i);
            end
            
        end
        
        
        
        
    end


    function B = bruteForce(G, F)
        %enumerates all 2N configurations of the variables and returns the marginals
        B = zeros(N,2);
        % joint distribution is equal to product of factors
        % JD is joint distribution where each dimention corresponds to a
        % variable and the value is the potential
        JD = zeros(2,2,2,2,2,2,2,2);
        % the nornalization factor
        NF = 0;
        
        for x1=1:2
            for x2=1:2
                for x3=1:2
                    for x4=1:2
                        for x5=1:2
                            for x6=1:2
                                for x7=1:2
                                    for x8=1:2
                                JD(x1,x2,x3,x4,x5,x6,x7,x8) = F(1,x1,x2) * F(2,x3,x4) * F(3,x2,x3) * F(4,x3,x6) * F(5,x2,x5)*F(6, x1,x7)*F(7,x4,x8);
                                NF = NF + JD(x1,x2,x3,x4,x5,x6,x7,x8);
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
        
        
        %the marginal distr. of x is obtained by summing the joint distribution over all variables except x
        
        for x=1:N
            for x1=1:2
                for x2=1:2
                    for x3=1:2
                        for x4=1:2
                            for x5=1:2
                                for x6=1:2
                                    for x7=1:2
                                        for x8=1:2
                                    i = [x1,x2,x3,x4,x5,x6,x7,x8];
                                    B(x,i(x)) = B(x, i(x)) + JD(x1,x2,x3,x4,x5,x6,x7,x8);
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
            
        end
        B = B / NF;
    end
end
