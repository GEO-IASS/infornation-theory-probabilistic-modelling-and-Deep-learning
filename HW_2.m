function HW_2()
%-----initialization----------
% # of variables
N = 6;
% # of factors
K = 5;

% the graph matrix
G = zeros(N,K);
G(1,1) = 1;
G(2,1) = 1; G(2,3) = 1; G(2,5) = 1;
G(3,2) = 1; G(3,3) = 1; G(3,4) = 1;
G(4,2) = 1;
G(5,5) = 1;
G(6,4) = 1;

% the values of factors
F = ones(K,2,2);
F(1,2,2) = 5;
F(2,1,2) = 0.5;
F(3,1,1) = 0;
F(4,1,1) = 2;
%------------main part-----------------------
B_compMarg = computeMarginals(G, F)

B_brForce = bruteForce(G, F)

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
        
        %find leaves
        leaves=[];
        %row = leaf , connected factor
        for i=1:N
            if nnz(G(i,:))==1
                leaves = [leaves; [i,find(G(i,:))]];
                %since it's a leaf, we can send the message to connected
                %factor straight away
                Vsent(leaves(end,1), leaves(end,2)) = 1;
            end
        end

        %------------main part-----------------------
        
        %Let the last leaf in the array 'leaves' is a root and other 'leaves' are the leaves
        
        %Let find messages from the leaves to the root
        for l=1:length(leaves(1:end-1,1))
            message_from_var_to_factor(leaves(l,1),leaves(l,2));
        end
        %Let find messages from the root to the leaves
        message_from_var_to_factor(leaves(end,1),leaves(end,2));
        
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
            
            % k_neigh is a set of factors that are neighbours of the node n 
            k_neigh = find(G(n,:));
            % exclude k from the set of neighbours
            k_neigh = k_neigh(k_neigh~=k);
            prod = 1;
            % ensure that we get all messages from the neighbours; if
            % prod==1 then we can move forward (case k_neigh==[] means that the variable n is connected to one factor;
            % we can move forward in this case)
            for i = k_neigh
                prod = prod * Fsent(i,n);
            end
            if prod
                % if k_neigh==[] then VF(n,k,:)=[1,1], otherwise it's a
                % product of potentials of connected to n factors 
                if ~isempty(k_neigh)
                    for i = k_neigh
                        VF(n,k,:) = VF(n,k,:) .* FV(i, n, :);
                    end
                    
                end
                % check that we sent the message
                Vsent(n,k) = 1;
                %using reccurtion we will send messenges from the factor k to connected to k nodes except n  
                var_neigh = find(G(:,k))';
                var_neigh = var_neigh(var_neigh~=n);
                
                for i=var_neigh
                    message_from_factor_to_var(k,i);
                end
            end
            
        end
        
            
        function message_from_factor_to_var(k,n)
            % compute message from factor k to variable n
            
            %look for set of nodes connected to factor k except n
            prev = find(G(:,k))';
            prev = prev(prev~=n);
            
            % since the smallest index corresponds to rows we need to make
            % sure that we summing out the 'prev' node  
            if prev < n
                FV(k,n,:) = squeeze(F(k,:,:))' * squeeze(VF(prev,k,:));
            else
                FV(k,n,:) = squeeze(F(k,:,:)) * squeeze(VF(prev,k,:));
            end
            % check that we sent the message
            Fsent(k,n) = 1;
            
            %using reccurtion we will send messenges from the node n to
            %connected to n factors except k
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
        JD = zeros(2,2,2,2,2,2);
        % the nornalization factor
        NF = 0;
        
        for x1=1:2
            for x2=1:2
                for x3=1:2
                    for x4=1:2
                        for x5=1:2
                            for x6=1:2
                                JD(x1,x2,x3,x4,x5,x6) = F(1,x1,x2) * F(2,x3,x4) * F(3,x2,x3) * F(4,x3,x6) * F(5,x2,x5);
                                NF = NF + JD(x1,x2,x3,x4,x5,x6);
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
                                    i = [x1,x2,x3,x4,x5,x6];
                                    B(x,i(x)) = B(x, i(x)) + JD(x1,x2,x3,x4,x5,x6);
                                end
                            end
                        end
                    end
                end
            end
            
        end
        % normalization
        B = B / NF;
    end
end
