function hystog()
%plot hystograms
dataset = 2;
% q is a set of parameters
q=[];
[a,b ]= HW3_part3(1, dataset);
q = [q,b];
[a,b ]= HW3_part3(2, dataset);
q = [q,b];
[a,b ]= HW3_part3(3, dataset);
q = [q,b];
hyperpar = q
%set of evidences
A = [HW3_part3(1, dataset), HW3_part3(2, dataset), HW3_part3(3, dataset)];
bar(A);
end