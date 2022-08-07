clear;
close all;
tic
pa=pwd;
loc_i = '/data/pre_orx1_57/point_v1_';
loc1 = strcat(pa,loc_i);
nu=2000;

%
[data_csi1,data_csi2,data_csi3] = data_log(1,loc1);

train1 = [ data_csi1(1:nu, :) ];
train2 = [ data_csi2(1:nu, :) ];
train3 = [ data_csi3(1:nu, :) ];
%
[data_csi1,data_csi2,data_csi3] = data_nor(1,loc1);

train4 = [ data_csi1(1:nu, :) ];
train5 = [ data_csi2(1:nu, :) ];
train6 = [ data_csi3(1:nu, :) ];
%}
%[2:57];%
%
train_index = [1:57];
test_index = [1,2,7,13,24,27,30,33,37,44,54];%[1,3,10,18,23,32,35,40,43,50,56]s2%[1,7,13,25,27,29,36,43,47,50,53]s3%[1,2,7,13,24,27,30,33,37,44,54]s1
train_index(test_index)=[];
test_index(1:2)=[];

for i = train_index

    [data_csi1,data_csi2,data_csi3] = data_log(i,loc1);
    
    train1 = [train1 ; data_csi1(1:nu, :) ];
    train2 = [train2 ; data_csi2(1:nu, :) ];
    train3 = [train3 ; data_csi3(1:nu, :) ];
    
    %
    [data_csi1,data_csi2,data_csi3] = data_nor(i,loc1);
    
    train4 = [train4 ; data_csi1(1:nu, :) ];
    train5 = [train5 ; data_csi2(1:nu, :) ];
    train6 = [train6 ; data_csi3(1:nu, :) ];
    %}
end

[data_csi1,data_csi2,data_csi3] = data_log(2,loc1);

test1 = [ data_csi1(1:1000, :) ];
test2 = [ data_csi2(1:1000, :) ];
test3 = [ data_csi3(1:1000, :) ];
%
[data_csi1,data_csi2,data_csi3] = data_nor(2,loc1);

test4 = [ data_csi1(1:1000, :) ];
test5 = [ data_csi2(1:1000, :) ];
test6 = [ data_csi3(1:1000, :) ];
%}
for i = test_index

    [data_csi1,data_csi2,data_csi3] = data_log(i,loc1);
    
    test1 = [test1 ; data_csi1(1:1000, :) ];
    test2 = [test2 ; data_csi2(1:1000, :) ];
    test3 = [test3 ; data_csi3(1:1000, :) ];
    %
    [data_csi1,data_csi2,data_csi3] = data_nor(i,loc1);
    
    test4 = [test4 ; data_csi1(1:1000, :) ];
    test5 = [test5 ; data_csi2(1:1000, :) ];
    test6 = [test6 ; data_csi3(1:1000, :) ];
    %}
end
%}
label_train = eye(length(train1)/nu, length(train1)/nu);
label_train = repelem(label_train, nu, 1);

loc_o = '/data/data_orx1_47_anlog.mat';
loc2 = strcat(pa,loc_o);

save(loc2 ,  'train1','test1','train2','test2','train3','test3','train4','test4','train5','test5','train6','test6','label_train');

toc
function [log_csi1,log_csi2,log_csi3] = data_log(input,loc)
k=input;
path = strcat(loc,char(string(k)));
load(path);
abs_csi1 = abs(csi11);
nor_csi1= abs_csi1./ sqrt(sum(abs_csi1, 2));
abs_csi2 = abs(csi21);
nor_csi2= abs_csi2./ sqrt(sum(abs_csi2, 2));
abs_csi3 = abs(csi31);
nor_csi3= abs_csi3./ sqrt(sum(abs_csi3, 2));

for i = 1:length(nor_csi3)
    log_csi1(i, :) = ((log10(nor_csi1(i, :).* 10^((double(rssi(i, 1))) /20))).*20 )./60;
    log_csi2(i, :) = ((log10(nor_csi2(i, :).* 10^((double(rssi(i, 2))) /20))).*20 )./60;
    log_csi3(i, :) = ((log10(nor_csi3(i, :).* 10^((double(rssi(i, 3))) /20))).*20 )./60;
end
random_index = randperm(size(csi11, 1));
log_csi1 = log_csi1(random_index, :);
log_csi2 = log_csi2(random_index, :);
log_csi3 = log_csi3(random_index, :);

log_csi1 = dan(log_csi1);
log_csi2 = dan(log_csi2);
log_csi3 = dan(log_csi3);
end
function [output] = dan(input)
output=input;
ind = output>0;
output(~ind) = 0;
ind = output<1;
output(~ind) = 1;
end
function [no_csi1,no_csi2,no_csi3] = data_nor(input,loc)
k=input;
path = strcat(loc,char(string(k)));
load(path);

abs_csi1 = abs(csi11);
nor_csi1 = abs_csi1./ sqrt(sum(abs_csi1, 2));
abs_csi2 = abs(csi21);
nor_csi2 = abs_csi2./ sqrt(sum(abs_csi2, 2));
abs_csi3 = abs(csi31);
nor_csi3 = abs_csi3./ sqrt(sum(abs_csi3, 2));

m=size(abs_csi1);
no_csi1 = zeros(m(1),m(2));
no_csi2 = zeros(m(1),m(2));
no_csi3 = zeros(m(1),m(2));
for i=1:m(1)
    no_csi1(i,:) = nom(nor_csi1(i,:));
    no_csi2(i,:) = nom(nor_csi2(i,:));
    no_csi3(i,:) = nom(nor_csi3(i,:));
end

random_index = randperm(size(csi11, 1));

no_csi1 = no_csi1(random_index, :);
no_csi2 = no_csi2(random_index, :);
no_csi3 = no_csi3(random_index, :);

end
function [output] = nom(input)
ma=max(max(input));
mi=min(min(input));
output = (input-mi)./(ma-mi);
end
function [rssi] = data_rssi(input,loc)
k=input;
path = strcat(loc,char(string(k)));
load(path);
end