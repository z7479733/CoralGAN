function X = sigmrnd(P)
    X = double(1./(1+exp(-P)) > rand(size(P)));%rand(size(P))产生01之间的随机数 double 把整数和小数都视为浮点数
end