function X = sigmrnd(P)
    X = double(1./(1+exp(-P)) > rand(size(P)));%rand(size(P))����01֮�������� double ��������С������Ϊ������
end