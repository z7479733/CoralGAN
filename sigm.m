function X = sigm(P)
    X = 1./(1+exp(-P));%sigmiod是0和1之间的数 不是0或1
end