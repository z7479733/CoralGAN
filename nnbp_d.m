function nn = nnbp_d(nn, y_h, y)
    % d表示残差，残差就是最终的loss对各层未激活值（z）的偏导，偏导数的计算需要采用链式求导法则-自己手动推出来
    n = nn.layers_count;
    % 最后一层的残差
    nn.layers{n}.d = delta_sigmoid_cross_entropy(y_h, y);
    for i = n-1:-1:2
        d = nn.layers{i+1}.d;
        w = nn.layers{i+1}.w;
        z = nn.layers{i}.z;
        % 每一层的残差是对每一层的未激活值求偏导数，所以是后一层的残差乘上w,再乘上对激活值对未激活值的偏导数
        nn.layers{i}.d = d*w' .* delta_relu(z);    
    end
    % 求出各层的残差之后，就可以根据残差求出最终loss对weights和biases的偏导数
    for i = 2:n
        d = nn.layers{i}.d;
        a = nn.layers{i-1}.a;
        % dw是对每层的weights进行偏导数的求解
        nn.layers{i}.dw = a'*d / size(d, 1);
        nn.layers{i}.db = mean(d, 1);
    end
end
% discriminator的bp，下面的bp涉及到对各个参数的求导
% 如果更改网络结构（激活函数等）则涉及到bp的更改，更改weights，biases的个数则不需要更改bp
% 为了更新w,b，就是要求最终的loss对w，b的偏导数，残差就是在求w，b偏导数的中间计算过程的结果
% generator的bp

