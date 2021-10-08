function g_net = nnbp_g(g_net, d_net)
    n = g_net.layers_count;
    a = g_net.layers{n}.a;
    % generator的loss是由label_fake得到的，(images_fake过discriminator得到label_fake)
    % 对g进行bp的时候，可以将g和d看成是一个整体
    % g最后一层的残差等于d第2层的残差乘上(a .* (a_o))
    g_net.layers{n}.d = d_net.layers{2}.d * d_net.layers{2}.w' .* (a .* (1-a));
    for i = n-1:-1:2
        d = g_net.layers{i+1}.d;
        w = g_net.layers{i+1}.w;
        z = g_net.layers{i}.z;
        % 每一层的残差是对每一层的未激活值求偏导数，所以是后一层的残差乘上w,再乘上对激活值对未激活值的偏导数
        g_net.layers{i}.d = d*w' .* delta_relu(z);    
    end
    % 求出各层的残差之后，就可以根据残差求出最终loss对weights和biases的偏导数
    for i = 2:n
        d = g_net.layers{i}.d;
        a = g_net.layers{i-1}.a;
        % dw是对每层的weights进行偏导数的求解
        g_net.layers{i}.dw = a'*d / size(d, 1);
        g_net.layers{i}.db = mean(d, 1);
    end
end
