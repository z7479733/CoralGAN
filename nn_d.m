function nn = nn_d(g_nn,g_d,nn)
 n = g_nn.layers_count;
%   ºÏ²¢ÌÝ¶È
    for i = 2:n 
         nn.layers{i}.dw = g_nn.layers{i}.dw + g_d.layers{i}.dw;
         nn.layers{i}.db = g_nn.layers{i}.db + g_d.layers{i}.db;
    end
     
end

