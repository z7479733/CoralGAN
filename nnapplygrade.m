function nn = nnapplygrade(nn,learning_rate)
   n = nn.layers_count;
   nn.t = nn.t+1;
   beta1 = nn.beta1;
   beta2 = nn.beta2;
   lr = learning_rate * sqrt(1-nn.beta2^nn.t) / (1-nn.beta1^nn.t);
    for i = 2:n
        dw = nn.layers{i}.dw;
        db = nn.layers{i}.db;
        % 下面的6行代码是使用adam更新weights与biases
        nn.layers{i}.w_m = beta1 * nn.layers{i}.w_m + (1-beta1) * dw;
        nn.layers{i}.w_v = beta2 * nn.layers{i}.w_v + (1-beta2) * (dw.*dw);
        nn.layers{i}.w = nn.layers{i}.w - lr * nn.layers{i}.w_m ./ (sqrt(nn.layers{i}.w_v) + nn.epsilon);
        nn.layers{i}.b_m = beta1 * nn.layers{i}.b_m + (1-beta1) * db;
        nn.layers{i}.b_v = beta2 * nn.layers{i}.b_v + (1-beta2) * (db.*db);
        nn.layers{i}.b = nn.layers{i}.b - lr * nn.layers{i}.b_m ./ (sqrt(nn.layers{i}.b_v) + nn.epsilon); 
    end
end
% 应用梯度
% 使用adam算法更新变量，可以参考：
% https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
% 保存图片，便于观察generator生成的images_fake
