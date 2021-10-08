function result = delta_sigmoid_cross_entropy(logits, labels)
    temp1 = max(logits, 0);
    temp1(temp1>0) = 1;
    temp2 = logits;
    temp2(temp2>0) = -1;
    temp2(temp2<0) = 1;
    result = temp1 - labels + exp(-abs(logits))./(1+exp(-abs(logits))) .* temp2;
end
