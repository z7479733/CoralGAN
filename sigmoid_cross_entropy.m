function result = sigmoid_cross_entropy(logits, labels)
    result = max(logits, 0) - logits .* labels + log(1 + exp(-abs(logits)));
    result = mean(result);
end


