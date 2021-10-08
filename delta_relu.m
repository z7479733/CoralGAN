function output = delta_relu(x)
    output = max(x,0);
    output(output>0) = 1;
end
