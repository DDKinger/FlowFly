[testnumcases testnumdims testnumbatches]=size(testbatchdata);
N=testnumcases;
Y_predict = [];
for batch = 1:testnumbatches
  data = [testbatchdata(:,:,batch)];
  data = [data ones(N,1)];
  w1probs = 1./(1 + exp(-data*w1)); w1probs = [w1probs  ones(N,1)];
  w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
  %w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  ones(N,1)];
  targetout = 1./(1 + exp(-w2probs*w_pre));
  Y_predict = [Y_predict;targetout];
end