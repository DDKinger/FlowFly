% Version 1.000 
%
% Code provided by Geoff Hinton and Ruslan Salakhutdinov 
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.

% This program trains Restricted Boltzmann Machine in which
% visible, binary, stochastic pixels are connected to
% hidden, binary, stochastic feature detectors using symmetrically
% weighted connections. Learning is done with 1-step Contrastive Divergence.   
% The program assumes that the following variables are set externally:
% maxepoch  -- maximum number of epochs
% numhid    -- number of hidden units 
% batchdata -- the data that is divided into batches (numcases numdims numbatches)
% restart   -- set to 1 if learning starts from beginning 
%
epsilonw      = 0.001;   % Learning rate for weights 
epsilonvb     = 0.001;   % Learning rate for biases of visible units 
epsilonhb     = 0.001;   % Learning rate for biases of hidden units 
epsilonfstd   = 0.0001;   % Learning rate for sigma
weightcost  = 0.0002;   
initialmomentum  = 0.5;
finalmomentum    = 0.9;

% epsilonw      = 0.01;   % Learning rate for weights 
% epsilonvb     = 0.01;   % Learning rate for biases of visible units 
% epsilonhb     = 0.01;   % Learning rate for biases of hidden units 
% epsilonfstd   = 0.001;   % Learning rate for sigma
% weightcost  = 0.002;   
% initialmomentum  = 0.5;
% finalmomentum    = 0.9;

[numcases numdims numbatches]=size(batchdata);

if restart ==1,
  restart=0;
  epoch=1;

% Initializing symmetric weights and biases. 
  vishid     = 0.1*randn(numdims, numhid);
  hidbiases  = zeros(1,numhid);
  visbiases  = zeros(1,numdims);
  fstd = ones(1,numdims);

  poshidprobs = zeros(numcases,numhid);
  neghidprobs = zeros(numcases,numhid);
  posprods    = zeros(numdims,numhid);
  negprods    = zeros(numdims,numhid);
  vishidinc  = zeros(numdims,numhid);
  hidbiasinc = zeros(1,numhid);
  visbiasinc = zeros(1,numdims);
  invfstdInc = zeros(1,numdims);
  batchposhidprobs=zeros(numcases,numhid,numbatches);
end

restruct_error = [];

for epoch = epoch:maxepoch,
 fprintf(1,'epoch %d\r',epoch); 
 errsum=0;
 for batch = 1:numbatches,
%  fprintf(1,'epoch %d batch %d\r',epoch,batch); 
  Fstd = ones(numcases,1)*fstd;
%%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  data0 = batchdata(:,:,batch);
%  rand('state',0);
  data = (rand(size(data0)) > 0.5) .* data0; %
 % data = 0.5 * randn(size(data0)) + data0;
  
  poshidprobs = 1./(1 + exp(-(data./Fstd)*vishid - repmat(hidbiases,numcases,1)));    
  batchposhidprobs(:,:,batch)=poshidprobs;
  posprods    = (data./Fstd)' * poshidprobs;    %vh
  poshidact   = sum(poshidprobs);   %sum(h)
  posvisact = sum(data)./(fstd.^2); %sum(v)

%%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   poshidstates = poshidprobs > rand(numcases,numhid);

%%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   negdata = 1./(1 + exp(-poshidstates*vishid' - repmat(visbiases,numcases,1)));
 % negdata = poshidstates*vishid' + repmat(visbiases,numcases,1) + randn(numcases,numdims).*Fstd;
  negdata = poshidstates*vishid'.*Fstd + repmat(visbiases,numcases,1) + randn(numcases,numdims).*Fstd;
  neghidprobs = 1./(1 + exp(-(negdata./Fstd)*vishid - repmat(hidbiases,numcases,1)));    
  negprods  = (negdata./Fstd)'*neghidprobs;
  neghidact = sum(neghidprobs);
  negvisact = sum(negdata)./(fstd.^2); 

%%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  err= sum(sum( (data0-negdata).^2 ));
%   err = -sum(sum(data0.*log(negdata) + (1-data0).*log(1-negdata)));
  errsum = err + errsum;

   if epoch>5,
     momentum=finalmomentum;
   else
     momentum=initialmomentum;
   end;

%%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    vishidinc = momentum*vishidinc + ...
                epsilonw*( (posprods-negprods)/numcases - weightcost*vishid);
    visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact);
    hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact);
    
    invfstd_grad = sum(2*data.*(repmat(visbiases,numcases,1)-data/2)./Fstd,1) + sum(data' .* (vishid*poshidprobs') ,2)';
    invfstd_grad = invfstd_grad - ( sum(2*negdata.*(repmat(visbiases,numcases,1)-negdata/2)./Fstd,1) + ...
                            sum( negdata'.*(vishid*neghidprobs') ,2 )' );                            
    invfstdInc = momentum*invfstdInc + epsilonfstd/numcases*invfstd_grad;

    vishid = vishid + vishidinc;
    visbiases = visbiases + visbiasinc;
    hidbiases = hidbiases + hidbiasinc;
    
    invfstd = 1./fstd;
    invfstd =  invfstd + invfstdInc;
    fstd = 1./invfstd;
    fstd = max(fstd, 0.005); %have a lower bound! 

%%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

 end
  restruct_error(epoch) = errsum;
  fprintf(1, 'epoch %4i error %6.1f  \n', epoch, errsum); 
end;
