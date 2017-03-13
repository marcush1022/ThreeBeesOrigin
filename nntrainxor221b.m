%   /* 
%   Training Artificial Neural Network.
%   XOR Problem.
%   Summation Units, Log-Sigmoid Neurons with Biases.
%   Input Layer: 2, Hidden Layer: 2, Output Layer: 1 neurons. 
%	Returns mean square error between desired and actual outputs.
%   Reference Papers:
%   D. Karaboga, B. Basturk Akay, C. Ozturk, Artificial Bee Colony (ABC) Optimization Algorithm for Training Feed-Forward Neural Networks, LNCS: Modeling Decisions for Artificial Intelligence, 4617/2007, 318-329, 2007.
%   D. Karaboga, C. Ozturk, Neural Networks Training by Artificial Bee Colony Algorithm on Pattern Classification, Neural Network World, 19(3), 279-292, 2009. 
%   */

function ObjVal = nntrainxor221b(Chrom)
[Nind Nvar]=size(Chrom);
trin=[0 0;0 1; 1 0; 1 1];
trout = [0 1 1 0]';
inp=size(trin,2);
out=size(trout,2);
hidden=2;

for i=1:Nind
    
x=Chrom(i,:);

    iw = reshape(x(1:hidden*inp),hidden,inp);
    b1 = reshape(x(hidden*inp+1:hidden*inp+hidden),hidden,1);
    lw = reshape(x(hidden*inp+hidden+1:hidden*inp+hidden+hidden*out),out,hidden);
    b2 = reshape(x(hidden*inp+hidden+hidden*out+1:hidden*inp+hidden+hidden*out+out),out,1);
    
    y = logsig(logsig(trin*iw'+repmat(b1',size(trin,1),1))*lw'+repmat(b2',size(trin,1),1));


    ObjVal(i)=mse(trout-y);
end;


