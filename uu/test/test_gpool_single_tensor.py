import torch
import torch.nn as nn
from math import prod

def main():

    # tiny avg pooling sum part
    b = 2
    c = 3
    h = 4
    w = 4
    input = (torch.reshape(torch.arange(0, b*c*h*w, step=1.0, dtype=torch.float), (b, c, h, w)))
    print("input ", input, input.size())
    tp = (len(input.size())-2, len(input.size())-1)
    print("tp ", tp)
    overhw = input.sum(dim=tp)
    print("input overhw", overhw, overhw.size())


    input2 = input * -0.5
    print("input2 ", input2, input2.size())
    tp = (len(input2.size())-2, len(input2.size())-1)
    print("tp ", tp)
    overhw2 = input2.sum(dim=tp)
    print("input overhw", overhw2, overhw2.size())
    accum = overhw2
    accum = accum + overhw

    print("accum", accum, accum.size())

    numofelement = h*w*2
    accum = accum/numofelement
    accum = accum[:, :, None,None] # expand to 4D tensor
    print("accum", accum, accum.size())

    #backward
    print(prod(accum.size()))
    back_tensor = torch.rand(b, c)
    #(torch.reshape(torch.arange(0, prod(accum.size()), step=1, dtype=torch.float), accum.size()))
    print("backward", back_tensor, back_tensor.size())
    # avg to expand out 
    grad_t1 = torch.zeros(b, c, h, w)
    print("grad_t1 original", grad_t1, grad_t1.size())

    for i in range(0,b):
        for j in range (0,c):
            grad_t1[i,j,:,:] = back_tensor[i,j]
    
    print("grad_t1 fillin", grad_t1, grad_t1.size())

    





    print("--------------------------------------------")


    # # tiny g max pooling 
    # input = (torch.reshape(torch.arange(0, b*c*h*w, step=1.0, dtype=torch.float), (b, c, h, w)))
    # filt_size = (input.size()[len(input.size())-2], input.size()[len(input.size())-1])
    # print("filt_size ", filt_size)
    # gmp = nn.MaxPool2d(filt_size)
    # partial_max1 = gmp(input)
    # #partial_max1 = partial_max1.squeeze()


    # input2 = input * -0.5
    # partial_max2 = gmp(input2)
    # #partial_max2 = partial_max2.squeeze()
    # print("partial_max1", partial_max1, partial_max1.size())
    # print("partial_max2", partial_max2, partial_max2.size())

    # maxi = partial_max1
    # maxi = torch.maximum(maxi, partial_max2)

    # # how to record the maxi index in bxc space??
    

    # print("final max", maxi, maxi.size())






if __name__=="__main__":
    main()