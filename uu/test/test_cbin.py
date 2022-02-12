import torch

if __name__=="__main__":

    k = 7
    o = 10
    i = 2*o+k-2

    input1 = torch.rand(1,1,i,i).cuda()
    input_size = input1.size()
    new_grad_out1 = torch.rand(1,1,o,o).cuda()
    weight_tensor = torch.rand(1,1,k,k).cuda()
    our_padding = (0,0)
    stride = (2,2)
    dilation = (1,1)
    groups = 1
    grad_input = torch.cudnn_convolution_backward_input(input_size, new_grad_out1,
     weight_tensor, our_padding, stride, dilation, groups, False, False, False)
