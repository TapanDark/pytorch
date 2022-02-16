import torch
import torch.nn as nn
from torch.cuda import init
from uu.utils import shape_infer 
from uu.utils import padding_calc
from uu.layers import maxpool2d, conv2d, sequential, tilesplit, tilecopy, relu, gavgpool2d
from torch.nn.parameter import Parameter
from uu.utils import correctness_check 
from uu.utils import memory 
from uu.utils import checkpoint
import time


Kh = Kw = 3
Ph = Pw = 1
b = 1
c = 1
h = w = 64
out_ch = 1
in_ch = 1
nTh = nTw = 4
strideH = 1



class Net_ref(nn.Module):
    def __init__(self, w1, b1, w2, b2):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=in_ch, 
                                  out_channels=out_ch, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw),
                                  stride=(strideH,strideH)
                                  )
        self.conv2d_2 = nn.Conv2d(in_channels=out_ch, 
                                  out_channels=out_ch,
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        # w1 = (torch.reshape(torch.arange(0, in_ch*out_ch*Kh*Kw, step=1.0, dtype=torch.float), (out_ch, in_ch, Kh, Kw)))
        #b1 = torch.tensor([-1000, -2000, -3000, -4000, -5000], dtype=torch.float)
        self.conv2d_1.weight = Parameter(w1)
        #self.conv2d_1.bias = Parameter(b1)
        self.conv2d_2.weight = Parameter(w2)
        #self.conv2d_2.bias = Parameter(b2)

        #print("conv bias ref", self.conv2d_1.bias)
        #print("conv bias ref", self.conv2d_2.bias)

        self.conv2d_1.register_full_backward_hook(print_grad)
        

    def forward(self, x):
        out = self.conv2d_1(x)
        #out = self.conv2d_2(out)
        #print("out.shape final", out.size(), out)
        return out


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = conv2d.TiledConv2d(in_channels=in_ch, 
                                  out_channels=out_ch, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw),
                                  stride=(strideH,strideH)
                                  )
        self.conv2d_2 = conv2d.TiledConv2d(in_channels=out_ch, 
                                  out_channels=out_ch,
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        # w1 = (torch.reshape(torch.arange(0, in_ch*out_ch*Kh*Kw, step=1.0, dtype=torch.float), (out_ch, in_ch, Kh, Kw)))
        # b1 = torch.tensor([-1000, -2000, -3000, -4000, -5000], dtype=torch.float)
        # self.conv2d_1.weight = Parameter(w1)
        # self.conv2d_1.bias = Parameter(b1)

        self.tsplit = tilesplit.TiledSplit()
        self.tcopy = tilecopy.TiledCopy()

        self.block1 = sequential.mSequential(*[self.tsplit, self.conv2d_1,
         #self.conv2d_2
         ])

        # self.conv2d_1.register_full_backward_hook(print_grad)
        

    def forward(self, x):
        model_device = next(self.parameters()).device
        N, C, oH, oW, shape_dict = shape_infer.shape_infer_sequence(self.block1, h, w, b, in_ch)
        
        print("!!!!!!!", N, C, oH, oW)
        stream_structure = self.block1

        out = torch.zeros(N, C, oH, oW, requires_grad=True).cuda()
        #print("out shape", out.size())
        for i in range(0,nTh): 
            for j in range(0,nTw):
                coord = [i,j]
                print("++++++++++++++++++++++++++++++++++++++++++++++++")
                print("coord", coord)
                input_shape = (N,C,h,w)
                output_shape = (N,C,oH,oW)
                info = padding_calc.compute_info_beta([i,j], input_shape, output_shape, nTh, nTw, stream_structure, shape_dict, model_device)
               
                out_temp = self.block1(x, info, stream_structure[1], model_device, [nTh, nTw])

                fake_pi = info[0][-11]
                tile_shape = fake_pi.cur_output_shape
                tile_size = [tile_shape[0], tile_shape[1]]
                output_index = fake_pi.input_slice
                print("corp output", tile_shape, tile_size, output_index)
                out = self.tcopy(out_temp, out, output_index, tile_size)
                del out_temp
                del info
                
        #print("out", out.size(), out)
        return out



def print_grad(self, grad_input, grad_output):
    print('Inside '+ self.__class__.__name__+ ' backward')
    print('grad_output size : ', grad_output[0].size())
    #print('ref grad_output  :\n ', grad_output[0])
    print('grad_input size : ', grad_input[0])
    #print('ref grad_input  : \n', grad_input[0])

def main():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    input = torch.rand(c,in_ch,h,w, requires_grad = True).cuda()
    
    

    input_our = input.data.clone() 
    input_our = input_our.cuda()
    input_our.requires_grad = True
    model_our = Net().to(device)
    

    w1 = model_our.conv2d_1.weight.data
    b1 = None # model_our.conv2d_1.bias.data
    w2 = model_our.conv2d_2.weight.data
    b2 = None # model_our.conv2d_2.bias.data
    # print("b1", b1)
    # print("b2", b2)
    # w2 = model_our.conv2d_2.weight.data
    # b1 = model_our.conv2d_2.bias.data
    #print("out_our", out_our)




    model = Net_ref(w1, b1, w2, b2).to(device)
    out = model(input)
    # print("ref_conv2d_1.bias", model.conv2d_1.bias)
    print("out", out.size())
    #out.sum().backward()

    print("\n&&&&&&&&&&&&&&&&&&& OUR forward &&&&&&&&&&&&&&&&&&&\n")


    out_our = model_our(input_our)
    #out_our.sum().backward()

    print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
    print("~~ check forward correctness ~~")
    correctness_check.check_equal(out, out_our, False)
    #print(out, out_our)

    print("#### compare w1")
    #correctness_check.check_equal(model.conv2d_1.weight.grad, model_our.conv2d_1.weight.grad, False)

    # print("#### compare b1")
    # correctness_check.check_equal(model.conv2d_1.bias.grad, model_our.conv2d_1.bias.grad, False)
    # print( model.conv2d_1.bias.grad)
    # print( model_our.conv2d_1.bias.grad)

    # print("#### compare w2")
    # correctness_check.check_equal(model.conv2d_2.weight.grad, model_our.conv2d_2.weight.grad, False)

    # print("#### compare b2")
    # correctness_check.check_equal(model.conv2d_2.bias.grad, model_our.conv2d_2.bias.grad, False)
    # print( model.conv2d_2.bias.grad)
    # print( model_our.conv2d_2.bias.grad)




    # print(ref_conv2d_1.weight.grad)
    
    
    






if __name__=="__main__":
    main()
   