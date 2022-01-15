import torch
import torch.nn as nn
from torch.cuda import init
from uu.utils import shape_infer 
from uu.utils import padding_calc
from uu.layers import maxpool2d, conv2d, sequential, tilesplit, tilecopy, relu, gavgpool2d, gmaxpool2d
from torch.nn.parameter import Parameter
from uu.utils import correctness_check 
from uu.utils import memory 
from uu.utils import checkpoint
import time

li = []
li_act = []
grad_dict_bk = {}
def print_grad(self, grad_input, grad_output):
    print('Inside '+ self.__class__.__name__+ ' backward')
    # print('grad_output size : ', grad_output[0].size())
    # print('ref grad_output  :\n ', grad_output[0])
    # print('grad_input size : ', grad_input[0].size())
    # print('ref grad_input  : \n', grad_input[0])
    #li.append( grad_output[0])

def print_activa(self, input, output):
    print('Inside '+ self.__class__.__name__+ ' forward')
    # print('input size : ', input[0].size())
    # #print('input : ', input[0])
    # print('output size : ', output[0].size())
    #li_act.append(input[0])

class Net_ref(nn.Module):
    def __init__(self, w1, w2):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=chanel, 
                                  out_channels=chanel, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        self.conv2d_2 = nn.Conv2d(in_channels=chanel, 
                                  out_channels=chanel,
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw)
                                  )
        
                                
        self.maxpool1 = nn.MaxPool2d((2,2), (2,2))
        self.avgp = nn.AdaptiveAvgPool2d((1, 1))
        self.maxgp = nn.AdaptiveMaxPool2d((1, 1))
        self.sft = nn.Softmax(dim=-1)

        self.conv2d_1.weight = Parameter(w1)
        self.conv2d_2.weight = Parameter(w2)
        self.conv2d_1.register_forward_hook(print_activa)
        self.conv2d_2.register_forward_hook(print_activa)
        self.maxpool1.register_forward_hook(print_activa)
        self.conv2d_1.register_full_backward_hook(print_grad)
        self.conv2d_2.register_full_backward_hook(print_grad)
        self.maxpool1.register_full_backward_hook(print_grad)
        self.avgp.register_full_backward_hook(print_grad)


        self.block1 = nn.Sequential(*[self.conv2d_1, self.conv2d_2, self.maxpool1,]) 

    def forward(self, x):
        out = self.block1(x)
        #print("out.shape 1 ", out.size(), out)
        out = self.maxgp(out)
        print("out.shape 2 ", out.size(), out)
        # out = self.sft(out)
        # print("out.shape final", out.size(), out)
        return out


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = conv2d.TiledConv2d(in_channels=chanel, 
                                  out_channels=chanel, 
                                  kernel_size=(Kh,Kw),
                                  bias = False,
                                  padding=(Ph,Pw),
                                  )  

        self.conv2d_2 = conv2d.TiledConv2d(in_channels=chanel, 
                                        out_channels=chanel, 
                                        kernel_size=(Kh,Kw),
                                        bias = False,
                                        padding=(Ph,Pw),
                                        ) 
        self.mxp1 = maxpool2d.cMaxPool2d((2, 2), (2, 2))
        #self.avgp = gavgpool2d.cGAvgPool2d()
        self.gmaxp = gmaxpool2d.cGMaxPool2d()
        self.sft = nn.Softmax(dim=-1)

        self.tsplit = tilesplit.TiledSplit()
        self.tcopy = tilecopy.TiledCopy()

        self.block1 = sequential.mSequential(*[self.tsplit, self.conv2d_1, self.conv2d_2, self.mxp1, self.gmaxp]) #
        
    # static  out_temp =None
    def forward(self, x, H, W, nTh, nTw):
        #nTh, nTw -- num of tiles in H,W
        model_device = next(self.parameters()).device
        N, C, oH, oW, shape_dict = shape_infer.shape_infer_sequence(self.block1, H, W, batch, chanel)
        
        print("!!!!!!!", N, C, oH, oW)
        stream_structure = self.block1

        out = torch.zeros(N, C, oH, oW, requires_grad=True).cuda()
        # print("shape_dict", shape_dict)
        for i in range(0,nTh): 
            for j in range(0,nTw):
                coord = [i,j]
                print("++++++++++++++++++++++++++++++++++++++++++++++++")
                print("coord", coord)
                input_shape = (N,C,H,W)
                output_shape = (N,C,oH,oW)
                info = padding_calc.compute_info_beta([i,j], input_shape, output_shape, nTh, nTw, stream_structure, shape_dict, model_device)
               
                out += self.block1( x, info, stream_structure[1], model_device, [nTh, nTw])
                #print("out", out.size(), out)
                #out += checkpoint.checkpoint(self.block1, x, info, stream_structure[1], model_device, [nTh, nTw])

                # if out_temp is not None:
                #     print("out tile", out_temp.size(), out_temp)


        #out = out_temp
        #out = self.sft(out)
        
        return out


def main():
    torch.set_printoptions(profile="full")
    torch.set_default_dtype(torch.float64)

    # add loss function here
    #criterion = nn.MSELoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = torch.rand(batch,chanel,H,W, requires_grad = True)
    model = Net().to(device)
    print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
    w1 = model.conv2d_1.weight.data
    w2 = model.conv2d_2.weight.data
    model_ref =  Net_ref(w1, w2).to(device)
    input_ref = input.data.clone() 
    input_ref = input_ref.cuda()
    input_ref.requires_grad = True
    out_ref = model_ref(input_ref)
    out_ref.sum().backward()
    
    # print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")


    out = model(input, H, W, nTh, nTw )

    


    print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
    print("~~ check forward correctness ~~")
    print("out ref ", out_ref)
    print("out  ", out)


    correctness_check.check_equal(out, out_ref, False)
    print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")


    out.sum().backward()


    print("#### compare w1")
    correctness_check.check_equal(model_ref.conv2d_1.weight.grad, model.conv2d_1.weight.grad, False)

    print("#### compare w2")
    correctness_check.check_equal(model_ref.conv2d_2.weight.grad, model.conv2d_2.weight.grad, False)
    print(model_ref.conv2d_2.weight.grad)




if __name__=="__main__":
    Kh = 3
    Kw = 3
    Ph = 1
    Pw = 1
    chanel = 2
    batch = 1



    H = 16
    W = 16
    oH = H//2
    oW = W//2
    nTh = 2
    nTw = 2
    main()