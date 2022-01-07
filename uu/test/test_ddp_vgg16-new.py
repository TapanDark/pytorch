import torch
import torch.nn as nn
from uu.utils import shape_infer 
from uu.utils import padding_calc
from uu.layers import maxpool2d, conv2d, sequential, tilesplit, tilecopy, relu, gavgpool2d
from torch.nn.parameter import Parameter
from uu.utils import correctness_check 
from uu.utils import checkpoint

import os                                                                                
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset



li = []
li_act = []
grad_dict_bk = {}
def print_grad(self, grad_input, grad_output):
#     print('Inside '+ self.__class__.__name__+ ' backward')
#     print('grad_output size : ', grad_output[0].size())
#     #print('ref grad_output  :\n ', grad_output[0])
#     print('grad_input size : ', grad_input[0].size())
#    # print('ref grad_input  : \n', grad_input[0])
    li.append( grad_output[0])

def print_activa(self, input, output):
    # print('Inside '+ self.__class__.__name__+ ' forward')
    # print('input size : ', input[0].size())
    # #print('input : ', input[0])
    # print('output size : ', output[0].size())
    li_act.append(input[0])

    

Kh = 3
Kw = 3
Ph = 1
Pw = 1
chanel = 3
batch = 24

H = 256
W = 256
oH = H//32
oW = W//32
nTh = 2
nTw = 2
gpool_size = 1
num_classes = 1024
class Net_ref(nn.Module):
    def __init__(self, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, fcw1, fcw2
    , b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=3, 
                                  out_channels=64, 
                                  kernel_size=(Kh,Kw),
                                  #bias = False,
                                  padding=(Ph,Pw)
                                  )
        self.conv2d_2 = nn.Conv2d(in_channels=64, 
                                  out_channels=64, 
                                  kernel_size=(Kh,Kw),
                                  #bias = False,
                                  padding=(Ph,Pw)
                                  )
        
                                
        self.maxpool1 = nn.MaxPool2d((2,2), (2,2))
        self.conv2d_3 = nn.Conv2d(in_channels=64, 
                                  out_channels=128, 
                                  kernel_size=(Kh,Kw),
                                  #bias = False,
                                  padding=(Ph,Pw)
                                  )
        self.conv2d_4 = nn.Conv2d(in_channels=128, 
                                  out_channels=128, 
                                  kernel_size=(Kh,Kw),
                                  #bias = False,
                                  padding=(Ph,Pw)
                                  )
        self.maxpool2 = nn.MaxPool2d((2,2), (2,2))                          
        self.conv2d_5 = nn.Conv2d(in_channels=128, 
                                  out_channels=256, 
                                  kernel_size=(Kh,Kw),
                                  #bias = False,
                                  padding=(Ph,Pw)
                                  )
        self.conv2d_6 = nn.Conv2d(in_channels=256, 
                                  out_channels=256, 
                                  kernel_size=(Kh,Kw),
                                  #bias = False,
                                  padding=(Ph,Pw)
                                  )
        self.conv2d_7 = nn.Conv2d(in_channels=256, 
                                  out_channels=256, 
                                  kernel_size=(Kh,Kw),
                                  #bias = False,
                                  padding=(Ph,Pw)
                                  )
        
                                
        self.maxpool3 = nn.MaxPool2d((2,2), (2,2))

        self.conv2d_8 = nn.Conv2d(in_channels=256, 
                                  out_channels=512, 
                                  kernel_size=(Kh,Kw),
                                  #bias = False,
                                  padding=(Ph,Pw)
                                  )
        self.conv2d_9 = nn.Conv2d(in_channels=512, 
                                  out_channels=512, 
                                  kernel_size=(Kh,Kw),
                                  #bias = False,
                                  padding=(Ph,Pw)
                                  )
        self.conv2d_10 = nn.Conv2d(in_channels=512, 
                                  out_channels=512, 
                                  kernel_size=(Kh,Kw),
                                  #bias = False,
                                  padding=(Ph,Pw)
                                  )

        self.maxpool4 = nn.MaxPool2d((2,2), (2,2))

        self.conv2d_11 = nn.Conv2d(in_channels=512, 
                                  out_channels=512, 
                                  kernel_size=(Kh,Kw),
                                  #bias = False,
                                  padding=(Ph,Pw)
                                  )
        self.conv2d_12 = nn.Conv2d(in_channels=512, 
                                  out_channels=512, 
                                  kernel_size=(Kh,Kw),
                                  #bias = False,
                                  padding=(Ph,Pw)
                                  )
        self.conv2d_13 = nn.Conv2d(in_channels=512, 
                                  out_channels=512, 
                                  kernel_size=(Kh,Kw),
                                  #bias = False,
                                  padding=(Ph,Pw)
                                  )

        self.maxpool5 = nn.MaxPool2d((2,2), (2,2))
        self.relu = nn.ReLU()
        
        
        self.avgp = nn.AvgPool2d(oH, stride=1)
        self.flat = nn.Flatten()
        in_feature = 512
        self.fc1 = nn.Linear(in_feature, 1024, bias=False)
        self.sft = nn.Softmax(dim=-1)
        
        
        # self.flat = nn.Flatten()
        # in_feature = chanel*oH*oW
        # self.fc1 = nn.Linear(in_feature, 1024, bias=False)
        # self.fc2 = nn.Linear(1024, 1024, bias=False)



        self.conv2d_1.weight = Parameter(w1)
        self.conv2d_2.weight = Parameter(w2)
        self.conv2d_3.weight = Parameter(w3)
        self.conv2d_4.weight = Parameter(w4)
        self.conv2d_5.weight = Parameter(w5)
        self.conv2d_6.weight = Parameter(w6)
        self.conv2d_7.weight = Parameter(w7)
        self.conv2d_8.weight = Parameter(w8)
        self.conv2d_9.weight = Parameter(w9)
        self.conv2d_10.weight = Parameter(w10)
        self.conv2d_11.weight = Parameter(w11)
        self.conv2d_12.weight = Parameter(w12)
        self.conv2d_13.weight = Parameter(w13)
        self.conv2d_1.bias = Parameter(b1)
        self.conv2d_2.bias = Parameter(b2)
        self.conv2d_3.bias = Parameter(b3)
        self.conv2d_4.bias = Parameter(b4)
        self.conv2d_5.bias = Parameter(b5)
        self.conv2d_6.bias = Parameter(b6)
        self.conv2d_7.bias = Parameter(b7)
        self.conv2d_8.bias = Parameter(b8)
        self.conv2d_9.bias = Parameter(b9)
        self.conv2d_10.bias = Parameter(b10)
        self.conv2d_11.bias = Parameter(b11)
        self.conv2d_12.bias = Parameter(b12)
        self.conv2d_13.bias = Parameter(b13)



        self.fc1.weight = Parameter(fcw1)
        #self.fc2.weight = Parameter(fcw2)
       
        self.conv2d_1.register_forward_hook(print_activa)
        self.conv2d_3.register_forward_hook(print_activa)
        self.conv2d_4.register_forward_hook(print_activa)
        self.conv2d_2.register_forward_hook(print_activa)
        self.conv2d_5.register_forward_hook(print_activa)
        self.conv2d_6.register_forward_hook(print_activa)
        self.conv2d_7.register_forward_hook(print_activa)
        self.conv2d_8.register_forward_hook(print_activa)
        self.conv2d_9.register_forward_hook(print_activa)
        self.conv2d_10.register_forward_hook(print_activa)
        self.conv2d_11.register_forward_hook(print_activa)
        self.conv2d_12.register_forward_hook(print_activa)
        self.conv2d_13.register_forward_hook(print_activa)
        self.maxpool1.register_forward_hook(print_activa)
        self.maxpool2.register_forward_hook(print_activa)
        self.maxpool3.register_forward_hook(print_activa)
        self.maxpool4.register_forward_hook(print_activa)
        self.maxpool5.register_forward_hook(print_activa)
        
        self.conv2d_1.register_full_backward_hook(print_grad)
        self.conv2d_2.register_full_backward_hook(print_grad)
        self.conv2d_3.register_full_backward_hook(print_grad)
        self.conv2d_4.register_full_backward_hook(print_grad)
        self.conv2d_5.register_full_backward_hook(print_grad)
        self.conv2d_6.register_full_backward_hook(print_grad)
        self.conv2d_7.register_full_backward_hook(print_grad)
        self.conv2d_8.register_full_backward_hook(print_grad)
        self.conv2d_9.register_full_backward_hook(print_grad)
        self.conv2d_10.register_full_backward_hook(print_grad)
        self.conv2d_11.register_full_backward_hook(print_grad)
        self.conv2d_12.register_full_backward_hook(print_grad)
        self.conv2d_13.register_full_backward_hook(print_grad)
        self.maxpool1.register_full_backward_hook(print_grad)
        self.maxpool2.register_full_backward_hook(print_grad)
        self.maxpool3.register_full_backward_hook(print_grad)
        self.maxpool4.register_full_backward_hook(print_grad)
        self.maxpool5.register_full_backward_hook(print_grad)

        self.block1 = nn.Sequential(*[self.conv2d_1, self.relu, self.conv2d_2, self.relu,self.maxpool1, \
                                               self.conv2d_3, self.relu, self.conv2d_4, self.relu, self.maxpool2,  \
                                               self.conv2d_5, self.relu, self.conv2d_6, self.relu, self.conv2d_7, self.relu, self.maxpool3, \
                                               self.conv2d_8, self.relu, self.conv2d_9, self.relu, self.conv2d_10, self.relu, self.maxpool4, \
                                               self.conv2d_11, self.relu, self.conv2d_12, self.relu, self.conv2d_13, self.relu, self.maxpool5, \
                                             ]) 

    def forward(self, x):
        out = self.block1(x)
        print("out.shape", out.size())
        out = self.avgp(out)
        print("out.shape", out.size())
        out = self.flat(out)
        print("out.shape", out.size())
        out = self.fc1(out)
        print("out.shape", out.size())
        out = self.sft(out)
        print("out.shape", out.size())

    
        return out
        





class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = conv2d.TiledConv2d(in_channels=3, 
                                  out_channels=64, 
                                  kernel_size=(Kh,Kw),
                                  # bias = False,
                                  padding=(Ph,Pw),
                                  )  

        self.conv2d_2 = conv2d.TiledConv2d(in_channels=64, 
                                        out_channels=64, 
                                        kernel_size=(Kh,Kw),
                                        # bias = False,
                                        padding=(Ph,Pw),
                                        ) 
        self.mxp1 = maxpool2d.cMaxPool2d((2, 2), (2, 2))

        self.conv2d_3 = conv2d.TiledConv2d(in_channels=64, 
                                        out_channels=128, 
                                        kernel_size=(Kh,Kw),
                                        # bias = False,
                                        padding=(Ph,Pw),
                                        )
        self.conv2d_4 = conv2d.TiledConv2d(in_channels=128, 
                                        out_channels=128, 
                                        kernel_size=(Kh,Kw),
                                        # bias = False,
                                        padding=(Ph,Pw),
                                        )

        self.mxp2 = maxpool2d.cMaxPool2d((2, 2), (2, 2))

        self.conv2d_5 = conv2d.TiledConv2d(in_channels=128, 
                                  out_channels=256, 
                                  kernel_size=(Kh,Kw),
                                  # bias = False,
                                  padding=(Ph,Pw),
                                  )  

        self.conv2d_6 = conv2d.TiledConv2d(in_channels=256, 
                                        out_channels=256, 
                                        kernel_size=(Kh,Kw),
                                        # bias = False,
                                        padding=(Ph,Pw),
                                        ) 
        self.conv2d_7 = conv2d.TiledConv2d(in_channels=256, 
                                        out_channels=256, 
                                        kernel_size=(Kh,Kw),
                                        # bias = False,
                                        padding=(Ph,Pw),
                                        )
        self.mxp3 = maxpool2d.cMaxPool2d((2, 2), (2, 2))

        self.conv2d_8 = conv2d.TiledConv2d(in_channels=256, 
                                        out_channels=512, 
                                        kernel_size=(Kh,Kw),
                                        # bias = False,
                                        padding=(Ph,Pw),
                                        )
        self.conv2d_9 = conv2d.TiledConv2d(in_channels=512, 
                                        out_channels=512, 
                                        kernel_size=(Kh,Kw),
                                        # bias = False,
                                        padding=(Ph,Pw),
                                        )
        self.conv2d_10 = conv2d.TiledConv2d(in_channels=512, 
                                        out_channels=512, 
                                        kernel_size=(Kh,Kw),
                                        # bias = False,
                                        padding=(Ph,Pw),
                                        )

        self.mxp4 = maxpool2d.cMaxPool2d((2, 2), (2, 2))

        self.conv2d_11 = conv2d.TiledConv2d(in_channels=512, 
                                        out_channels=512, 
                                        kernel_size=(Kh,Kw),
                                        # bias = False,
                                        padding=(Ph,Pw),
                                        )
        self.conv2d_12 = conv2d.TiledConv2d(in_channels=512, 
                                        out_channels=512, 
                                        kernel_size=(Kh,Kw),
                                        # bias = False,
                                        padding=(Ph,Pw),
                                        )
        self.conv2d_13 = conv2d.TiledConv2d(in_channels=512, 
                                        out_channels=512, 
                                        kernel_size=(Kh,Kw),
                                        # bias = False,
                                        padding=(Ph,Pw),
                                        )

        #self.mxp5 = maxpool2d.cMaxPool2d((2, 2), (2, 2))
        self.relu = relu.cReLu()   


        self.avgp = gavgpool2d.cGAvgPool2d()
        self.sft = nn.Softmax(dim=-1)

        self.tsplit = tilesplit.TiledSplit()
        self.tcopy = tilecopy.TiledCopy()

        self.block1 = sequential.mSequential(*[self.tsplit, self.conv2d_1, self.relu, self.conv2d_2, self.relu,self.mxp1, \
                                                self.conv2d_3, self.relu, self.conv2d_4, self.relu, self.mxp2,  \
                                                self.conv2d_5, self.relu, self.conv2d_6, self.relu, self.conv2d_7, self.relu, self.mxp3, \
                                                self.conv2d_8, self.relu, self.conv2d_9, self.relu, self.conv2d_10, self.relu, self.mxp4, \
                                                self.conv2d_11, self.relu, self.conv2d_12, self.relu, self.conv2d_13, self.avgp]) #
        
        self.classifier = nn.Sequential(
            nn.Linear(512*gpool_size*gpool_size,num_classes),
            nn.Softmax(dim=1),
        )


    # I need input shape and #tiles
    def forward(self, x: torch.Tensor, N, C, H, W, nTh, nTw) -> torch.Tensor:
        model_device = next(self.parameters()).device
        N, C, oH, oW, shape_dict = shape_infer.shape_infer_sequence(self.block1, H, W, N, C)
        #x.requires_grad = True
        stream_structure = self.block1

        # alloc final output of checkpoint segment.
        out = torch.zeros(N, C, oH, oW, requires_grad=True).to(model_device)
        # print("!!!!!!!", out[0,0,0,0], model_device_local)
        # print("input shape", x.size())
        # print("nTh nTw", nTh, nTw)

        for i in range(0,nTh): 
            for j in range(0,nTw):
                coord = [i,j]
                print("coord", coord)
                # print("loop ...", out[0,0,0,0],  model_device_local)
                input_shape = (N,C,H,W)
                output_shape = (N,C,oH,oW)
                info = padding_calc.compute_info_beta([i,j], input_shape, output_shape, nTh, nTw, stream_structure, shape_dict,  model_device)
                
                #print("info", info[1][-11].model_device)
                # # have to accumulate here otherwise no bp.
                out += checkpoint.checkpoint(self.block1, x, info, stream_structure[1], model_device, [nTh, nTw])
                # temp = self.block1( x, info, stream_structure[1], model_device_local, [nTh, nTw])
                # print("temp ", temp[0,0,0,0], out[0,0,0,0])
                # out += temp

        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


import time

class RanTensorDataset(Dataset):
    def __init__(self, B, chanel,H,W):
        self.samples = []
        for i in range(0,B):
            input = torch.rand(chanel,H,W)
            self.samples.append(input)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def main():
    torch.set_printoptions(profile="full")
    torch.set_default_dtype(torch.float32)
    
    print("hello")
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = '172.18.0.1'
    os.environ['MASTER_PORT'] = '8888'

    model = Net()

    #input = torch.rand(batch,chanel,H,W, requires_grad = True)
    print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
    w1 = model.conv2d_1.weight.data
    w2 = model.conv2d_2.weight.data
    w3 = model.conv2d_3.weight.data
    w4 = model.conv2d_4.weight.data
    w5 = model.conv2d_5.weight.data
    w6 = model.conv2d_6.weight.data
    w7 = model.conv2d_7.weight.data
    w8 = model.conv2d_8.weight.data
    w9 = model.conv2d_9.weight.data
    w10 = model.conv2d_10.weight.data
    w11 = model.conv2d_11.weight.data
    w12 = model.conv2d_12.weight.data
    w13 = model.conv2d_13.weight.data
    b1 = model.conv2d_1.bias.data
    b2 = model.conv2d_2.bias.data
    b3 = model.conv2d_3.bias.data
    b4 = model.conv2d_4.bias.data
    b5 = model.conv2d_5.bias.data
    b6 = model.conv2d_6.bias.data
    b7 = model.conv2d_7.bias.data
    b8 = model.conv2d_8.bias.data
    b9 = model.conv2d_9.bias.data
    b10 = model.conv2d_10.bias.data
    b11 = model.conv2d_11.bias.data
    b12 = model.conv2d_12.bias.data
    b13 = model.conv2d_13.bias.data
    # fcw1 = model.fc1.weight.data
    # fcw2 = model.fc2.weight.data
    fcw1 = None
    fcw2= None

    #prepare reference
    model_ref =  Net_ref(w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, fcw1, fcw2
    , b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13).cuda()
    
    train_dataset = RanTensorDataset(batch, chanel,H,W)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                                )
    
    # total_step = len(train_loader)
    # print("epoch, step", args.epochs, total_step)
    # for epoch in range(args.epochs):
    #     for i, (images ) in enumerate(train_loader):
    #         images = images.cuda()
    #         images.requires_grad = True
    #         # Forward pass
    #         print("images size ", images.size(), images[0,0,0,0:10])
    #         print("reference itr", i)
    #         out_ref = model_ref(images)
    #         #print("output-t", outputs)
    #         # Backward and optimize
    #         out_ref.sum().backward()

    # print(out_ref.size())


    # run ddp
    print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
    print("RUN DDP")
    args.model = model
    args.td = train_dataset
    mp.spawn(train, nprocs=args.gpus, args=(args,))


    # print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
    # print("~~ check forward correctness ~~")
    # # print("out shape", out)
    # # print("out_ref ", out_ref)
    # # # not_same_num = correctness_check.point_wise_compare_4d(1,1,oH, oW, out, out_ref)
    # correctness_check.check_equal(out, out_ref, False)

    # # print("#### compare grad_in")
    # # # print("input ref grad", input_ref.grad)
    # # # print("input grad", input.grad)
    # # #not_same_num = correctness_check.point_wise_compare_4d(1,1,H, W, input.grad, input_ref.grad.to('cpu'))
    # # correctness_check.check_equal(input.grad, input_ref.grad, False)


    # print("#### compare w1")
    # correctness_check.check_equal(model_ref.conv2d_1.weight.grad, model.conv2d_1.weight.grad, False)

    # print("#### compare w2")
    # correctness_check.check_equal(model_ref.conv2d_2.weight.grad, model.conv2d_2.weight.grad, False)

    # print("#### compare w3")
    # correctness_check.check_equal(model_ref.conv2d_3.weight.grad, model.conv2d_3.weight.grad, False)

    # print("#### compare w4")
    # correctness_check.check_equal(model_ref.conv2d_4.weight.grad, model.conv2d_4.weight.grad, False)

    # print("#### compare w5")
    # correctness_check.check_equal(model_ref.conv2d_5.weight.grad, model.conv2d_5.weight.grad, False)

    # print("#### compare w6")
    # correctness_check.check_equal(model_ref.conv2d_6.weight.grad, model.conv2d_6.weight.grad, False)

    # print("#### compare w7")
    # correctness_check.check_equal(model_ref.conv2d_7.weight.grad, model.conv2d_7.weight.grad, False)

    # print("#### compare w8")
    # correctness_check.check_equal(model_ref.conv2d_8.weight.grad, model.conv2d_8.weight.grad, False)

    # print("#### compare w9")
    # correctness_check.check_equal(model_ref.conv2d_9.weight.grad, model.conv2d_9.weight.grad, False)

    # print("#### compare w10")
    # correctness_check.check_equal(model_ref.conv2d_10.weight.grad, model.conv2d_10.weight.grad, False)

    # print("#### compare w11")
    # correctness_check.check_equal(model_ref.conv2d_11.weight.grad, model.conv2d_11.weight.grad, False)

    # print("#### compare w12")
    # correctness_check.check_equal(model_ref.conv2d_12.weight.grad, model.conv2d_12.weight.grad, False)

    # print("#### compare w13")
    # correctness_check.check_equal(model_ref.conv2d_13.weight.grad, model.conv2d_13.weight.grad, False)



    # print("#### compare bias1")
    # correctness_check.check_equal(model_ref.conv2d_1.bias.grad, model.conv2d_1.bias.grad, False)
    
    # print("#### compare bias2")
    # correctness_check.check_equal(model_ref.conv2d_2.bias.grad, model.conv2d_2.bias.grad, False)
    
    # print("#### compare bias3")
    # correctness_check.check_equal(model_ref.conv2d_3.bias.grad, model.conv2d_3.bias.grad, False)
    
    # print("#### compare bias4")
    # correctness_check.check_equal(model_ref.conv2d_4.bias.grad, model.conv2d_4.bias.grad, False)
    
    # print("#### compare bias5")
    # correctness_check.check_equal(model_ref.conv2d_5.bias.grad, model.conv2d_5.bias.grad, False)
    
    # print("#### compare bias6")
    # correctness_check.check_equal(model_ref.conv2d_6.bias.grad, model.conv2d_6.bias.grad, False)
    
    # print("#### compare bias7")
    # correctness_check.check_equal(model_ref.conv2d_7.bias.grad, model.conv2d_7.bias.grad, False)
    
    # print("#### compare bias8")
    # correctness_check.check_equal(model_ref.conv2d_8.bias.grad, model.conv2d_8.bias.grad, False)
    
    # print("#### compare bias9")
    # correctness_check.check_equal(model_ref.conv2d_9.bias.grad, model.conv2d_9.bias.grad, False)
    
    # print("#### compare bias10")
    # correctness_check.check_equal(model_ref.conv2d_10.bias.grad, model.conv2d_10.bias.grad, False)
    
    # print("#### compare bias11")
    # correctness_check.check_equal(model_ref.conv2d_11.bias.grad, model.conv2d_11.bias.grad, False)
    
    # print("#### compare bias12")
    # correctness_check.check_equal(model_ref.conv2d_12.bias.grad, model.conv2d_12.bias.grad, False)
    
    # print("#### compare bias13")
    # correctness_check.check_equal(model_ref.conv2d_13.bias.grad, model.conv2d_13.bias.grad, False)




def train(gpu, args):
    train_dataset = args.td
    rank = args.nr * args.gpus + gpu
    print ("im rank ", rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    torch.manual_seed(0)

    # Wrap the model
    model = args.model
    model = model.to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    #TODO: need to double check correctness
    # model._set_static_graph()
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    #print("*** batch ", batch)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler = train_sampler,
                                               )
    
    total_step = len(train_loader)
    #print("epoch, step", args.epochs, total_step)
    for epoch in range(args.epochs):
        for i, (images ) in enumerate(train_loader):
            # Forward pass
            images.requires_grad = True
            # print("images size ", images.size())
            # print("reference itr", i, rank)

            out = model(images, images.size()[0], images.size()[1], images.size()[2], images.size()[3], nTh, nTw)
            print("output-t size()", out.size())
            # Backward and optimize 
            out.sum().backward()




if __name__=="__main__":
    main()