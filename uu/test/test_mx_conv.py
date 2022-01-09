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

gpool_size = 1
num_classes = 1024
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
        print("!!!!!!!", out.size())
        print("input shape", x.size())
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


def main():
    torch.set_printoptions(profile="full")
    torch.set_default_dtype(torch.float32)

    # add loss function here
    criterion = nn.MSELoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    memUsage = memory.MeasureMemory(device)
    #print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
    # print("==== real init ...")
    # our_initmem = memUsage.currentValue()
    # print(memory.MemSize(our_initmem))     
    # print(memUsage.available())

    model = Net().to(device)
    #print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")

    # print("==== our init ...")
    # our_initmem = memUsage.currentValue()
    # print(memory.MemSize(our_initmem))     
    # print(memUsage.available())


    ref_elapsed_fwd = 0
    ref_elapsed_bwk = 0
    start_time = time.time()    
    for i in range(0,1):
        input = torch.rand(batch,chanel,H,W, requires_grad = True)
        labels = torch.rand(batch, 1024).cuda()

        local_start = time.time() 
        out = model(input, 1, 3, H, W, nTh, nTw )
        # torch.cuda.synchronize()
        # ref_fwd_done = time.time()
        # ref_elapsed_fwd += (ref_fwd_done - local_start)


        print("==== fwd done ...")
        # ref_fwd_use = memUsage.currentValue()-our_initmem
        # print(memory.MemSize(ref_fwd_use) )    
        # print("avail ref",memUsage.available())
        # print("max ref", memUsage.maxx(), memUsage.maximumValue())

        loss = criterion(out, labels)
        loss.backward()
        # torch.cuda.synchronize()
        # ref_elapsed_bwk += (time.time()-ref_fwd_done)


    
    torch.cuda.synchronize()    
    ref_elapsed_total = time.time() - start_time
    print("done ref bkw")
    print("\n&& {}\n".format(ref_elapsed_total) )
    


    # print("==== our_bwd done ...")
    # our_bwd_use = memUsage.currentValue()-our_fwd_use
    # our_bwd_use_total = memUsage.currentValue()-our_initmem
    # print("our_bwd_use", memory.MemSize(our_bwd_use))   
    # print("our_bwd_use t", memory.MemSize(our_bwd_use_total))    
    # print("avail our",memUsage.available())
    # print("max our", memUsage.maxx(), memUsage.maximumValue())
    #print("input graad", input.grad[0,0,0,17])

import sys

if __name__=="__main__":
    Kh = 3
    Kw = 3
    Ph = 1
    Pw = 1
    chanel = 3
    batch = 1


    #roughly largest for this network 
    H = 1840
    W = 1840
   
    nTh = 1
    nTw = nTh
    main()