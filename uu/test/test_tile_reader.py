import torch
import torch.nn as nn
from torch.cuda import init
from uu.utils import shape_infer 
from uu.utils import padding_calc
from uu.layers import maxpool2d, conv2d, sequential, tilesplit, tilecopy, relu, gavgpool2d, gmaxpool2d, move
from torch.nn.parameter import Parameter
from uu.utils import correctness_check 
from uu.utils import memory 
from uu.utils import checkpoint
import time
from torch.nn.parameter import Parameter


Kh = 3
Kw = 3
Ph = 1
Pw = 1
chanel = C = 3
batch = N = 1

H = 512
W = 512
oH = H//32
oW = W//32
nTh = 4
nTw = 4

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

        self.relu = nn.ReLU()
        self.maxpool5 = nn.MaxPool2d((2,2), (2,2))
        
        
        self.maxgp = nn.MaxPool2d(oH, stride=1)
        self.avggp = nn.AvgPool2d(oH, stride=1)
        self.flat = nn.Flatten()
        in_feature = 512
        self.fc1 = nn.Linear(in_feature, 1024, bias=False)
        self.sft = nn.Softmax(dim=1)
        
        
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
       
        # self.conv2d_1.register_forward_hook(print_activa)
        # self.conv2d_3.register_forward_hook(print_activa)
        # self.conv2d_4.register_forward_hook(print_activa)
        # self.conv2d_2.register_forward_hook(print_activa)
        # self.conv2d_5.register_forward_hook(print_activa)
        # self.conv2d_6.register_forward_hook(print_activa)
        # self.conv2d_7.register_forward_hook(print_activa)
        # self.conv2d_8.register_forward_hook(print_activa)
        # self.conv2d_9.register_forward_hook(print_activa)
        # self.conv2d_10.register_forward_hook(print_activa)
        # self.conv2d_11.register_forward_hook(print_activa)
        # self.conv2d_12.register_forward_hook(print_activa)
        # self.conv2d_13.register_forward_hook(print_activa)
        # self.maxpool1.register_forward_hook(print_activa)
        # self.maxpool2.register_forward_hook(print_activa)
        # self.maxpool3.register_forward_hook(print_activa)
        # self.maxpool4.register_forward_hook(print_activa)
        # self.maxpool5.register_forward_hook(print_activa)
        
        # self.conv2d_1.register_full_backward_hook(print_grad)
        # self.conv2d_2.register_full_backward_hook(print_grad)
        # self.conv2d_3.register_full_backward_hook(print_grad)
        # self.conv2d_4.register_full_backward_hook(print_grad)
        # self.conv2d_5.register_full_backward_hook(print_grad)
        # self.conv2d_6.register_full_backward_hook(print_grad)
        # self.conv2d_7.register_full_backward_hook(print_grad)
        # self.conv2d_8.register_full_backward_hook(print_grad)
        # self.conv2d_9.register_full_backward_hook(print_grad)
        # self.conv2d_10.register_full_backward_hook(print_grad)
        # self.conv2d_11.register_full_backward_hook(print_grad)
        # self.conv2d_12.register_full_backward_hook(print_grad)
        # self.conv2d_13.register_full_backward_hook(print_grad)
        # self.maxpool1.register_full_backward_hook(print_grad)
        # self.maxpool2.register_full_backward_hook(print_grad)
        # self.maxpool3.register_full_backward_hook(print_grad)
        # self.maxpool4.register_full_backward_hook(print_grad)
        # self.maxpool5.register_full_backward_hook(print_grad)

        self.block1 = nn.Sequential(*[self.conv2d_1, self.relu, self.conv2d_2, self.relu,self.maxpool1, \
                                               self.conv2d_3, self.relu, self.conv2d_4, self.relu, self.maxpool2,  \
                                               self.conv2d_5, self.relu, self.conv2d_6, self.relu, self.conv2d_7, self.relu, self.maxpool3, \
                                               self.conv2d_8, self.relu, self.conv2d_9, self.relu, self.conv2d_10, self.relu, self.maxpool4, \
                                               self.conv2d_11, self.relu, self.conv2d_12, self.relu, self.conv2d_13, self.relu, self.maxpool5, \
                                             ]) 

    def forward(self, x):
        out = self.block1(x)
        out = self.maxgp(out)

        print("out.shape", out.size())
        out = self.flat(out)
        print("out.shape", out.size())
        out = self.fc1(out)
        print("out.shape", out.size())
        out = self.sft(out)
        print("out.shape", out.size())

    
        return out

class VGG_TT_TILE(nn.Module):
    """
    This model uses the convolution layers from VGG16.
    Followed by a global max pool (so it is independant of input image size).
    Following by a single linear layer and softmax.
    """
    def __init__(
        self, num_classes: int = 1000, init_weights: bool = False, dropout: float = 0.5
    ) -> None:
        super().__init__()
        gpool_size = 1
        Kh = 3
        Kw = 3
        Ph = 1
        Pw = 1

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

        self.mxp5 = maxpool2d.cMaxPool2d((2, 2), (2, 2))
        self.relu = relu.cReLu()   


        self.gavgp = gavgpool2d.cGAvgPool2d()
        self.gmaxp = gmaxpool2d.cGMaxPool2d()

        self.df_maxgp = nn.MaxPool2d(oH, stride=1)
        self.df_avggp = nn.AvgPool2d(oH, stride=1)
        #self.gmaxp.register_full_backward_hook(print_grad)

        self.tsplit = tilesplit.TiledSplit()
        self.tcopy = tilecopy.TiledCopy()
        self.movnode = move.dMove()
        self.block1 = sequential.mSequential(*[self.movnode, self.conv2d_1, self.relu, self.conv2d_2, self.relu,self.mxp1, \
                                                self.conv2d_3, self.relu, self.conv2d_4, self.relu, self.mxp2,  \
                                                self.conv2d_5, self.relu, self.conv2d_6, self.relu, self.conv2d_7, self.relu, self.mxp3, \
                                                self.conv2d_8, self.relu, self.conv2d_9, self.relu, self.conv2d_10, self.relu, self.mxp4, \
                                                self.conv2d_11, self.relu, self.conv2d_12, self.relu, self.conv2d_13, self.relu, self.mxp5, self.gmaxp]) #

        
        in_feature = 512
        self.fc1 = nn.Linear(in_feature, 1024, bias=False)
        self.sft = nn.Softmax(dim=1)
        
        # self.classifier = nn.Sequential(
        #     nn.Linear(512*gpool_size*gpool_size,num_classes),
        #     nn.Softmax(dim=1),
        # )

    # I need input shape and #tiles
    def forward(self, x: torch.Tensor, N, C, H, W, nTh, nTw, rank) -> torch.Tensor:
        model_device = next(self.parameters()).device
        N, C, oH, oW, shape_dict = shape_infer.shape_infer_sequence(self.block1, H, W, N, C)
        #x.requires_grad = True
        
        stream_structure = self.block1
        tile_list = []
        input_shape = (N,C,H,W)
        output_shape = (N,C,oH,oW)
        for i in range(0,nTh): 
            for j in range(0,nTw):
                print("##coord", [i,j])
                info = padding_calc.compute_info_beta([i,j], input_shape, output_shape, nTh, nTw, stream_structure, shape_dict,  model_device)
                first_op_in_seg = id(stream_structure[1])
                # print("first_op_in_seg", first_op_in_seg)

                pi = info[0][first_op_in_seg]
                slice_info = pi.input_slice  #(l,r,t,b)
                tile_list.append( (slice_info, (i, j)) ) #(coordinates, opaque_obj)
                del info

       
        #TODO
        #x.setTiles(tile_list)
        #tile_loader = x.getTileLoader()

        #FOR dubugging; fake tile_loader
        tile_loader = []
        for elm in tile_list:
            slice_info = elm[0]
            coord = elm[1]

            print("--itr ", slice_info, coord)
            temp_view = x[:, :, slice_info[2]:slice_info[3]+1, slice_info[0]:slice_info[1]+1]      #NCHW
            tile_loader.append( (coord, temp_view) )



        # alloc final output of checkpoint segment.
        out = torch.zeros(N, C, oH, oW, requires_grad=True).to(model_device)
        # print("!!!!!!!", out[0,0,0,0])
        print("input shape{} out shape{} nth {} ntw {} rank {}".format(x.size(), out.size(), nTh, nTw, rank))
        
        # print("nTh nTw", nTh, nTw)


        #out = torch.zeros(N, C, oH, oW, requires_grad=True).cuda()
        isTileReader = True
        for coord, tile in tile_loader:
            print("coord", coord)
            info = padding_calc.compute_info_beta([coord[0],coord[1]], input_shape, output_shape, nTh, nTw, stream_structure, shape_dict,  model_device)

            # # have to accumulate here otherwise no bp.
            #TODO: potential issue for the input.grad computation.
            # It will be computed now; since we are avoiding useing a customized op to dispatch input....
            out += checkpoint.checkpoint(self.block1, tile, info, model_device, isTileReader)
            del info

        print("FF", out.size())   
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.sft(out)
        return out


def main():
    torch.set_printoptions(profile="full")
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG_TT_TILE().to(device)

    criterion =  nn.MSELoss()
    labels = torch.rand(batch, 1024).cuda()

    input = torch.rand(batch,chanel,H,W, requires_grad = True)
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

  
    fcw1 = model.fc1.weight.data
    # fcw2 = model.fc2.weight.data
    fcw2= None
    
    model_ref =  Net_ref(w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, fcw1, fcw2
    , b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13).to(device)
    
    start_time = time.time()
    input_ref = input.data.clone() 
    input_ref = input_ref.cuda()
    input_ref.requires_grad = True
    out_ref = model_ref(input_ref)
    ref_elapsed_fwd = time.time() - start_time

    loss1 = criterion(out_ref, labels)
    print("loss1", loss1)
    loss1.backward()
    
    ref_elapsed_total = time.time() - start_time
    print("done ref")
    print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n", ref_elapsed_fwd, ref_elapsed_total)
    
    
    start_time = time.time()

    out = model(input, N, C, H, W, nTh, nTw, 0 )
    our_elapsed_fwd = time.time() - start_time
   
    loss2 = criterion(out, labels)
    print("loss2", loss2)
    loss2.backward()
    our_elapsed_total = time.time() - start_time

    print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n", our_elapsed_fwd, our_elapsed_total)
    print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
    print("~~ check forward correctness ~~")
    # print("out shape", out)
    # print("out_ref ", out_ref)
    correctness_check.check_equal(out, out_ref, False)

    # print("#### compare grad_in")
    # # print("input ref grad", input_ref.grad)
    # # print("input grad", input.grad)
    # #not_same_num = correctness_check.point_wise_compare_4d(1,1,H, W, input.grad, input_ref.grad.to('cpu'))
    # correctness_check.check_equal(input.grad, input_ref.grad, False)


    print("#### compare w1")
    correctness_check.check_equal(model_ref.conv2d_1.weight.grad, model.conv2d_1.weight.grad, False)

    print("#### compare w2")
    correctness_check.check_equal(model_ref.conv2d_2.weight.grad, model.conv2d_2.weight.grad, False)

    print("#### compare w3")
    correctness_check.check_equal(model_ref.conv2d_3.weight.grad, model.conv2d_3.weight.grad, False)

    print("#### compare w4")
    correctness_check.check_equal(model_ref.conv2d_4.weight.grad, model.conv2d_4.weight.grad, False)

    print("#### compare w5")
    correctness_check.check_equal(model_ref.conv2d_5.weight.grad, model.conv2d_5.weight.grad, False)

    print("#### compare w6")
    correctness_check.check_equal(model_ref.conv2d_6.weight.grad, model.conv2d_6.weight.grad, False)

    print("#### compare w7")
    correctness_check.check_equal(model_ref.conv2d_7.weight.grad, model.conv2d_7.weight.grad, False)

    print("#### compare w8")
    correctness_check.check_equal(model_ref.conv2d_8.weight.grad, model.conv2d_8.weight.grad, False)

    print("#### compare w9")
    correctness_check.check_equal(model_ref.conv2d_9.weight.grad, model.conv2d_9.weight.grad, False)

    print("#### compare w10")
    correctness_check.check_equal(model_ref.conv2d_10.weight.grad, model.conv2d_10.weight.grad, False)

    print("#### compare w11")
    correctness_check.check_equal(model_ref.conv2d_11.weight.grad, model.conv2d_11.weight.grad, False)

    print("#### compare w12")
    correctness_check.check_equal(model_ref.conv2d_12.weight.grad, model.conv2d_12.weight.grad, False)

    print("#### compare w13")
    correctness_check.check_equal(model_ref.conv2d_13.weight.grad, model.conv2d_13.weight.grad, False)

    print(model_ref.conv2d_1.weight.grad[0,0,:,:])

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

    # print(model_ref.conv2d_1.weight.grad[0,0,:,:])




if __name__=="__main__":
    main()