import torch
from  torch.nn.modules.pooling import _AvgPoolNd
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t

from uu.utils import correctness_check 


USE_DEFAULT_CTX = True
myctx_dict = {}
partial_sums_tile = []

class cGAvgPool2dFunction(torch.autograd.Function):
    # create a static variable
    @staticmethod
    def forward(ctx, *inputs):
        
        print("\n^^^^^cGavgPool2dFunction fwd")
        input = inputs[0]   # tiled input Do we need to get disjoint part??
        cinfo = inputs[1]   # current info
        ctx.info = cinfo
        uniq_id = inputs[2]
        is_ccheckpoint = inputs[3]
        f_info = cinfo[0][uniq_id]
        b_info = cinfo[1][uniq_id]

        if USE_DEFAULT_CTX:

            print(f_info.coord)
            print(f_info.numof_tiles)

            coord_h = f_info.coord[0]
            coord_w = f_info.coord[1]
            nTh = f_info.numof_tiles[0]
            nTw = f_info.numof_tiles[1]

            # have to find the info from previous op
            non_disjoint_tile_size_h = f_info.non_disjoint_tile_size[0]
            non_disjoint_tile_size_w = f_info.non_disjoint_tile_size[1]
            

            # sum all tile piece elements:
            dim_tp = (len(input.size())-2, len(input.size())-1)
            current_sum_overhw = input.sum(dim=dim_tp)
            partial_sums_tile.append(current_sum_overhw)

            if coord_h == nTh-1 and coord_w == nTw-1:
                # get all tiles partial sum
                assert(len(partial_sums_tile) == nTh*nTw)
                accum = partial_sums_tile[0]
                # accumlate all partial sum
                for i in range(1, len(partial_sums_tile)):
                    accum += partial_sums_tile[i]
                
                # averaging 
                # non-tiled input shape of avgpool
                input_size = [non_disjoint_tile_size_h*nTh, non_disjoint_tile_size_w*nTw]
                num_of_element = non_disjoint_tile_size_h*nTh * non_disjoint_tile_size_w*nTw
                out_value = accum / num_of_element 
                return out_value # tensor
            else:
                # none-last tile return None
                return None


    @staticmethod
    def backward(ctx, grad_output):
        if USE_DEFAULT_CTX:
            f_info = ctx.info[0][ctx.uniq_id]
            b_info = ctx.info[1][ctx.uniq_id]
            rev_g_depth = f_info.op_idex # must be last in our global seg
            if rev_g_depth == 0:
                grad_in = torch.zeros(b, c, h, w)

                print("grad_t1 original", grad_in, grad_in.size())
                for i in range(0,b):
                    for j in range (0,c):
                        grad_in[i,j,:,:] = grad_output[i,j]
        return grad_in, None, None, None
       
        
        


class cGAvgPool2d(_AvgPoolNd):
    def __init__(self, kernel_size: _size_2_t = None, stride: _size_2_t = None,
                 padding: _size_2_t = (0,0), dilation: _size_2_t = 1,
                 return_indices: bool = False, ceil_mode: bool = False,
                 is_ccheckpoint = False):
        super(cGAvgPool2d, self).__init__()
        self.is_ccheckpoint = is_ccheckpoint

    def forward(self, *inputs):
        if type (inputs[0]) == tuple:
            # to remove additional packing in tuple
            inputs = list(inputs[0])

        if len(inputs) == 2:
            input, info = inputs
            self.is_ccheckpoint = False
        elif len(inputs) == 3:
            input, info, is_ccheckpoint = inputs
            self.is_ccheckpoint = is_ccheckpoint
        else:
            print("missing info in cGAvgPool2d")
            assert False

        cgavgplool = cGAvgPool2dFunction.apply
        uniq_id = id(self)
        # info[0] is the forward meta info
        pi = info[0][uniq_id]

        # Gppoling must be the last op in a checkpoint segment
        assert (pi.op_idex == 0)
        out = cgavgplool(input, info, uniq_id, is_ccheckpoint)
        #print ("mxp FF", out.size())
        return out
       



 
