import torch
from  torch.nn.modules.pooling import _AvgPoolNd
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t
from uu.utils import correctness_check 
from typing import List, Optional

USE_DEFAULT_CTX = True
myctx_dict = {}
partial_sums_tile = []

class cGAvgPool2dFunction(torch.autograd.Function):
    # create a static variable
    @staticmethod
    def forward(ctx, *inputs):
        
        print("\n^^^^^cGavgPool2dFunction fwd")
        input = inputs[0]   # tiled input Do we need to get disjoint part??
        info = inputs[1]   # current info
        ctx.info = info
        uniq_id = inputs[2]
        is_ccheckpoint = inputs[3]
        f_info = info[0][uniq_id]
        b_info = info[1][uniq_id]


        

        if USE_DEFAULT_CTX:
            ctx.uniq_id = uniq_id
            b = input.size()[0]
            c = input.size()[1]
            h = input.size()[2]
            w = input.size()[3]
            ctx.b = b
            ctx.c = c
            ctx.h = h
            ctx.w = w
            coord_h = f_info.coord[0]
            coord_w = f_info.coord[1]
            nTh = f_info.numof_tiles[0]
            nTw = f_info.numof_tiles[1]

            # have to find the info from previous op
            # for input view non_disjoint
            previous_op_id = b_info.next_id
            pre_f_info = info[0][previous_op_id]

            non_disjoint_tile_size_h = pre_f_info.non_disjoint_tile_size[0]
            non_disjoint_tile_size_w = pre_f_info.non_disjoint_tile_size[1]

            #print("previous f_info", pre_f_info)
            
            print("input ", input.size())
            # sum all tile piece elements:
            dim_tp = (len(input.size())-2, len(input.size())-1)
            # extract non-disjoint
            fake_pi = info[0][-11]
            # tile_shape = fake_pi.cur_output_shape
            # tile_size = [tile_shape[0], tile_shape[1]]
            input_index = fake_pi.input_slice

            non_disj_index_h_b = coord_h*non_disjoint_tile_size_h
            non_disj_index_h_e = non_disj_index_h_b + non_disjoint_tile_size_h - 1
            non_disj_index_w_b = coord_w*non_disjoint_tile_size_w
            non_disj_index_w_e = non_disj_index_w_b + non_disjoint_tile_size_w - 1
            #non_disj_index = [non_disj_index_w_b, non_disj_index_w_e, non_disj_index_h_b, non_disj_index_h_e] #(l,r,t,b) 

            #relative offset
            l = abs(non_disj_index_w_b-input_index[0])
            r = abs(non_disj_index_w_e-input_index[1])
            t = abs(non_disj_index_h_b-input_index[2])
            b = abs(non_disj_index_h_e-input_index[3])
            in_tile_h = input.size()[2]
            in_tile_w = input.size()[3]

            
            # print("output_index", input_index)
            # print("non_disj_index", non_disj_index)
            # print("l, r, t, b", l, r, t, b)
            # print("input ext ", input[:,:, t:in_tile_h-b, l:in_tile_w-r])
            current_sum_overhw = input[:,:, t:in_tile_h-b, l:in_tile_w-r].sum(dim=dim_tp)
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
                num_of_element = non_disjoint_tile_size_h*nTh * non_disjoint_tile_size_w*nTw
                #?????
                ctx.num_of_element = num_of_element
                accum = accum[:, :, None,None]
                # print("accum", accum)
                # print("num_of_element", num_of_element)
                out_value = accum / num_of_element 
                print("out_value size", out_value.size())
                return out_value # tensor
            else:
                # none-last tile return None
                num_of_element = non_disjoint_tile_size_h*nTh * non_disjoint_tile_size_w*nTw
                ctx.num_of_element = num_of_element
                fake_out = torch.zeros(partial_sums_tile[0].size(), requires_grad=True).cuda()
                fake_out = fake_out[:, :, None,None]
                print("fake size", fake_out.size())
                return fake_out


    @staticmethod
    def backward(ctx, grad_output):
        print("^^^^^cGavgPool2dFunction bkw")
        if USE_DEFAULT_CTX:
            f_info = ctx.info[0][ctx.uniq_id]
            b_info = ctx.info[1][ctx.uniq_id]
            b = ctx.b
            c = ctx.c
            h = ctx.h
            w = ctx.w

            rev_g_depth = f_info.op_idex # must be last in our global seg
            if rev_g_depth == 0:
                grad_in = torch.zeros(b, c, h, w).cuda()

                print("grad_t1 original", grad_in.size())
                for i in range(0,b):
                    for j in range (0,c):
                        grad_in[i,j,:,:] = grad_output[i,j]/ctx.num_of_element
            
            
            print("grad_output", grad_output)
            print("grad_t1 original", grad_in)
        return grad_in, None, None, None
       
        
        


class cGAvgPool2d(_AvgPoolNd):
    def __init__(self, kernel_size: Optional[_size_2_t] = None, stride: Optional[_size_2_t] = None, padding: _size_2_t = 0,
                 ceil_mode: bool = False, count_include_pad: bool = True, divisor_override: Optional[int] = None,
                 is_ccheckpoint = False):

        super(cGAvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
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
       



 
