import torch
from  torch.nn.modules.pooling import _AvgPoolNd
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t

from uu.utils import correctness_check 




myctx_dict = {}
partial_sums_tile = []

class cGAvgPool2dFunction(torch.autograd.Function):
    # create a static variable
    @staticmethod
    def forward(ctx, *inputs):
        #print("\n^^^^^cMaxPool2dFunction fwd")
        input = inputs[0]   # tiled input Do we need to get dijoint part??
        kernel_size = inputs[1]
        stride = inputs[2]
        padding = inputs[3]
        cinfo = inputs[4] # current info
        ctx.info = cinfo
        uniq_id = inputs[5]
        is_ccheckpoint = inputs[6]

        print(cinfo.coord)
        print(cinfo.numof_tiles)


        coord_h = cinfo.coord[0]
        coord_w = cinfo.coord[1]
        nTh = cinfo.numof_tiles[0]
        nTw = cinfo.numof_tiles[1]

        # have to find the info from previous op
        non_disjoint_tile_size_h = cinfo.non_disjoint_tile_size[0]
        non_disjoint_tile_size_w = cinfo.non_disjoint_tile_size[1]
        

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


        
        
    
    # @staticmethod
    # def backward(ctx, grad_output):
    #     # print("\n^^^^^cMaxPool2dFunction bwd")
    #     # #case1
    #     # if ctx.input.is_cuda:
    #     #     grad_in = maxpool_2d_bkw_cuda.backward(grad_output, ctx.input, ctx.kernel_size, ctx.stride, ctx.padding, (1,1), False, ctx.arg_max)
    #     # else:
    #     #     grad_in = maxpool_2d_bkw_cpp.backward(grad_output, ctx.input, ctx.kernel_size, ctx.stride, ctx.padding, (1,1), False, ctx.arg_max)
        
    #     myctx = myctx_dict[ctx.uniq_id]

    #     # print("input size", myctx.input.size())
    #     # print("grad_out size",grad_output.size())
    #     #print("grad_out ",grad_output)
    #     # print("arg size",myctx.arg_max.size())


    #     f_info = myctx.info[0][ctx.uniq_id]
    #     b_info = myctx.info[1][ctx.uniq_id]
    #     rev_g_depth = f_info.op_idex
    #     g_depth = b_info.op_idex    # global depth
    #     if g_depth == 0: 
    #         grad_in = torch._C._nn.max_pool2d_with_indices_backward(grad_output, myctx.input, myctx.kernel_size, myctx.stride, myctx.padding, (1,1), False, myctx.arg_max)
    #         # reshape to tile size before leaving the segment

    #     elif rev_g_depth == 0:
    #         # the last stage in regular order
    #         new_grad_out = grad_output[:, :, b_info.input_slice[2]:b_info.input_slice[3]+1, b_info.input_slice[0]:b_info.input_slice[1]+1]
    #         #print("new_grad_out", new_grad_out.size())
    #         grad_in = torch._C._nn.max_pool2d_with_indices_backward(new_grad_out, myctx.input, myctx.kernel_size, myctx.stride, myctx.padding, (1,1), False, myctx.arg_max)
    #     else:
    #         grad_in = torch._C._nn.max_pool2d_with_indices_backward(grad_output, myctx.input, myctx.kernel_size, myctx.stride, myctx.padding, (1,1), False, myctx.arg_max)
    
    #     #print("##############grad_in in maxp", grad_in.size()) 
    #     #print("grad in", grad_in)
    #     torch.cuda.empty_cache()
    #     return grad_in, None, None, None, None, None, None
        
        


class cGAvgPool2d(_AvgPoolNd):
    def __init__(self, kernel_size: _size_2_t, stride: _size_2_t = None,
                 padding: _size_2_t = (0,0), dilation: _size_2_t = 1,
                 return_indices: bool = False, ceil_mode: bool = False,
                 is_ccheckpoint = False, #mdepth = 1, num_maxp = 1
                 ):
        super(cGAvgPool2d, self).__init__(kernel_size, stride,
                 padding, dilation, return_indices, ceil_mode)
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
        out = cgavgplool(input, self.kernel_size, self.stride,
                        self.padding, info, uniq_id, is_ccheckpoint)
        #print ("mxp FF", out.size())
        return out
       



 
