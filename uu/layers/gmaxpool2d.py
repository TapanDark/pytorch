import torch
from  torch.nn.modules.pooling import _MaxPoolNd
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t
from typing import List, Optional

from uu.utils import correctness_check 
import numpy
import math

class MMctx:
    def __init__(self):
        self.input = None
        self.kernel_size = None
        self.padding = None
        self.stride = None
        self.uniq_id = None
        self.info = None
        self.arg_max = None

USE_DEFAULT_CTX = True
myctx_dict = {}
partial_max_tile_all = None
# BK_FLAG = False

ROUND_PRECI = 16
FINAL_res = None
# FINAL_ind = None
META_tile = None

class cGMaxPool2dFunction(torch.autograd.Function):
    # create a static variable
    @staticmethod
    def forward(ctx, *inputs):
        #print("\n^^^^^cGmaxPool2dFunction fwd")
        #global BK_FLAG
        global FINAL_res
        # global FINAL_ind
        global META_tile
        global partial_max_tile_all
        input = inputs[0]   # tiled input Do we need to get dijoint part??
        #kernel_size = inputs[1]
        stride = inputs[2]
        padding = inputs[3]
        info = inputs[4] # current info
        ctx.info = info
        uniq_id = inputs[5]
        is_ccheckpoint = inputs[6]
        ctx.model_device = inputs[7]

        f_info = info[0][uniq_id]
        b_info = info[1][uniq_id]

        # print(f_info)
        # print(b_info)

        if is_ccheckpoint: 
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
            ctx.coord_h = coord_h
            ctx.coord_w = coord_w
            nTh = f_info.numof_tiles[0]
            nTw = f_info.numof_tiles[1]
            
            if coord_h == 0 and coord_w == 0:
                # partial_max_tile = []
                # assert(len(partial_max_tile) == 0), "init partial_max_tile size coord {} {} --> {}".format(coord_h, coord_w, len(partial_sums_tile))
                del partial_max_tile_all
                partial_max_tile_all = numpy.empty((ctx.b, ctx.c),dtype=object)
                for itr_b in range(ctx.b):
                    # select the largest among all partial max
                    for itr_c in range(ctx.c):
                        partial_max_tile_all[itr_b][itr_c]=[]

            

            for itr_b in range(ctx.b):
                # select the largest among all partial max
                for itr_c in range(ctx.c):
                    #print("inout {} b {} c {}".format(input.size(), itr_b, itr_c))
                    tmp_slice = input[itr_b,itr_c,:,:]
                    tmp_slice = tmp_slice[None,None, :]
                    #print("slice shape (2D ??)" ,tmp_slice.size())
                    out = F.max_pool2d(tmp_slice, (h,w), stride, padding, return_indices=True)
                    out_value = out[0]
                    out_index = out[1]
                    #print("out v", out_value, out_index)
                    partial_max_tile_all[itr_b][itr_c].append([out_value, out_index, (coord_h, coord_w), (ctx.h, ctx.w)])



            # #print("kernel_size ",kernel_size)
            # # print("cord {},  H {} W {} Th {} Tw {}"
            # #     .format((coord_h, coord_w),  ctx.h, ctx.w, nTh, nTw))
            
            if coord_h == nTh-1 and coord_w == nTw-1:
                #print("*** ",partial_max_tile_all[0][0])
                del META_tile
                META_tile = numpy.empty((ctx.b, ctx.c),dtype=object)
                FINAL_res = torch.zeros(b,c,1,1, requires_grad=True).to(ctx.model_device)
                for itr_b in range(ctx.b):
                    # select the largest among all partial max
                    # have to comapre each B and each C
                    for itr_c in range(ctx.c):
                        # get current b,c list of partial max results
                        partial_max_tile = partial_max_tile_all[itr_b][itr_c]
                        #print("len(partial_max_tile) ", len(partial_max_tile), nTh, nTw)
                        assert(len(partial_max_tile) == nTh*nTw) # each HW surface is in nTh*nTw tiles

                        for i in range(0, len(partial_max_tile)):
                            
                            if i == 0:
                                # init maxmax and cordTh, cordTw
                                # init Meta info as empty list
                                META_tile[itr_b][itr_c] = []
                                maxmax = partial_max_tile[i][0]
                                maxmax = round(maxmax.item(), ROUND_PRECI)
            
                                hh = partial_max_tile[i][2][0]  
                                ww = partial_max_tile[i][2][1]
                                max_h_w = [hh, ww] # max tile coord
                                maxindi = partial_max_tile[i][1]
                                cur_h = partial_max_tile[i][3][0]
                                cur_w = partial_max_tile[i][3][1]
                                META_tile[itr_b][itr_c].append( (max_h_w, maxindi, [cur_h, cur_w]) )
                            
                            # to overcome precision issue
                            if round(partial_max_tile[i][0].item(), ROUND_PRECI) > maxmax:
                                #print("abs large ", i )
                                maxmax = round(partial_max_tile[i][0].item(), ROUND_PRECI)
                                maxindi = partial_max_tile[i][1]
                                #full_precision_max = partial_max_tile[i][0].item()

                                cur_h = partial_max_tile[i][3][0]
                                cur_w = partial_max_tile[i][3][1]
                                # max_position_w = maxindi.item()  % cur_w
                                # max_position_h = maxindi.item()  // cur_w
                                #assert (max_position_w < cur_w and max_position_w>= 0 and max_position_h >=0 and max_position_h<cur_h), "abs indi {} h {} w {} ph{} - pw{}".format(maxindi,cur_h, cur_w, max_position_h, max_position_w)

                                # replacing largest t_h and t_w
                                hh = partial_max_tile[i][2][0]  
                                ww = partial_max_tile[i][2][1]

                                # need to clean up buffer
                                META_tile[itr_b][itr_c]=[]
                                meta_inf = ([hh,ww], maxindi, [cur_h, cur_w])
                                META_tile[itr_b][itr_c].append(meta_inf)
                                #print("{} {} --> {}", itr_b, itr_c,META_tile[itr_b][itr_c] )
                            elif round(partial_max_tile[i][0].item(), ROUND_PRECI) == maxmax:
                                #print("equal caching ", i)
                                maxindi = partial_max_tile[i][1]

                                cur_h = partial_max_tile[i][3][0]
                                cur_w = partial_max_tile[i][3][1]


                                # max_position_w = maxindi.item()  % cur_w
                                # max_position_h = maxindi.item()  // cur_w
                                #assert (max_position_w < cur_w and max_position_w>= 0 and max_position_h >=0 and max_position_h<cur_h), "abs indi {} h {} w {} ph{} - pw{}".format(maxindi,cur_h, cur_w, max_position_h, max_position_w)

                                hh = partial_max_tile[i][2][0]  
                                ww = partial_max_tile[i][2][1]
                                #print("cacheing ", hh, ww)
                                # need to also put it in the 
                                # tile size is not same, so need to store cur_h, cur_w for tile h/w extent
                                meta_inf = ([hh,ww], maxindi, [cur_h, cur_w])
                                META_tile[itr_b][itr_c].append(meta_inf)
                                #print("{} {} --> {}", itr_b, itr_c,META_tile[itr_b][itr_c] )
                        
                        FINAL_res[itr_b][itr_c][0][0] = maxmax
                        # if (itr_b == 0 and itr_c == 0):
                        #     print(" 00 MAX", maxmax, META_tile[itr_b][itr_c])

                #print("FINAL_res ", FINAL_res.size())
                return FINAL_res # tensor
            else:
                # if not the last tile, return fake 0 tensor
                fake_out = torch.zeros(b,c,1,1, requires_grad=True).to(ctx.model_device)
                # print("partial_max_tile[0][0] size", partial_max_tile[0][0].size())
                #print("fake size", fake_out.size())
                return fake_out
        else:
            # BK skip recomputation
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
            #ctx.metainfo = META_tile

            ctx.coord_h = coord_h
            ctx.coord_w = coord_w
            #print("recomp", ctx.coord_h, ctx.coord_w)
            return FINAL_res


    @staticmethod
    def backward(ctx, grad_output):
        # print("\n^^^^^cGGMaxPool2dFunction bwd")
        #print("grad_output", grad_output.size())
        #print("META_tile", META_tile)
        if USE_DEFAULT_CTX:
            f_info = ctx.info[0][ctx.uniq_id]
            b = ctx.b
            c = ctx.c
            h = ctx.h
            w = ctx.w
          
            rev_g_depth = f_info.op_idex # must be last in our global seg
            if rev_g_depth == 0:
                grad_in = torch.zeros(b, c, h, w).to(ctx.model_device)
                # print("grad_in", grad_in.size())
                # print("bkkk", ctx.coord_h, ctx.coord_w, h, w)

                for i in range(0,b):
                    for j in range (0,c):
                        #print("??",i, j, META_tile[i][j])
                        for pp in META_tile[i][j]: # get currnt b,c maxmax
                            if pp[0][0] == ctx.coord_h and pp[0][1] == ctx.coord_w:
                                # if the current tile contains the maxmax, restore back
                                #print("large tile at {}{} - {} {}".format(i, j, ctx.coord_h, ctx.coord_w))
                                indi = pp[1]
                               

                                cur_h = pp[2][0]
                                cur_w = pp[2][1]
  
                                max_position_w = indi.item()  % cur_w
                                max_position_h = indi.item()  // cur_w
                                
                                # print("-->b {} c {} mxh {} mxw {} gradvalue {}".format(i,j, max_position_h, max_position_w, grad_output[i][j][0][0]))
                                # print( "-->indi {} cooH {} cooW {} h {} w {} ph {} - pw {}".format(indi,ctx.coord_h, ctx.coord_w , cur_h, cur_w, max_position_h, max_position_w))
                                # assert (max_position_w < cur_w and max_position_w>= 0 and max_position_h >=0 and max_position_h<cur_h)

                                #TODO: 4D tensor specified and we know it is 1x1 in HW
                                grad_in[i][j][max_position_h][max_position_w] = grad_output[i][j][0][0]

                                # if (j <= 1):
                                #     print(grad_in)
            
        return grad_in, None, None, None, None, None, None, None
        
        


class cGMaxPool2d(_MaxPoolNd):
    def __init__(self, kernel_size: Optional[_size_2_t] = None, stride: Optional[_size_2_t] = None,
                 padding: _size_2_t = (0,0), dilation: _size_2_t = 1,
                 return_indices: bool = False, ceil_mode: bool = False,
                 is_ccheckpoint = False, #mdepth = 1, num_maxp = 1
                 ):
        super(cGMaxPool2d, self).__init__(kernel_size, stride,
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
            print("missing info in cGMaxPool2d")
            assert False

        cgMaxplool = cGMaxPool2dFunction.apply
        uniq_id = id(self)
        pi = info[0][uniq_id]
        model_device = info[1][-11].model_device


        # Gppoling must be the last op in a checkpoint segment
        assert (pi.op_idex == 0)
        out = cgMaxplool(input, self.kernel_size, self.stride,
                        self.padding, info, uniq_id, is_ccheckpoint, model_device)
        #print ("mxp FF", out.size())
        return out
       



 
