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
partial_max_tile = []
BK_FLAG = False

FINAL_res = None
# FINAL_ind = None
META_tile = None

class cGMaxPool2dFunction(torch.autograd.Function):
    # create a static variable
    @staticmethod
    def forward(ctx, *inputs):
        #print("\n^^^^^cGmaxPool2dFunction fwd")
        global BK_FLAG
        global FINAL_res
        # global FINAL_ind
        global META_tile
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

        if USE_DEFAULT_CTX and not BK_FLAG:
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

            # for input view non_disjoint
            # previous_op_id = b_info.next_id
            # pre_f_info = info[0][previous_op_id]

            # non_disjoint_tile_size_h = pre_f_info.non_disjoint_tile_size[0]
            # non_disjoint_tile_size_w = pre_f_info.non_disjoint_tile_size[1]

            # dim_tp = (len(input.size())-2, len(input.size())-1)
            # # extract non-disjoint
            # fake_pi = info[0][-11]
            # tile_shape = fake_pi.cur_output_shape
            # tile_size = [tile_shape[0], tile_shape[1]]
            # input_index = fake_pi.input_slice

            # non_disj_index_h_b = coord_h*non_disjoint_tile_size_h
            # non_disj_index_h_e = non_disj_index_h_b + non_disjoint_tile_size_h - 1
            # non_disj_index_w_b = coord_w*non_disjoint_tile_size_w
            # non_disj_index_w_e = non_disj_index_w_b + non_disjoint_tile_size_w - 1
            #non_disj_index = [non_disj_index_w_b, non_disj_index_w_e, non_disj_index_h_b, non_disj_index_h_e] #(l,r,t,b) 

            #relative offset
            # l = abs(non_disj_index_w_b-input_index[0])
            # r = abs(non_disj_index_w_e-input_index[1])
            # t = abs(non_disj_index_h_b-input_index[2])
            # b = abs(non_disj_index_h_e-input_index[3])
            # in_tile_h = input.size()[2]
            # in_tile_w = input.size()[3]

            # disjoint view of input:= input[:,:, t:in_tile_h-b, l:in_tile_w-r]
            #TODO: is checkpoint????
            #input_disjoint = input[:,:, t:in_tile_h-b, l:in_tile_w-r]
            ## TODO:?? I think no disjoint need here for maxp
            # print("l, r, t, b", l, r, t, b)
            # print("input_disjoint ext ", input_disjoint[:,:, t:in_tile_h-b, l:in_tile_w-r].size())
            # local tile do gloable maxpool, kernel size == h,w extend
            # kernel_size = (input_disjoint.size()[2], input_disjoint.size()[2])


            #print("input size", input.size(), input)
            out = F.max_pool2d(input, (h,w), stride, padding, return_indices=True)
            out_value = out[0]
            out_index = out[1]

            #print("kernel_size ",kernel_size)
            # print("out_value ",out_value, out_value.size())
            # print("out_index ",out_index, out_index.size())

            # store temp (space is 2 element per tile)
            partial_max_tile.append([out_value, out_index, (coord_h, coord_w)] )

            if coord_h == nTh-1 and coord_w == nTw-1:
                
                # META_tile / FINAL_ind  should be an [B][C][List]
                META_tile = numpy.empty((ctx.b, ctx.c),dtype=object)
                #FINAL_ind = numpy.empty((ctx.b, ctx.c),dtype=object)

                BK_FLAG = True # it must be the end of the FWD pass, so set the flag to avoid recomputation
                assert(len(partial_max_tile) == nTh*nTw)
                maxmax = partial_max_tile[0][0]
                # maxindex = partial_max_tile[0][1]
                hh = partial_max_tile[0][2][0]  
                ww = partial_max_tile[0][2][1]
                max_h_w = [hh, ww]
               

                for itr_b in range(ctx.b):
                    # select the largest among all partial max
                    # have to comapre each B and each C
                    for itr_c in range(ctx.c):
                        for i in range(0, len(partial_max_tile)):
                            #print("i ", i)
                            # init
                            if i == 0:
                                META_tile[itr_b][itr_c] = []
                                maxindi = partial_max_tile[i][1][itr_b][itr_c][0][0]
                                META_tile[itr_b][itr_c].append( (max_h_w, maxindi) )
                            else:
                                # first 2 index locates the tile; 
                                if partial_max_tile[i][0][itr_b][itr_c][0][0] > maxmax[itr_b][itr_c][0][0]:
                                    maxmax[itr_b][itr_c][0][0] = partial_max_tile[i][0][itr_b][itr_c][0][0]
                                    maxindi = partial_max_tile[i][1][itr_b][itr_c][0][0]

                                    #print("abs large ")
                                    # replacing largest t_h and t_w
                                    hh = partial_max_tile[i][2][0]  
                                    ww = partial_max_tile[i][2][1]

                                    # need to clean up buffer
                                    META_tile[itr_b][itr_c].clear()
                                    meta_inf = ([hh,ww], maxindi)
                                    META_tile[itr_b][itr_c].append(meta_inf)
                                    #print("{} {} --> {}", itr_b, itr_c,META_tile[itr_b][itr_c] )
                                elif partial_max_tile[i][0][itr_b][itr_c][0][0] == maxmax[itr_b][itr_c][0][0]:
                                    #print("equal caching ")
                                    maxindi = partial_max_tile[i][1][itr_b][itr_c][0][0]
                                    hh = partial_max_tile[i][2][0]  
                                    ww = partial_max_tile[i][2][1]
                                    # need to also put it in the 
                                    meta_inf = ([hh,ww], maxindi)
                                    META_tile[itr_b][itr_c].append(meta_inf)
                                    #print("{} {} --> {}", itr_b, itr_c,META_tile[itr_b][itr_c] )


                FINAL_res = maxmax
                #FINAL_ind = maxindex
                #ctx.metainfo = META_tile
                return maxmax # tensor
            else:
                # if not the last tile, return fake 0 tensor
                fake_out = torch.zeros(partial_max_tile[0][0].size(), requires_grad=True).to(ctx.model_device)
                # print("partial_max_tile[0][0] size", partial_max_tile[0][0].size())
                # print("fake size", fake_out.size())
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
            return FINAL_res


    @staticmethod
    def backward(ctx, grad_output):
        # print("\n^^^^^cGGMaxPool2dFunction bwd")
        # print("grad_output", grad_output.size())
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
                #print("grad_in", grad_in.size())

                for i in range(0,b):
                    for j in range (0,c):
                        for pp in META_tile[i][j]:
                            if pp[0][0] == ctx.coord_h and pp[0][1] == ctx.coord_w:
                                # if the current tile contains the maxmax, restore back
                                #print("large tile at {}{} - {} {}".format(i, j, ctx.coord_h, ctx.coord_w))
                                indi = pp[1]
                                #print(type(indi.item()), type(w))
                                
                                max_position_w = indi.item()  % w
                                max_position_h = indi.item()  // w

                                #print("indi {} {} {} {}".format(indi,w, max_position_h, max_position_w))
                                grad_in[i][j][max_position_h][max_position_w] = grad_output[i,j]

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
       



 
