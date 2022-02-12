import torch
from typing import Dict, List
from torch.autograd.variable import Variable
from uu.layers import maxpool2d, conv2d, tilesplit, relu, gavgpool2d, gmaxpool2d
import math

from uu.utils import correctness_check
from uu.utils.meta_info import Pad_info
import copy

def compute_info_beta(output_tile_coord: List, input_shape, output_shape, nTh, nTw, stream_structure, shape_dict, model_device) -> Dict:
    list_op__in_chckp_seg = []
    # just extract prue conv-max linklist
    for op in stream_structure._modules.values():
        if isinstance(op, tilesplit.TiledSplit) or isinstance(op, relu.cReLu):
            continue
        list_op__in_chckp_seg.append(op)
        #print("hash", id(op))

    # calc bwd_index info first
    b_info = compute_bwd_info_beta(output_tile_coord, input_shape, nTh, nTw, list_op__in_chckp_seg.copy(), shape_dict, model_device)
    
    bwd_out_shape = b_info[id(op)].cur_output_shape
    fwd_out_shape = (output_shape[2]//nTh, output_shape[3]//nTw)
    print("bwd_out_shape ", bwd_out_shape)
    print("fwd_out_shape ", fwd_out_shape)


    if not shape_compatible(fwd_out_shape, bwd_out_shape):
        #print("Yes, fwd is smaller")
        f_info = compute_fwd_info_beta(output_tile_coord, list_op__in_chckp_seg.copy(), shape_dict, b_info, nTh, nTw)
    else:
        f_info = compute_fwd_info_beta(output_tile_coord, list_op__in_chckp_seg.copy(), shape_dict, b_info, nTh, nTw)

    # print("op_list_in_seg", list_op__in_chckp_seg)
    print("------------------------------")
    print("f_info", f_info)
    print("------------------------------")
    print("b_info", b_info)

    assert len(f_info) != 0 and len(b_info) != 0
    info = [f_info, b_info]
    # return both fwd and bkw meta info
    return info


def compute_fwd_info_beta(output_tile_coord, list_op__in_chckp_seg, shape_dict, b_info, nTh, nTw) -> Dict:
    # stream_structure is list of ops
    # compute fwd is from the last stage
    list_op__in_chckp_seg.reverse()
    fwd_info_dict = {}
    with torch.no_grad():
        # get info from b_info
        op_idex = 0
        local_idex = 0  # if after maxpool, local_idex reset to 0
        peek_conv2d_pos = 0
        next_id = -99 # end of a conv2d chain
        for op in list_op__in_chckp_seg:
            uniq_opid = id(op)
            if op_idex == 0:    # the very first one(last), get info from b_info
                cur_output_shape = b_info[uniq_opid].cur_output_shape
                padding_info = ()
                input_slice = b_info[uniq_opid].input_slice
                real_index = input_slice
                # output of forward
                opname = "fake"
                pi = Pad_info(output_tile_coord, cur_output_shape, padding_info, input_slice, (), real_index, opname, -11, -11, -11, False, [], [nTh, nTw])
                fwd_info_dict[-11] = pi

            if isinstance(op, conv2d.TiledConv2d):
                if peek_conv2d_pos == 0:
                    peek_conv2d_pos = peek_position(list_op__in_chckp_seg, op_idex)
                    total_conv2d_in_seg = peek_conv2d_pos
                    #print("total_conv2d_in_seg", total_conv2d_in_seg)
                    
                ph = op.padding[0]
                pw = op.padding[1]
                # real_index is the key loop variable 
                none_tiled_output_shape = shape_dict[uniq_opid].output_shape
                oH = none_tiled_output_shape[2]
                oW = none_tiled_output_shape[3]
                non_disjoint_tile_size = [oH//nTh, oW//nTw]
                                
                cur_output_shape = (input_slice[1]-input_slice[0]+1, input_slice[3]-input_slice[2]+1) # r-l, b-t
                if op.stride[0] == 1:
                    padding_info, input_slice, internal_expand, real_index = conv2d_revr_padding_info(real_index, none_tiled_output_shape, [ph, pw], op.stride[0], op.kernel_size[0])
                else:
                    #re-adjust to get input slice when we do forward pass 
                    none_tiled_input_shape = shape_dict[uniq_opid].input_shape
                    padding_info, input_slice, internal_expand, real_index = conv2d_revr_padding_stride_fwd_info(real_index, none_tiled_output_shape, none_tiled_input_shape, [ph, pw], op.stride[0], op.kernel_size[0])


                opname = "conv2d"+str(uniq_opid)
                local_idex = total_conv2d_in_seg-peek_conv2d_pos
                peek_conv2d_pos -= 1
                if peek_conv2d_pos == 0:
                    local_first = True
                else:
                    local_first = False
                pi = Pad_info(output_tile_coord, cur_output_shape, padding_info, input_slice, internal_expand, real_index, opname, op_idex, local_idex, next_id, local_first, non_disjoint_tile_size, [nTh, nTw])
                fwd_info_dict[uniq_opid] = pi  # insert into info_dict
                next_id = uniq_opid
            elif isinstance(op, maxpool2d.cMaxPool2d):
                cur_output_shape = (input_slice[1]-input_slice[0]+1, input_slice[3]-input_slice[2]+1) # r-l, b-t
                opname = "maxpool2d"+str(uniq_opid)
                #produce input shape
                mxp_stride = op.stride[0]
                real_index = [x*mxp_stride for x in input_slice] # expand it and get parent index
                none_tiled_input_shape = shape_dict[uniq_opid].input_shape

                H = none_tiled_input_shape[2]
                W = none_tiled_input_shape[3]

                none_tiled_output_shape = shape_dict[uniq_opid].output_shape
                oH = none_tiled_output_shape[2]
                oW = none_tiled_output_shape[3]
                non_disjoint_tile_size = [oH//nTh, oW//nTw]
                # produce real_index for next op
                real_index[1] = min(W-1, real_index[1] +1) # +1 since 0-based 
                real_index[3] = min(H-1, real_index[3] +1)
                input_slice = real_index    # maxpooling no padding here.
                pi = Pad_info(output_tile_coord, cur_output_shape, (), input_slice, (), real_index, opname, op_idex, -1, next_id, False, non_disjoint_tile_size, [nTh, nTw])
                fwd_info_dict[uniq_opid] = pi # insert into info_dict
                next_id = uniq_opid
            elif isinstance(op, gavgpool2d.cGAvgPool2d) or isinstance(op, gmaxpool2d.cGMaxPool2d):
                opname = "gloablpool2d"+str(uniq_opid)
                cur_output_shape=[1,1]
                non_disjoint_tile_size = []
                pi = Pad_info(output_tile_coord, cur_output_shape, (), input_slice, (), real_index, opname, op_idex, -1, next_id, False, non_disjoint_tile_size, [nTh, nTw])
                fwd_info_dict[uniq_opid] = pi # insert into info_dict
                next_id = uniq_opid


            else:
                None
            op_idex += 1
        
        # fake dummy node to give output info
    return fwd_info_dict


def compute_bwd_info_beta(output_tile_coord: List, input_shape, nTh, nTw, list_op__in_chckp_seg, shape_dict, model_device) -> Dict:
    bwd_info_dict = {}
    # !! in the backward info, the cur_output_shape is in fact the input of this op

    with torch.no_grad():
        H = input_shape[2]
        W = input_shape[3]
        next_id = -99 # end of a conv2d chain
        op_idex = 0
        peek_conv2d_pos = 0
        print("AA", H, W)

        for op in list_op__in_chckp_seg:
            uniq_id = id(op)
            if op_idex == 0:    # the very first one, compute info manually; 
                Th = H // nTh
                Tw = W // nTw
                tile_top = output_tile_coord[0]*Th
                tile_bottom = tile_top+Th-1
                tile_left = output_tile_coord[1]*Tw
                tile_right = tile_left+Tw-1
                input_slice = [tile_left, tile_right, tile_top, tile_bottom]
                real_index = input_slice
                opname = "fake-grad-IN"
                pi = Pad_info(output_tile_coord, [Th, Tw], (), input_slice, (), real_index, opname, -11, -11, -101, False, [], [nTh, nTw], model_device)
                bwd_info_dict[-11] = pi
            
            if isinstance(op, conv2d.TiledConv2d):
                if peek_conv2d_pos == 0:
                    peek_conv2d_pos = peek_position(list_op__in_chckp_seg, op_idex)
                    total_conv2d_in_seg = peek_conv2d_pos
                
                local_idex = total_conv2d_in_seg-peek_conv2d_pos
                peek_conv2d_pos -= 1

                ph = op.padding[0]
                pw = op.padding[1]
                # real_index is the key loop variable 
                none_tiled_input_shape = shape_dict[id(op)].input_shape
                print("BB ph {}, none_tiled_input_shape {}".format(ph, none_tiled_input_shape))
                if op.stride[0] != 1:
                    padding_info, input_slice, internal_expand, real_index = conv2d_revr_padding_info_stride(real_index, none_tiled_input_shape, [ph, pw], op.stride[0], op.kernel_size[0])        
                else:
                    padding_info, input_slice, internal_expand, real_index = conv2d_revr_padding_info(real_index, none_tiled_input_shape, [ph, pw], op.stride[0], op.kernel_size[0])

                cur_output_shape = (input_slice[1]-input_slice[0]+1, input_slice[3]-input_slice[2]+1) # r-l, b-t
                opname = "bk-conv2d"+str(id(op))
                pi = Pad_info(output_tile_coord, cur_output_shape, padding_info, input_slice, internal_expand, real_index, opname, op_idex, local_idex, next_id, False, [], [nTh, nTw])
                bwd_info_dict[uniq_id] = pi  # insert into info_dict
                next_id = uniq_id

            elif isinstance(op, maxpool2d.cMaxPool2d):
                # get a logic global view
                opname = "bk-maxpool2d"+str(id(op))
                maxp_stride = op.stride[0]
                #print("pp real_index ", s_real_index)
                real_index = [math.floor(x / maxp_stride) for x in input_slice]
                H = math.floor(H / maxp_stride)
                W = math.floor(W / maxp_stride)
                cur_output_shape = (real_index[1]-real_index[0]+1, real_index[3]-real_index[2]+1) # r-l, b-t
                # produce real_index for next op
                real_index[1] = min(W-1, real_index[1])
                real_index[3] = min(H-1, real_index[3])
                input_slice = real_index    # maxpooling no padding here.
                pi = Pad_info(output_tile_coord, cur_output_shape, (), input_slice, (), real_index, opname, op_idex, -1, next_id, False, [], [nTh, nTw])
                bwd_info_dict[id(op)] = pi # insert into info_dict
                next_id = uniq_id
            elif isinstance(op, gavgpool2d.cGAvgPool2d) or isinstance(op, gmaxpool2d.cGMaxPool2d):
                opname = "bk-gloablpool2d"+str(id(op))
                cur_output_shape=[1,1]
                non_disjoint_tile_size = []
                pi = Pad_info(output_tile_coord, cur_output_shape, (), input_slice, (), real_index, opname, op_idex, -1, next_id, False, [], [nTh, nTw])
                bwd_info_dict[id(op)] = pi # insert into info_dict
                next_id = uniq_id

            else:
                None
            op_idex += 1
    return bwd_info_dict

def samestorage(x,y):
    if x.storage().data_ptr()==y.storage().data_ptr():
        print("same storage")
    else:
        print("different storage")
import time

# G_TIME = 0.0
def get_input_tile(info:Dict, input, first_op_in_seg):
    #global G_TIME
    input_tile = None
    with torch.no_grad():
        pi = info[first_op_in_seg]
        slice_info = pi.input_slice
        # TODO: optimize copy
        temp_view = input[:, :, slice_info[2]:slice_info[3]+1, slice_info[0]:slice_info[1]+1]      #NCHW
        #start = time.time()

        # #print(input_tile.storage_offset(),input_tile.size(), input_tile.stride())
        # new_storage = copy.deepcopy(temp_view.storage())
        # input_tile = torch.empty(temp_view.size())
        # input_tile.set_(new_storage, input_tile.storage_offset(), input_tile.size(), input_tile.stride()) # deep-copy (38 s) | conti

        input_tile = temp_view                                        # shallow-copy (25 s) | non-conti
        # input_tile = temp_view.contiguous()                           # deep-copy (25 s) | conti
        # input_tile = copy.deepcopy(temp_view)                         # overall time is slowest (40 s) | non-conti
       # input_tile = torch.empty_like(temp_view).copy_(temp_view)     # deep-copy (25 s) | conti
        #input_tile = temp_view.clone().detach()                       # deep-copy (25 s) | conti

        # input_tile = torch.empty(temp_view.size())
        #print("type of temp_view.size()", type(temp_view.size()))
        # print("input id", id(input))
        # print("input_tile id", id(input_tile))
        #input_tile = input_tile.contiguous()
        # samestorage(input, input_tile)
        # #print("is_contiguous()", input_tile.is_contiguous())
        # print("type of tensor data", type(temp_view.data_ptr()))

        # stop = time.time()
        # print("duration :", stop-start)
        # G_TIME += (stop-start)
        # print("G duration :", G_TIME)
        

    input_tile.requires_grad = input.requires_grad
    assert input_tile is not None
    return input_tile
    #return Variable(input_tile, requires_grad = True)


def resize_grad_in(info, grad_input):
    
    print("grad_input old", grad_input.size(), grad_input)
    if info.padding_info != [0] * len(info.padding_info):
        top = 0 + info.padding_info[2]
        bottom = grad_input.size()[2]-info.padding_info[3]
        left = 0 + info.padding_info[0]
        right = grad_input.size()[3]-info.padding_info[1]

        grad_input = grad_input[:, :, top:bottom, left:right]
        # TODO: if not in the first ...
        grad_input_shape = grad_input.size()
        #grad_input_new = torch.cuda.FloatTensor(torch.Size([grad_input_shape[0], grad_input_shape[1], bottom-top+info.padding_info[2]+info.padding_info[3], right-left+info.padding_info[0]+info.padding_info[1]]))
        
        #assert grad_input_new.size() == grad_input_shape
        #old padding
        #print(" resize_grad_in padding info ::")
        pd = torch.nn.ConstantPad2d(info.padding_info, 0)
        grad_input = pd(grad_input)
        print("grad_input new", grad_input.size(), grad_input)
        return grad_input
    else:
        return grad_input

def resize_grad_in_1(info, grad_input):
    #print("padding info ::", info.padding_info)
    if info.padding_info != [0] * len(info.padding_info):
        grad_input = grad_input[:, :, info.padding_info[2]:grad_input.size()[2]-info.padding_info[3], \
                    info.padding_info[0]:grad_input.size()[3]-info.padding_info[1]]
    return grad_input

def reshape_for_final(need_info, f_info, grad_input):
    #print("reshape_for_final", grad_input)
    #remove padding part
    grad_input = resize_grad_in_1(f_info, grad_input)
    # print("f_info ::", f_info, need_info)
    need_info_index = need_info.input_slice
    b_info_index = f_info.input_slice
    
    crop = []
    # assumption: b_info >> need_info
    for i in range(len(need_info_index)):
        crop.append(abs( b_info_index[i] - need_info_index[i]))
    # print("crop", crop)
    # print("reshape_for_final ##", crop[2], grad_input.size()[2]-crop[3], crop[0], grad_input.size()[3]-crop[1] )
    grad_input = grad_input[:,:,crop[2]:grad_input.size()[2]-crop[3], crop[0]:grad_input.size()[3]-crop[1]]
    # print("after crop g_in", grad_input.size())
    return grad_input


def debug_reshape_grad_out_input_tensor_for_weight_update(grad_output, input_tensor, f_info, next_f_info, weight_size, orig_padding, stride, nontiled_grad_out, nontiled_activation):
    # debug (, nontiled_grad_out, nontiled_activation)
    # to get disjoint part of grad_output
    # for stride 1 and same shape in/out-put
    # cal disjoint g_index
    # cal input index based on f_info and pure-calc

    
    Th = f_info.non_disjoint_tile_size[0]
    Tw = f_info.non_disjoint_tile_size[1]
    tile_top = f_info.coord[0]*Th
    tile_bottom = tile_top+Th-1
    tile_left = f_info.coord[1]*Tw
    tile_right = tile_left+Tw-1
    actual_index = [tile_left, tile_right, tile_top, tile_bottom]
    
    
    if next_f_info.opname != "fake":
        current_stage_g_index = next_f_info.input_slice
        crop = []
        # assumption: current_stage_g_index >> actual_index
        for i in range(len(actual_index)):
            crop.append(abs( current_stage_g_index[i] - actual_index[i]))
        
        current_padd = next_f_info.padding_info
        if len(current_padd) > 0 and current_padd != [0] * len(current_padd):
            grad_output = grad_output[:, :, current_padd[2]:grad_output.size()[2]-current_padd[3], \
                        current_padd[0]:grad_output.size()[3]-current_padd[1]]
            input_tensor = input_tensor[:, :, current_padd[2]:input_tensor.size()[2]-current_padd[3], \
                        current_padd[0]:input_tensor.size()[3]-current_padd[1]]

        print("fake grad_out size", grad_output.size())
        print("crop", crop, current_stage_g_index, actual_index)
        print("##", crop[2],grad_output.size()[2]-crop[3], crop[0],grad_output.size()[3]-crop[1])
        grad_output = grad_output[:,:,crop[2]:grad_output.size()[2]-crop[3], crop[0]:grad_output.size()[3]-crop[1]]
        input_tensor = input_tensor[:,:,crop[2]:input_tensor.size()[2]-crop[3], crop[0]:input_tensor.size()[3]-crop[1]]
    else:
        print("input_tensor size", input_tensor.size())
        grad_output = grad_output[:,:, tile_top: tile_bottom+1, tile_left: tile_right+1]
        
        input_g_index = f_info.input_slice
        out_g_index = next_f_info.input_slice
        actual_index = [tile_left, tile_right, tile_top, tile_bottom]

        crop = []
        for i in range(len(actual_index)):
            crop.append(abs( out_g_index[i] - actual_index[i]))
        
        # current_padd = f_info.padding_info
        # print("current_padd", current_padd)
        # if len(current_padd) > 0 and current_padd != [0] * len(current_padd):
        #     input_tensor = input_tensor[:, :, current_padd[2]:input_tensor.size()[2]-current_padd[3], \
        #                 current_padd[0]:input_tensor.size()[3]-current_padd[1]]
        print("grad_out size", grad_output.size())
        print("input_tensor size", input_tensor.size())
        print("crop", crop, out_g_index, actual_index)
        print("##", crop[2],input_tensor.size()[2]-crop[3], crop[0],input_tensor.size()[3]-crop[1])
        input_tensor = input_tensor[:,:,crop[2]:input_tensor.size()[2]-crop[3], crop[0]:input_tensor.size()[3]-crop[1]]




    # for debug
    nontiled_grad_out = nontiled_grad_out[:,:, tile_top: tile_bottom+1, tile_left: tile_right+1]
    iH = nontiled_activation.size()[2]
    iW = nontiled_activation.size()[3]
    input_top = tile_top
    input_bottom = min(iH-1, (tile_top+Th*stride[0]-1+weight_size[2]-1))
    input_left = tile_left
    input_right = min(iW-1, (tile_left+Tw*stride[1]-1+weight_size[3]-1))
    nontiled_activation = nontiled_activation[:,:, input_top: input_bottom+1, input_left: input_right+1]    
    print("A", tile_top, tile_bottom+1, tile_left, tile_right+1)
    print("B", input_top, input_bottom+1, input_left, input_right+1)

    
    print("csaved input", input_tensor.size())
    print("cgrad_output", grad_output.size())
    print("real nontiled_activation", nontiled_activation.size())
    print("real nontiled_grad_out", nontiled_grad_out.size())
    # nontiled_activation[0][0][0][0] = -99

    # correctness_check.check_equal(grad_output, nontiled_grad_out, False)
    # correctness_check.check_equal(input_tensor, nontiled_activation, False)
    
    return grad_output, input_tensor


def reshape_grad_out_input_tensor_for_weight_update(grad_output, input_tensor, f_info, next_f_info, weight_size, orig_padding, stride):
    # to get disjoint part of grad_output
    # for stride 1 and same shape in/out-put
    # cal disjoint g_index
    # cal input index based on f_info and pure-calc    
    Th = f_info.non_disjoint_tile_size[0]
    Tw = f_info.non_disjoint_tile_size[1]
    tile_top = f_info.coord[0]*Th
    tile_bottom = tile_top+Th-1
    tile_left = f_info.coord[1]*Tw
    tile_right = tile_left+Tw-1
    actual_index = [tile_left, tile_right, tile_top, tile_bottom]
    
    
    if next_f_info.opname != "fake":
        current_stage_g_index = next_f_info.input_slice
        crop = []
        # assumption: current_stage_g_index >> actual_index
        for i in range(len(actual_index)):
            crop.append(abs( current_stage_g_index[i] - actual_index[i]))
        
        current_padd = next_f_info.padding_info
        if len(current_padd) > 0 and current_padd != [0] * len(current_padd):
            grad_output = grad_output[:, :, current_padd[2]:grad_output.size()[2]-current_padd[3], \
                        current_padd[0]:grad_output.size()[3]-current_padd[1]]
            input_tensor = input_tensor[:, :, current_padd[2]:input_tensor.size()[2]-current_padd[3], \
                        current_padd[0]:input_tensor.size()[3]-current_padd[1]]

        # print("fake grad_out size", grad_output.size())
        # print("crop", crop, current_stage_g_index, actual_index)
        # print("##", crop[2],grad_output.size()[2]-crop[3], crop[0],grad_output.size()[3]-crop[1])
        grad_output = grad_output[:,:,crop[2]:grad_output.size()[2]-crop[3], crop[0]:grad_output.size()[3]-crop[1]]
        input_tensor = input_tensor[:,:,crop[2]:input_tensor.size()[2]-crop[3], crop[0]:input_tensor.size()[3]-crop[1]]
    else:
        #print("input_tensor size", input_tensor.size())
        grad_output = grad_output[:,:, tile_top: tile_bottom+1, tile_left: tile_right+1]
        
        input_g_index = f_info.input_slice
        out_g_index = next_f_info.input_slice
        actual_index = [tile_left, tile_right, tile_top, tile_bottom]

        crop = []
        for i in range(len(actual_index)):
            crop.append(abs( out_g_index[i] - actual_index[i]))
        
        # print("grad_out size", grad_output.size())
        # print("input_tensor size", input_tensor.size())
        # print("crop", crop, out_g_index, actual_index)
        # print("##", crop[2],input_tensor.size()[2]-crop[3], crop[0],input_tensor.size()[3]-crop[1])
        input_tensor = input_tensor[:,:,crop[2]:input_tensor.size()[2]-crop[3], crop[0]:input_tensor.size()[3]-crop[1]]

    return grad_output, input_tensor



#TODO can simplify
def recreate_input_tile_f(info:Dict, input, next_id):
    with torch.no_grad():
        pi = info[0][next_id]
        padding_info = pi.padding_info
        if padding_info != [0] * len(padding_info):
            #shifting tile to extract
            input_shape = input.size()
            top = 0 + padding_info[2]
            bottom = input_shape[2]-padding_info[3]
            left = 0 + padding_info[0]
            right = input_shape[3]-padding_info[1]

            # top_bound = 0
            # bottom_bound = input_shape[2]
            # left_bound = 0
            # right_bound = input_shape[3]
            # print("\n===\n")
            # print(input_shape)
            # print(padding_info)
            # print(slice_info)
            
            # print("\n===\n")
            
            input_tile = input[:, :, top:bottom, left:right]       #NCHW, included index
            # print("top, bottom, left, right " , top, bottom, left, right)
            # print("== inputtile for next", input_tile.size())
            #print(padding_info)
            # print(" recreate_input_tile_f p ", pi.coord ,padding_info)
            pd = torch.nn.ConstantPad2d(padding_info, 0)
            input_tile = pd(input_tile)
            # return input_tile
            #input_tile = torch.cuda.FloatTensor(torch.Size([input_shape[0], input_shape[1], bottom-top+padding_info[2]+padding_info[3], right-left+padding_info[0]+padding_info[1]]))
            #input_tile_new[:, :, top:bottom, left:right]  = input[:, :, top:bottom, left:right] 
            
            # print(input_shape[0], input_shape[1], bottom-top+padding_info[2]+padding_info[3], right-left+padding_info[0]+padding_info[1])
            # print(input_tile_new.size(), input_tile.size())
            #assert input_tile_new.size() == input.size()

            return input_tile
        else:
            return input

def output_element_covered(begin, end, kernel_span, stride, original_shape):
   # 0 based index
   far_left_begin = max(0, begin-kernel_span)
   #far_right_begin = min(original_shape-1, begin+kernel_span)
   far_left_end = max(0, end-kernel_span)
   #far_right_end = min(original_shape-1, end+kernel_span)
   possible_end = original_shape-kernel_span
   end = min(end, possible_end) # to bounded by last index which can hold a
   # print("original_shape", original_shape)
   # print(type(far_left_begin), far_left_begin, list(range(far_left_begin, begin+1))  )
   # print(type(far_left_end), far_left_end, list(range(far_left_end, end+1))  )
   smallest_output_elem = min(list(filter(lambda x: (x%stride == 0) , list(range(far_left_begin, begin+1)))))
   largest_output_elem = max(list(filter(lambda x: (x%stride == 0) , list(range(far_left_end, end+1)))))
 
   smallest_output_elem = smallest_output_elem//stride
   largest_output_elem = largest_output_elem//stride
 
   return [smallest_output_elem, largest_output_elem]
 

def conv2d_revr_padding_stride_fwd_info(tile_indx: List, none_tiled_output_shape, none_tiled_input_shape, pads: List, stride, RS):
    oH = none_tiled_output_shape[2]
    oW = none_tiled_output_shape[3]
    

    print("fwsinfo Oh, Ow", oH, oW)


    # input as forward input and output as forward output
    iH = none_tiled_input_shape[2]
    iW = none_tiled_input_shape[3]

    tile_top = tile_indx[2]
    tile_bottom = tile_indx[3]
    tile_left = tile_indx[0]
    tile_right = tile_indx[1]
    print("fwdinfo tile_top", tile_top, tile_bottom, tile_left, tile_right)


    # do we need to cover +- kernal span??

    #input view in the padded global index field
    input_expand_top = tile_top * stride
    input_expand_left = tile_left * stride
    input_expand_bottom = tile_bottom * stride  # included
    input_expand_right = tile_right * stride  # included


    # cover - out_tile index
    # TODO:I do not think it need pad and internal expand
    pad_top = max(0-(input_expand_top-pads[0]), 0)
    pad_bottom = max((input_expand_bottom+pads[0])-iH, 0)
    pad_left = max(0-(input_expand_left-pads[1]), 0)
    pad_right = max((input_expand_right+pads[1])-iW, 0)
    # padding 0 element
    padding_info = [pad_left, pad_right, pad_top, pad_bottom]
    #print(padding_info)

    # TODO: is it general??
    iexp_top = pads[0] if (input_expand_top-pads[0])>=0 else 0
    iexp_bottom = pads[0] if (input_expand_bottom+pads[0])<=(iH-1) else 0
    iexp_left = pads[1] if (input_expand_left-pads[1]) >= 0 else 0
    iexp_right = pads[1] if (input_expand_right+pads[1])<=(iW-1) else 0
    internal_expand = [iexp_left, iexp_right, iexp_top, iexp_bottom]

    
    # if no adjust, disjoint view of output (Aka, real input)
    input_top = max(0, (input_expand_top-pads[0]))
    input_bottom = min(iH-1, (input_expand_bottom+pads[0]))
    input_left = max(0, (input_expand_left-pads[1]))
    input_right = min(iW-1, (input_expand_right+pads[1]))
    #input_tile view, the 4 point(l,r,t,b) in input tenser. Value is include [l, r], [t, b]
    input_slice = [input_left, input_right, input_top, input_bottom]

    # left , right, top, bottom; the naming is misleading; it means the relative index of current input view in its parent's view.
    # real index can have negative value and larger than iH,iW value, since it shows info one level up. 
    real_index = [(tile_left-pads[1]), (tile_right+pads[1]), (tile_top-pads[0]), (tile_bottom+pads[0])]
    # print("--tile_indx", tile_indx)
    # print("--input_slice", input_slice)
    # print("--real_index", real_index)
    return padding_info, input_slice, internal_expand, real_index


def conv2d_revr_padding_info_stride(tile_indx: List, none_tiled_output_shape, pads: List, stride, RS):
    oH = none_tiled_output_shape[2]
    oW = none_tiled_output_shape[3]
    # in fact it is the shape of non-tield output if given the forward input shape(none_tiled_output_shape)
    iH = (oH-RS+2*pads[0])//stride+1
    iW = (oW-RS+2*pads[0])//stride+1

    tile_top = tile_indx[2]
    tile_bottom = tile_indx[3]
    tile_left = tile_indx[0]
    tile_right = tile_indx[1]

    one_side_pad = pads[0]
    # boundary shift in padding input veiw   in -> (conv) -> out
    shf_tile_top = tile_top + one_side_pad
    shf_tile_bottom = tile_bottom + one_side_pad
    shf_tile_left = tile_left + one_side_pad
    shf_tile_right = tile_right + one_side_pad

    print("tile_top", tile_top, tile_bottom, tile_left, tile_right)
    print("shf_tile_top", shf_tile_top, shf_tile_bottom, shf_tile_left, shf_tile_right)
    print("Oh, Ow", oH, oW)

    # Assumption, our tile size >> kernel size
    # TODO: why do I need padding here?? maybe is the closest strided number
    adj_tile_top = (shf_tile_top // stride)*stride
    adj_tile_left = (shf_tile_left // stride)*stride
    padded_H = oH+2*pads[0]
    padded_W = oW+2*pads[0]
    H_cover = output_element_covered(adj_tile_top, shf_tile_bottom, RS-1, stride, padded_H)
    W_cover = output_element_covered(adj_tile_left, shf_tile_right, RS-1, stride, padded_W)

    print("H_cover", adj_tile_top, adj_tile_left, H_cover, W_cover)
    
    out_tile_top = tile_top // stride
    out_tile_bottom = tile_bottom // stride
    out_tile_left = tile_left // stride
    out_tile_right = tile_right // stride


    print("out_tile_top", out_tile_top, out_tile_bottom)
    # cover - out_tile index
    pad_top = max(0-(H_cover[0]-pads[0]), 0)
    pad_bottom = max((H_cover[1]+pads[0])-out_tile_bottom, 0)
    pad_left = max(0-(W_cover[0]-pads[1]), 0)
    pad_right = max((W_cover[1]+pads[1])-out_tile_right, 0)
    # padding 0 element
    padding_info = [pad_left, pad_right, pad_top, pad_bottom]
    #print(padding_info)

    # TODO: is it general??
    iexp_top = pads[0] if (H_cover[0]-pads[0])>=0 else 0
    iexp_bottom = pads[0] if (H_cover[1]+pads[0])<=(out_tile_bottom) else 0
    iexp_left = pads[1] if (W_cover[0]-pads[1]) >= 0 else 0
    iexp_right = pads[1] if (W_cover[1]+pads[1])<=(out_tile_right) else 0
    internal_expand = [iexp_left, iexp_right, iexp_top, iexp_bottom]

    # input_top = max(0, (tile_top-pads[0]))
    # input_bottom = min(iH-1, (tile_bottom+pads[0]))
    # input_left = max(0, (tile_left-pads[1]))
    # input_right = min(iW-1, (tile_right+pads[1]))

    # if no adjust, disjoint view of output (Aka, real input)
    input_top = H_cover[0]
    input_bottom = H_cover[1]
    input_left = W_cover[0]
    input_right = W_cover[1]
    #input_tile view, the 4 point(l,r,t,b) in input tenser. Value is include [l, r], [t, b]
    # input for next op
    input_slice = [input_left, input_right, input_top, input_bottom]

    # left , right, top, bottom; the naming is misleading; it means the relative index of current input view in its parent's view.
    # real index can have negative value and larger than iH,iW value, since it shows info one level up. 
    real_index = [(tile_left-pads[1]), (tile_right+pads[1]), (tile_top-pads[0]), (tile_bottom+pads[0])]

    print("--input_slice", input_slice)
    print("--real_index", real_index)
    print("--padding_info", padding_info)
    print("--internal_expand", internal_expand)
    return padding_info, input_slice, internal_expand, real_index
    
   



# Assume conv2d input output are same shape
def conv2d_revr_padding_info(tile_indx: List, none_tiled_output_shape, pads: List, stride, RS):
    #print("--", tile_indx, none_tiled_output_shape, pads, stride, RS)
    #pdb.set_trace()
    oH = none_tiled_output_shape[2]
    oW = none_tiled_output_shape[3]
    iH = (oH-1)*stride+RS-2*pads[0]
    iW = (oW-1)*stride+RS-2*pads[1]
    # output view

    print("OH IH", oH, iH)

    tile_top = tile_indx[2]
    tile_bottom = tile_indx[3]
    tile_left = tile_indx[0]
    tile_right = tile_indx[1]
    #print("index", tile_left, tile_right, tile_top, tile_bottom)

    #here we only consider stride = 1
    pad_top = max(0-(tile_top-pads[0]), 0)
    pad_bottom = max((tile_bottom+pads[0])-(iH-1), 0)
    pad_left = max(0-(tile_left-pads[1]), 0)
    pad_right = max((tile_right+pads[1])-(iW-1), 0)
    # padding 0 element
    padding_info = [pad_left, pad_right, pad_top, pad_bottom]
    #print(padding_info)

    # TODO: is it general??
    iexp_top = pads[0] if (tile_top-pads[0])>=0 else 0
    iexp_bottom = pads[0] if (tile_bottom+pads[0])<=(iH-1) else 0
    iexp_left = pads[1] if (tile_left-pads[1]) >= 0 else 0
    iexp_right = pads[1] if (tile_right+pads[1])<=(iW-1) else 0
    internal_expand = [iexp_left, iexp_right, iexp_top, iexp_bottom]

    input_top = max(0, (tile_top-pads[0]))
    input_bottom = min(iH-1, (tile_bottom+pads[0]))
    input_left = max(0, (tile_left-pads[1]))
    input_right = min(iW-1, (tile_right+pads[1]))
    #input_tile view, the 4 point(l,r,t,b) in input tenser. Value is include [l, r], [t, b]
    input_slice = [input_left, input_right, input_top, input_bottom]

    # left , right, top, bottom; the naming is misleading; it means the relative index of current input view in its parent's view.
    # real index can have negative value and larger than iH,iW value, since it shows info one level up. 
    real_index = [(tile_left-pads[1]), (tile_right+pads[1]), (tile_top-pads[0]), (tile_bottom+pads[0])]
    # print("--tile_indx", tile_indx)
    # print("--input_slice", input_slice)
    # print("--real_index", real_index)
    return padding_info, input_slice, internal_expand, real_index





def shape_compatible(fwd_out_shape, bwd_out_shape):
    if fwd_out_shape[0] >= bwd_out_shape[0] and fwd_out_shape[1] >= bwd_out_shape[1]:
        return True
    else:
        return False


def peek_position(stream_structure, op_idex):
    num_conv_in_seg = 0
    # uniq_id_array = []
    while op_idex < len(stream_structure):
        op = stream_structure[op_idex]
        if isinstance(op, conv2d.TiledConv2d):
            num_conv_in_seg += 1
            # uniq_id_array.append[id(op)]
        else:
            break
        op_idex += 1
    return num_conv_in_seg