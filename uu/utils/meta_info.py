# might need to create a memo structure. 

# current linear chain
# backprop meta info
# fake-bp (-11)  <--------- op1 <--------op2
# [disjoint view of
# the grad of input image 
# ]
#


class Pad_info:
    def __init__(self, coord, cur_output_shape, padding_info, input_slice, internal_expand, real_index, opname, \
        op_idex, local_idex, next_id, local_first, non_disjoint_tile_size, numof_tiles, model_device=None):
        # tile coodination info on H/W surface; 2d list; based on [0,0]
        self.coord = coord 
        # output shape after current op(either f or b); 2d list
        self.cur_output_shape = cur_output_shape 
        # padding 0 for current tile; 4d list; (l,r,t,b) 
        self.padding_info = padding_info
        # input_tile view, the 4 point(l,r,t,b) in input tenser. Value is included [l, r], [t, b]
        self.input_slice = input_slice
        # internal extend for current tile; 4d list; (l,r,t,b) 
        self.internal_expand = internal_expand
        # the relative index of current input view in its parent's view
        # see function conv2d_revr_padding_info
        self.real_index = real_index
        # name of the op '[bk-] name hashvalue'
        self.opname = opname
        # global order/position of the op; 0 is the last in the fwd chain
        self.op_idex = op_idex
        # local order/position of the op; 
        # maxpool is an end mark of a local segment; its local_idex is -1
        self.local_idex = local_idex
        #point to next op in computation graph/autograd graph
        self.next_id = next_id
        #local_first is in fact local_last in fwd local segment; but the first in reversed autograd
        self.local_first = local_first

        self.non_disjoint_tile_size = non_disjoint_tile_size
        #[Nth, Ntw]
        self.numof_tiles = numof_tiles 

        self.model_device = model_device
        
    # def copy(self): 
    #     return type(self)(self.coord, self.cur_output_shape, self.padding_info, \
    #         self.input_slice, self.internal_expand, self.real_index, self.opname, self.op_idex, self.local_idex, self.next_id)

    def __repr__(self) -> str:
        rep = self.opname +"[" +str(self.op_idex)+","+str(self.local_idex) + "]" +'\n PI( <' + "".join([str(x)+"," for x in self.coord]) + '>,\n <otileshape ' + \
                    "".join([str(x)+"," for x in self.cur_output_shape]) + '>,\n <padding ' + \
                    "".join([str(x)+"," for x in self.padding_info]) + '>,\n <inpslidx ' + \
                    "".join([str(x)+"," for x in self.input_slice]) + '>, \n <internal ' + \
                    "".join([str(x)+"," for x in self.internal_expand]) + '>, \n <realidx ' + \
                    "".join([str(x)+"," for x in self.real_index]) + '>, \n <ndtsize ' + \
                    "".join([str(x)+"," for x in self.non_disjoint_tile_size]) + '>, \n ' + \
                        " local_first " + str(self.local_first) +'\n' + \
                        " next_id " + str(self.next_id) + "\n numof tiles " + "".join([str(x)+"," for x in self.numof_tiles]) + " ) \n"
        return rep

# 