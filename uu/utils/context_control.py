class conv_2d_ctx:
    def __init__(self):
        self.input = None
        self.weight = None
        self.padding = None
        self.stride = None
        self.groups = None
        self.uniq_id = None
        self.info = None
        self.coord = None
        self.input_real_size = None
    def __repr__(self) -> str:
        rep = "[[ " + str(self.uniq_id) +"[" +' PI( <' + "".join([str(x)+"," for x in self.coord]) + '>,\n'
        return rep



class maxpool_2d_ctx:
    def __init__(self):
        self.input = None
        self.kernel_size = None
        self.padding = None
        self.stride = None
        self.uniq_id = None
        self.info = None
        self.arg_max = None
