import torch
from uu.utils import padding_calc

       
class dMove(torch.nn.Module):
    def __init__(self):
        super(dMove, self).__init__()

    def forward(self, *inputs):
        # add isTileRead as one more arg
        if len(inputs) == 4:
            is_ccheckpoint = False
            model_device = inputs[-2]
        elif len(inputs) == 5:
            is_ccheckpoint = inputs[-1]
            model_device = inputs[-3] # if manully append checkpoint flag in cCheckpoint, model_device is on -3, otherwise is in -2
        else:
            print("missing info in move node")
            assert False


        tile = inputs[0] 
        # send tile to proper device
        is_m_cuda = True if "cuda" in str(model_device) else False
        # print("shape of input", tile.size())
        # print("model_device",  is_m_cuda,  str(model_device) )
        input_is_cuda = tile.is_cuda

        if input_is_cuda != is_m_cuda:
            # print(model_device)
            if is_m_cuda == True: # model is on GPU 
                tile = tile.to(model_device)    # explicitly load input tile to device 
            else:
                device = torch.device("cpu")
                tile = tile.to(device) 
        # DONE sending to device 

        info = inputs[1]
        return tile, info, is_ccheckpoint
 
