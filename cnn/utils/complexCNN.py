import torch.nn as nn

from .complexUtils import complex_conv2d
from .complexUtils import cLeakyReLu

class complexNet(nn.Module):
    """Torch module for creation of complex neural network with CNN arch.
    
       Attributes
        R : int 
        undersampling factor 
        
        layer_design: dict
        Network-Architecture. Here is a example with two hidden layers:
        
        layer_design_raki = {'num_hid_layer': 2, # number of hidden layers, in this case, its 2
                        'input_unit': nC,    # number channels in input layer, nC is coil number 
                            1:[256,(2,5)],   # the first hidden layer has 256 channels, and a kernel size of (2,5) in PE- and RO-direction
                            2:[128,(1,1)],   # the second hidden layer has 128 channels, and a kernel size of (1,1) in PE- and RO-direction
                        'output_unit':[(R-1)*nC,(1,5)] # the output layer has (R-1)*nC channels, and a kernel size of (1,5) in PE- and RO-direction
                        }
        
    
    """
    def __init__(self, layer_design, R):
        super(complexNet, self).__init__()
        # input layer
        self.conv1 = complex_conv2d(in_channels=layer_design['input_unit'],
                                    out_channels=layer_design[1][0],
                                    kernel_size=layer_design[1][1],
                                    stride=1,
                                    padding=0,
                                    dilation=[R, 1],
                                    groups=1,
                                    bias=False)
        self.cnt = 1
        # hidden layer
        for k in range(layer_design['num_hid_layer'] - 1):
            name = 'conv' + str(k + 2)
            setattr(
                self, name,
                complex_conv2d(in_channels=layer_design[k+1][0],
                               out_channels=layer_design[k+2][0],
                               kernel_size=layer_design[k+2][1],
                               stride=1,
                               padding=0,
                               dilation=[1, 1],
                               groups=1,
                               bias=False)
            )
            self.cnt += 1

        # output layer
        name = 'conv' + str(layer_design['num_hid_layer'] + 1)
        setattr(
            self, name,
            complex_conv2d(in_channels=layer_design[self.cnt][0],
                           out_channels=layer_design['output_unit'][0],
                           kernel_size=layer_design['output_unit'][1],
                           stride=1,
                           padding=0,                                       
                           dilation=[1, 1],
                           groups=1,
                           bias=False)
        )
        self.cnt += 1
    
    
    def forward(self, input_data):
        # input layer
        x = self.conv1(input_data)
        x = cLeakyReLu(x)
        # hidden layer
        for numLayer in range(self.cnt - 2):
            convLayer = getattr(self, 'conv' + str(numLayer + 2))
            x = convLayer(x)
            x = cLeakyReLu(x)
        # output layer
        convLayer = getattr(self, 'conv' + str(self.cnt))
        x = convLayer(x)
        conv = {'tot' : x}
        return conv
