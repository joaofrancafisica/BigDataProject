#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 14:41:25 2018

@author: manuel
"""

from keras.utils.vis_utils import model_to_dot
from keras.layers import Wrapper

import numpy as np
import re, inspect, time, sys
from PIL import ImageFont


def _hex2RGB(h):
    if h[0] == "#":
        h = h[1:]
    R = int("0x"+h[0:2],16)
    G = int("0x"+h[2:4],16)
    B = int("0x"+h[4:6],16)
    A = 255
    if len(h) == 8:
        A = int("0x"+h[6:8],16)
    return R,G,B,A

def _get_model_Svg(model,
                   filename=None,
                   display_shapes=True,
                   display_params=False,
                   display_wrappers=False,
                   display_lambdas=False,
                   verbose=False):
    
    
    if (verbose):
        t0 = time.time()
        sys.stdout.write("Creating model_to_dot...")
    # Get model dot (optimal tags locations)
    ddot = model_to_dot(model).create_plain().splitlines() # split linebreaks
    if (verbose):
        t1 = time.time()
        sys.stdout.write("%f (s)\n"%(t1-t0))
    layersInfo = dict()

    zoom = 100

    # Before anything else, let's parse the information contained inside ddot
    if (verbose):
        t2 = time.time()
        sys.stdout.write("Parsing dot data...")
    i = 1
    ddot_tmp = ddot[i]
    while (ddot_tmp != "stop"):
        ddot_type = ddot_tmp[0:4]
        
        # Regular expressions were built using the online tool: https://regex101.com/r/9y9n85/1
        if ddot_type == "node":
            #print(ddot_tmp)
            #pattern = re.compile("node (\d+) ([0-9]+(?:\.[0-9]+)?) ([0-9]+(?:\.[0-9]+)?) ([0-9]+(?:\.[0-9]+)?) ([0-9]+(?:\.[0-9]+)?) (?:\")(\w+)(?:\:) (\w+)(?:\") (\w+) (\w+) (\w+) (\w+)")
            pattern = re.compile("node (\d+) ([0-9]+(?:\.[0-9]+)?) ([0-9]+(?:\.[0-9]+)?) ([0-9]+(?:\.[0-9]+)?) ([0-9]+(?:\.[0-9]+)?)")
            matches = pattern.findall(ddot_tmp)[0]
            dotId = str(matches[0])
            dotXc = zoom*float(matches[1])
            dotYc = zoom*float(matches[2])
            dotW = zoom*float(matches[3])
            dotH = zoom*float(matches[4])
            
            if dotId in layersInfo:
                layersInfo[dotId]['dotPosition'] = [dotXc,dotYc,dotW,dotH]
            else:
                layersInfo[dotId] = {'dotPosition':[dotXc,dotYc,dotW,dotH]}
            
        i += 1
        ddot_tmp = ddot[i]  
        
    if (verbose):
        t3 = time.time()
        sys.stdout.write("%f (s)\n"%(t3-t2))


    # get model layers
    layers = model.layers
    SvgTag = ""
    bbox = [np.Inf, np.inf, -1, -1]
    
    if (verbose):
        sys.stdout.write("Extracting information from layers\n")

    for layer in layers:
        if (verbose):
            t4 = time.time()
            sys.stdout.write("\tLayer %s"%(layer.name))
        layer_id = str(id(layer))
    
        # initialize dictionary with layer information
        if not layer_id in layersInfo:
            layersInfo[layer_id] = dict()
    
        # Append a wrapped layer's label to node's label, if it exists.
        layer_name = layer.name
        layer_type = layer.__class__.__name__
        layer_typeName = layer_type
        iswrapper = False
        if isinstance(layer, Wrapper):
            layer_name = '{}({})'.format(layer_name, layer.layer.name)
            child_layer_type = layer.layer.__class__.__name__
            if display_wrappers:
                layer_typeName = '{}({})'.format(layer_type, child_layer_type)
            else:
                layer_typeName = child_layer_type
            layer_type = child_layer_type
            iswrapper = True
        
        layersInfo[layer_id]['name'] = layer_name
        layersInfo[layer_id]['type'] = layer_type
        
        oshape = [] # if empty means that oshape did not changed
        
        # Now let's switch the class
        if (layer_type == "InputLayer"):
            tag_color = "#acacace2"
            border_color = "#4d4d4da9"
            font_color = "#ffffffff"
            txt = "Input"
            if not iswrapper:
                params = {'shape':layer.input.shape.as_list()}
            else:
                params = {'shape':layer.layer.input.shape.as_list()}
            oshape = layer.output_shape
        elif (layer_type == "Conv1D") or (layer_type == "Conv2D") or (layer_type == "Conv3D") or (layer_type == "Conv2DTranspose"):
            tag_color = "#2a7fffff"
            border_color = "#5151c0ff"
            font_color = "#ffffffff"
            if not iswrapper:
                params = {'activation':layer.activation.func_name,
                      'kernel':layer.kernel.shape.as_list(),
                      'padding':layer.padding,
                      'strides':layer.strides}
            else:
                params = {'activation':layer.layer.activation.func_name,
                      'kernel':layer.layer.kernel.shape.as_list(),
                      'padding':layer.layer.padding,
                      'strides':layer.layer.strides}
            if not display_params:
                txt = layer_typeName
            else:
                kernel = tuple(params['kernel'])
                txt = [layer_typeName,
                       str(kernel[:-1]) + " x " + str(kernel[-1])]
            oshape = layer.output_shape
        elif (layer_type == "SeparableConv1D") or (layer_type == "SeparableConv2D"):
            tag_color = "#2a7fffff"
            border_color = "#5151c0ff"
            font_color = "#ffffffff"
            if not iswrapper:
                params = {'activation':layer.activation.func_name,
                          'kernel':layer.kernel_size,
                          'padding':layer.padding,
                          'strides':layer.strides}
            else:
                params = {'activation':layer.layer.activation.func_name,
                          'kernel':layer.layer.kernel_size,
                          'padding':layer.layer.padding,
                          'strides':layer.layer.strides}
            if not display_params:
                txt = layer_typeName
            else:
                kernel = tuple(params['kernel'])
                #if layer.activation.func_name != 'linear':
                if params['sstrides'] != (1,1):
                    txt = [layer_typeName,
                           str(kernel[:-1]) + " x " + str(kernel[-1]),
                           "s"+str(params['strides'])]
                else:
                    txt = [layer_typeName,
                           str(kernel[:-1]) + " x " + str(kernel[-1]),
                           "s"+str(params['strides'])]
            oshape = layer.output_shape
        elif (layer_type == "LocallyConnected1D") or (layer_type == "LocallyConnected2D"):
            tag_color = "#ffaaeeff"
            border_color = "#d400aaff"
            font_color = "#000000ff"
            if not iswrapper:
                params = {'activation':layer.activation.func_name,
                          'kernel':layer.kernel_size,
                          'padding':layer.padding,
                          'strides':layer.strides}
            else:
                params = {'activation':layer.layer.activation.func_name,
                          'kernel':layer.layer.kernel_size,
                          'padding':layer.layer.padding,
                          'strides':layer.layer.strides}
            if not display_params:
                txt = layer_typeName
            else:
                kernel = tuple(params['kernel'])
                #if layer.activation.func_name != 'linear':
                if params['strides'] != (1,1):
                    txt = [layer_typeName,
                           str(kernel[:-1]) + " x " + str(kernel[-1]),
                           "s"+str(params['strides'])]
                else:
                    txt = [layer_typeName,
                           str(kernel[:-1]) + " x " + str(kernel[-1]),
                           "s"+str(params['strides'])]
            oshape = layer.output_shape
        elif (layer_type == "Cropping1D") or (layer_type == "Cropping2D") or (layer_type == "Cropping3D"):
            tag_color = "#2a7fffff"
            border_color = "#5151c0ff"
            font_color = "#ffffffff"
            if not iswrapper:
                params = {'cropping':layer.cropping}
            else:
                params = {'cropping':layer.layer.cropping}
            if not display_params:
                txt = layer_typeName
            else:
                cropping = tuple(params['cropping'])
                #if layer.activation.func_name != 'linear':
                if cropping != (1,1):
                    txt = [layer_typeName,
                          str(cropping)]
                else:
                    txt = layer_typeName
            oshape = layer.output_shape
        elif (layer_type == "UpSampling1D") or (layer_type == "UpSampling2D") or (layer_type == "UpSampling3D"):
            tag_color = "#2a7fffff"
            border_color = "#5151c0ff"
            font_color = "#ffffffff"
            if not iswrapper:
                params = {'size':layer.size}
            else:
                params = {'size':layer.layer.size}
            if not display_params:
                txt = layer_type
            else:
                txt = [layer_type,
                      str(params['size'])]
            oshape = layer.output_shape
        elif (layer_type == "ZeroPadding1D") or (layer_type == "ZeroPadding2D") or (layer_type == "ZeroPadding3D"):
            tag_color = "#2a7fffff"
            border_color = "#5151c0ff"
            font_color = "#ffffffff"
            if not iswrapper:
                params = {'padding':layer.padding}
            else:
                params = {'padding':layer.layer.padding}
            if not display_params:
                txt = layer_type
            else:
                txt = [layer_type,
                      str(params['padding'])]
        elif (layer_type == "MaxPooling1D") or (layer_type == "MaxPooling2D") or (layer_type == "MaxPooling3D"):
            tag_color = "#d3605bff"    
            border_color = "#9c3030ff"
            font_color = "#ffffffff"
            if not iswrapper:
                params = {'pool_size':layer.pool_size,
                          'strides':layer.strides}
            else:
                params = {'pool_size':layer.layer.pool_size,
                          'strides':layer.layer.strides}
            if params['pool_size'] != (1,1) and display_params:
                txt = ["MaxPool" + layer_type[-2:],
                       str(params['pool_size'])]
            else:
                txt = "MaxPool" + layer_type[-2:]
            oshape = layer.output_shape
        elif (layer_type == "AveragePooling1D") or (layer_type == "AveragePooling2D") or (layer_type == "AveragePooling3D"):
            tag_color = "#d3605bff"    
            border_color = "#9c3030ff"
            font_color = "#ffffffff"
            if not iswrapper:
                params = {'pool_size':layer.pool_size,
                          'strides':layer.strides}
            else:
                params = {'pool_size':layer.layer.pool_size,
                          'strides':layer.layer.strides}
            if params['pool_size'] != (1,1) and display_params:
                txt = ["AvgPool" + layer_type[-2:],
                       str(params['pool_size'])]
            else:
                txt = "AvgPool" + layer_type[-2:]
            oshape = layer.output_shape
        elif (layer_type == "GlobalAveragePooling1D") or (layer_type == "GlobalAveragePooling2D") or (layer_type == "GlobalAveragePooling3D"):
            tag_color = "#d3605bff"    
            border_color = "#9c3030ff"
            font_color = "#ffffffff"
            params = dict()
            txt = "GlobalAvgPool" + layer_type[-2:]
            oshape = layer.output_shape
        elif (layer_type == "GlobalMaxPooling1D") or (layer_type == "GlobalMaxPooling2D") or (layer_type == "GlobalMaxPooling3D"):
            tag_color = "#d3605bff"    
            border_color = "#9c3030ff"
            font_color = "#ffffffff"
            txt = "GlobalMaxPool" + layer_type[-2:]
            oshape = layer.output_shape
        elif (layer_type == "BatchNormalization"):
            tag_color = "#f6f65bff"
            border_color = "#b7b700ab"
            font_color = "#000000ff"
            txt = "Bnorm"
            params = dict()
        elif (layer_type == "Activation"):
            tag_color = "#bcfebcff"
            border_color = "#0fb40fa8"
            font_color = "#000000ff"
            if not iswrapper:
                if (layer.activation.func_name == 'relu'):
                    layer_typeName = 'ReLU'
                elif (layer.activation.func_name == 'elu'):
                    layer_typeName = 'eLU'
                elif (layer.activation.func_name == 'softmax'):
                    layer_typeName = 'Softmax'
                elif (layer.activation.func_name == 'softplus'):
                    layer_typeName = 'Softplus'
                elif (layer.activation.func_name == 'softsign'):
                    layer_typeName = 'Softsign'
                elif (layer.activation.func_name == 'tanh'):
                    layer_typeName = 'tanH'
                elif (layer.activation.func_name == 'sigmoid'):
                    layer_typeName = 'Sigmoid'
                elif (layer.activation.func_name == 'hard_sigmoid'):
                    layer_typeName = 'hard Sigmoid'
                elif (layer.activation.func_name == 'linear'):
                    layer_typeName = 'Linear'
            else:
                if (layer.layer.activation.func_name == 'relu'):
                    layer_typeName = 'ReLU'
                elif (layer.layer.activation.func_name == 'elu'):
                    layer_typeName = 'eLU'
                elif (layer.layer.activation.func_name == 'softmax'):
                    layer_typeName = 'Softmax'
                elif (layer.layer.activation.func_name == 'softplus'):
                    layer_typeName = 'Softplus'
                elif (layer.layer.activation.func_name == 'softsign'):
                    layer_typeName = 'Softsign'
                elif (layer.layer.activation.func_name == 'tanh'):
                    layer_typeName = 'tanH'
                elif (layer.layer.activation.func_name == 'sigmoid'):
                    layer_typeName = 'Sigmoid'
                elif (layer.layer.activation.func_name == 'hard_sigmoid'):
                    layer_typeName = 'hard Sigmoid'
                elif (layer.layer.activation.func_name == 'linear'):
                    layer_typeName = 'Linear' 
            txt = layer_typeName
            params = dict()
        elif (layer_type == "LeakyReLU") or (layer_type == "ELU"):
            tag_color = "#bcfebcff"
            border_color = "#0fb40fa8"
            font_color = "#000000ff"
            if not iswrapper:
                params = {'alpha':layer.alpha}
            else:
                params = {'alpha':layer.layer.alpha}
            if display_params:
                txt = u"\u03b1 = " + "%0.2f"%(params['alpha'])
                txt = txt.encode('utf-8')
                txt = [layer_typeName,
                       txt]
            else:
                txt = layer_typeName
        elif (layer_type == "PReLU") or (layer_type == "Softmax"):
            tag_color = "#bcfebcff"
            border_color = "#0fb40fa8"
            font_color = "#000000ff"
            txt = layer_type
            params = dict()
        elif (layer_type == "ThresholdedReLU"):
            tag_color = "#bcfebcff"
            border_color = "#0fb40fa8"
            font_color = "#000000ff"
            if not iswrapper:
                params = {'theta':layer.theta}
            else:
                params = {'theta':layer.layer.theta}
            if display_params:
                txt = u"\u03b8 = " + "%0.2f"%(params['theta'])
                txt = txt.encode('utf-8')
                txt = [layer_typeName,
                       txt]
            else:
                txt = layer_typeName
        elif (layer_type == "Flatten"):
            tag_color = "#c6afe9e3"
            border_color = "#975deeff"
            font_color = "#000000ff"
            txt = layer_typeName
            params = dict()
            oshape = layer.output_shape
        elif (layer_type == "Dense"):
            tag_color = "#e47ca6e3"
            border_color = "#aa114ee3"
            font_color = "#000000ff"
            if not iswrapper:
                params = {'units':layer.units}
            else:
                params = {'units':layer.layer.units}
            txt = layer_typeName + str(params['units'])
            oshape = layer.output_shape
        elif (layer_type == "Reshape") or (layer_type == "Permute") or (layer_type == "RepeatVector"):
            tag_color = "#cd8a63e2"
            border_color = "#a47a4aff"
            font_color = "#ffffffff"
            txt = layer_typeName 
            oshape = layer.output_shape
        elif (layer_type == "Lambda"):
            tag_color = "#ffb380ff"
            border_color = "#ff6600ff"
            font_color = "#000000ff"
            if display_lambdas:
                if not iswrapper:
                    layer_function = inspect.getsource(layer.function)
                else:
                    layer_function = inspect.getsource(layer.layer.function)
                #patt = re.compile("layer = Lambda\(lambda (\w+): ([^)]+)\)\(input\)")
                patt = re.compile("[^)]+\= Lambda\(\w+ \w+: ([^)]+)\)")
                matches = patt.findall(layer_function)[0]
                layer_function = matches.replace(' ','')
                txt = u"\u03bb: " + layer_function
                txt = txt.encode('utf-8')
            else:
                txt = u"\u03bb"
                txt = txt.encode('utf-8')
            oshape = layer.output_shape
        elif (layer_type == "ActivityRegularization"):
            tag_color = "#ffffffff"
            border_color = "#000000ff"
            font_color = "#000000ff"
            txt = "ActReg"
            params = dict()
        elif (layer_type == "Masking"):
            tag_color = "#ffffffff"
            border_color = "#000000ff"
            font_color = "#000000ff"
            txt = layer_typeName
            if not iswrapper:
                params = {"mask_value":layer.mask_value}
            else:
                params = {"mask_value":layer.layer.mask_value}
        elif (layer_type == "Concatenate"):
            tag_color = "#ffffffff"
            border_color = "#000000ff"
            font_color = "#000000ff"
            txt = "⌒"
            params = dict()
            oshape = layer.output_shape
        elif (layer_type == "Merge") or (layer_type == "Add"):
            tag_color = "#ffffffff"
            border_color = "#000000ff"
            font_color = "#000000ff"
            txt = "+"
            params = dict()
            oshape = layer.output_shape
        elif (layer_type == "Subtract"):
            tag_color = "#ffffffff"
            border_color = "#000000ff"
            font_color = "#000000ff"
            txt = "-"
            params = dict()
            oshape = layer.output_shape
        elif (layer_type == "Multiply"):
            tag_color = "#ffffffff"
            border_color = "#000000ff"
            font_color = "#000000ff"
            txt = "*"
            params = dict()
            oshape = layer.output_shape
        elif (layer_type == "Average"):
            tag_color = "#ffffffff"
            border_color = "#000000ff"
            font_color = "#000000ff"
            txt = "~"
            params = dict()
            oshape = layer.output_shape
        elif (layer_type == "Maximum"):
            tag_color = "#ffffffff"
            border_color = "#000000ff"
            font_color = "#000000ff"
            txt = "⋁"
            params = dict()
            oshape = layer.output_shape
        elif (layer_type == "Dot"):
            tag_color = "#ffffffff"
            border_color = "#000000ff"
            font_color = "#000000ff"
            txt = "×"
            params = dict()
            oshape = layer.output_shape
        elif (layer_type == "Dropout") or (layer_type == "SpatialDropout1D") or (layer_type == "SpatialDropout2D") or (layer_type == "SpatialDropout3D"):
            tag_color = "#00ffffff"
            border_color = "#006680ff"
            font_color = "#000000ff"
            if not iswrapper:
                params = {'rate':layer.rate}
            else:
                params = {'rate':layer.layer.rate}
            if (display_params):
                txt = [layer_typeName,str(int(100*params['rate']))+"%"]
            else:
                txt = layer_typeName
            
        elif (layer_type == "RNN"):
            tag_color = "#eeffaaff"
            border_color = "#aad400ab"
            font_color = "#000000ff"
            txt = layer_type
            params = dict()
            oshape = layer.output_shape
        elif (layer_type == "SimpleRNN") or (layer_type == "GRU") or (layer_type == "CuDNNGRU") or (layer_type == "CuDNNLSTM") or (layer_type == "LSTM"):
            tag_color = "#eeffaaff"
            border_color = "#aad400ab"
            font_color = "#000000ff"
            if not iswrapper:
                params = {'units':layer.units}
            else:
                params = {'units':layer.layer.units}
            if (display_params):
                txt = [layer_typeName,
                       str(params['units'])]
            else:
                txt = layer_typeName
            oshape = layer.output_shape
        elif (layer_type == "ConvLSTM2D"):
            tag_color = "#eeffaaff"
            border_color = "#aad400ab"
            font_color = "#000000ff"
            if not iswrapper:
                params = {"kernel":layer.kernel_size,
                      "strides":layer.strides}
            else:
                params = {"kernel":layer.layer.kernel_size,
                      "strides":layer.layer.strides}
            if (display_params):
                txt = [layer_typeName,
                       str(params['kernel'])]
            else:
                txt = layer_typeName
            oshape = layer.output_shape
        elif (layer_type == "Embedding"):
            tag_color = "#afafe9e3"
            border_color = "#3737c8ff"
            font_color = "#000000ff"
            txt = layer_type
            if not iswrapper:
                params = {"input_dim":layer.input_dim,
                          "output_dim":layer.output_dim,
                          "input_length":layer.input_length}
            else:
                params = {"input_dim":layer.layer.input_dim,
                          "output_dim":layer.layer.output_dim,
                          "input_length":layer.layer.input_length}
            oshape = layer.output_shape
        elif (layer_type == "GaussianNoise"):
            tag_color = "#e9c6afff"
            border_color = "#784421a8"
            font_color = "#000000ff"
            if not iswrapper:
                params = {"stddev":layer.stddev}
            else:
                params = {"stddev":layer.layer.stddev}
            if display_params:
                txt = u"\u03c3 = " + str(params['stddev'])
                txt = [layer_typeName,
                       txt.encode('utf-8')]
            else:
                txt = layer_typeName
            
        elif (layer_type == "GaussianDropout"):
            tag_color = "#e9c6afff"
            border_color = "#784421a8"
            font_color = "#000000ff"
            if not iswrapper:
                params = {"rate":layer.rate}
            else:
                params = {"rate":layer.layer.rate}
            if display_params:
                txt = [layer_typeName,
                       str(int(100*params['rate']))+"%"]
            else:
                txt = layer_typeName
            
        elif (layer_type == "AlphaDropout"):
            tag_color = "#e9c6afff"
            border_color = "#784421a8"
            font_color = "#000000ff"
            if not iswrapper:
                params = {"rate":layer.rate,
                      "noise_shape":layer.noise_shape}
            else:
                params = {"rate":layer.layer.rate,
                      "noise_shape":layer.layer.noise_shape}
            if display_params:
                txt = [layer_typeName,
                       str(int(100*params['rate']))+"%"]
            else:
                txt = layer_typeName
            
        else:
            tag_color = "#e9c6afff"
            border_color = "#005444a9"
            font_color = "#0000004f"
            txt = layer_typeName
            params = dict()
        
        layersInfo[layer_id]['typeName'] = layer_typeName
        layersInfo[layer_id]['tag'] = txt
        layersInfo[layer_id]['tagColor'] = tag_color
        layersInfo[layer_id]['borderColor'] = border_color
        layersInfo[layer_id]['fontColor'] = font_color
        layersInfo[layer_id]['params'] = params
        layersInfo[layer_id]['output_shape'] = oshape
        
        # Now let's calculate the size of this tag
        font = ImageFont.truetype("UbuntuMono-R.ttf", 40)
        params_font = ImageFont.truetype("Ubuntu-L.ttf", 20)
        if type(txt) is not list:
            tagSize = font.getsize(txt)
        else:
            tagSize = []
            for itxt_tmp,txt_tmp in enumerate(txt):
                if itxt_tmp == 0:
                    tagSize.append(font.getsize(txt_tmp))
                else:
                    tagSize.append(params_font.getsize(txt_tmp))
            #tagSize = [font.getsize(txt_tmp) for txt_tmp in txt]
            tag_space = 1
            tagSize = (np.max([t[0] for t in tagSize]),np.sum([t[1] + tag_space for t in tagSize]))
            tagSize = (tagSize[0], tagSize[1] + 2*tag_space)
            layersInfo[layer_id]['dotPosition'][3] = tagSize[1]
        h_border = 10
        bradius = 20
        border = 5
        
        layersInfo[layer_id]['tagSize'] = tagSize
        
        X0 = layersInfo[layer_id]['dotPosition'][0]
        Y0 = layersInfo[layer_id]['dotPosition'][1]
        W0 = layersInfo[layer_id]['dotPosition'][2]
        H0 = layersInfo[layer_id]['dotPosition'][3]
        
        bbox[0] = np.min((bbox[0],X0-tagSize[0]//2 - h_border-border))
        bbox[1] = np.min((bbox[1],Y0-border))
        bbox[2] = np.max((bbox[2],X0-tagSize[0]//2 + tagSize[0] + h_border+border))
        bbox[3] = np.max((bbox[3],Y0+H0+border))
        
        if (verbose):
            t5 = time.time()
            sys.stdout.write("...%f (s)\n"%(t5-t4))
        
        #_H = np.max((_H,Y0+H0+border))
        #_W = np.max((_W,X0+tagSize[0] + 2*h_border+border))
        
    #_H -= 4*border
    SvgTag = ""
    
    # define the arrow marker
    SvgTag += '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="0" refY="3" orient="auto" markerUnits="strokeWidth"> <path d="M0,1.5 L0,4.5 L3.5,3 z" fill="#000000" /> </marker> </defs>'
    
    if (verbose):
        t6 = time.time()
        sys.stdout.write("Creating individual svg tags...")
    
    _H = bbox[-1] + 100
    for layer_id in layersInfo:
        
        X0 = layersInfo[layer_id]['dotPosition'][0]
        Y0 = _H - layersInfo[layer_id]['dotPosition'][1]
        W0 = layersInfo[layer_id]['dotPosition'][2]
        H0 = layersInfo[layer_id]['dotPosition'][3]
        
        tagSize = layersInfo[layer_id]['tagSize']
        txt = layersInfo[layer_id]['tag']
        tag_color = layersInfo[layer_id]['tagColor']
        border_color = layersInfo[layer_id]['borderColor']
        font_color = layersInfo[layer_id]['fontColor']
        
        # it is important than H0 > tagSize[1]
        if type(txt) is not list:
            # check if concatenate or sum
            if (txt == "⌒"):
                radius = 30
                dy = 3
                circle_svg = '<circle cx="%f" cy="%f" r="%f" stroke="black" stroke-width="5" fill="white" />'%(X0,Y0+radius-dy,radius)
                circle_svg += '<path stroke-width="5" d="M%f,%f C %f %f, %f %f, %f,%f" stroke="black" fill="none" />'%(X0-15,Y0+5+radius-dy,X0-15,Y0-15+radius-dy,X0+15,Y0-15+radius-dy,X0+15,Y0+5+radius-dy)
                tagSvg = "<g>" + circle_svg + "</g>"    

            elif (txt == "+") or (txt == "-") or (txt == "*") or (txt == "~") or (txt == "·") or (txt == "×"):
                radius = 30
                dy = 3
                circle_svg = '<circle cx="%f" cy="%f" r="%f" stroke="black" stroke-width="5" fill="white" />'%(X0,Y0+radius-dy,radius)
                circle_svg += '<text x="%f" y="%f" text-anchor="middle" fill="%s" fill-opacity="%1.2f" font-size="50px" font-family="Ubuntu Light" dy=".3em">%s</text>'%(X0,Y0+radius-dy,font_color[0:-2],_hex2RGB(font_color)[3]/255.,txt)
                tagSvg = "<g>" + circle_svg + "</g>" 
            elif (txt == "⋁"):
                radius = 30
                dy = 3
                circle_svg = '<circle cx="%f" cy="%f" r="%f" stroke="black" stroke-width="5" fill="white" />'%(X0,Y0+radius-dy,radius)
                circle_svg += '<text x="%f" y="%f" text-anchor="middle" fill="%s" fill-opacity="%1.2f" font-size="30px" font-family="Ubuntu Light" dy=".3em">%s</text>'%(X0,Y0+radius-dy,font_color[0:-2],_hex2RGB(font_color)[3]/255.,txt)
                tagSvg = "<g>" + circle_svg + "</g>" 
            else:
                outter_rectangle_svg = '<rect x="%f" y="%f" width="%f" height="%f" rx="%f" ry="%f" fill="%s" fill-opacity="%1.2f" />'%(X0-tagSize[0]//2 - h_border-border,Y0-border,tagSize[0] + 2*h_border+2*border,H0+2*border,1.2*bradius,1.2*bradius,border_color[0:-2],_hex2RGB(border_color)[3]/255.)
                inner_rectangle_svg = '<rect x="%f" y="%f" width="%f" height="%f" rx="%f" ry="%f" fill="%s" fill-opacity="%1.2f" />'%(X0-tagSize[0]//2 - h_border,Y0,tagSize[0] + 2*h_border,H0,bradius,bradius,tag_color[0:-2],_hex2RGB(tag_color)[3]/255.)
                text_svg = '<text x="%f" y="%f" text-anchor="middle" fill="%s" fill-opacity="%1.2f" font-size="30px" font-family="Ubuntu Light" dy=".3em">%s</text>'%(X0,Y0+H0//2,font_color[0:-2],_hex2RGB(font_color)[3]/255.,txt)
                tagSvg = "<g>" + outter_rectangle_svg + inner_rectangle_svg + text_svg + "</g>"    
        
        else:
            outter_rectangle_svg = '<rect x="%f" y="%f" width="%f" height="%f" rx="%f" ry="%f" fill="%s" fill-opacity="%1.2f" />'%(X0-tagSize[0]//2 - h_border-border,Y0-border,tagSize[0] + 2*h_border+2*border,H0+2*border,1.2*bradius,1.2*bradius,border_color[0:-2],_hex2RGB(border_color)[3]/255.)
            inner_rectangle_svg = '<rect x="%f" y="%f" width="%f" height="%f" rx="%f" ry="%f" fill="%s" fill-opacity="%1.2f" />'%(X0-tagSize[0]//2 - h_border,Y0,tagSize[0] + 2*h_border,H0,bradius,bradius,tag_color[0:-2],_hex2RGB(tag_color)[3]/255.)
                
            ycum = 0
            text_svg = ""
            for ittx,ttx in enumerate(txt):
                if ittx == 0:
                    tmp_size = font.getsize(ttx)
                    text_svg += '<text x="%f" y="%f" text-anchor="middle" fill="%s" fill-opacity="%1.2f" font-size="30px" font-family="Ubuntu Light" font-weight="normal" dy=".3em">%s</text>'%(X0,ycum+Y0+tmp_size[1]//2,font_color[0:-2],_hex2RGB(font_color)[3]/255.,ttx)
                else:
                    tmp_size = params_font.getsize(ttx)
                    text_svg += '<text x="%f" y="%f" text-anchor="middle" fill="%s" fill-opacity="%1.2f" font-size="20px" font-family="Ubuntu Light" dy=".3em">%s</text>'%(X0,ycum+Y0+tmp_size[1]//2,font_color[0:-2],_hex2RGB(font_color)[3]/255.,ttx)                    
                ycum += tmp_size[1]
            
            tagSvg = "<g>" + outter_rectangle_svg + inner_rectangle_svg + text_svg + "</g>"    

        
        x0 = X0 - tagSize[0]//2 - h_border - border
        x1 = X0 - tagSize[0]//2 + tagSize[0] + h_border + border
        y0 = Y0 - border
        y1 = Y0 + H0 + border
        layersInfo[layer_id]['outter_bbox'] = [x0,y0,x1,y1] 
        
        
        SvgTag += tagSvg
        
    if (verbose):
        t7 = time.time()
        sys.stdout.write("%f (s)\n"%(t7-t6))
            
    if (verbose):
        t8 = time.time()
        sys.stdout.write("Adding edges to graph...")
        
    # Now we need to add the edges (lines)
    edgesLength = []
    stroke_width = 4
    i = 1
    ddot_tmp = ddot[i]
    while (ddot_tmp != "stop"):
        ddot_type = ddot_tmp[0:4]
        if ddot_type == "edge":
            #pattern = re.compile("edge (\d+) (\d+) (\d+) ([0-9]+(?:\.[0-9]+)?) ([0-9]+(?:\.[0-9]+)?) ([0-9]+(?:\.[0-9]+)?) ([0-9]+(?:\.[0-9]+)?) ([0-9]+(?:\.[0-9]+)?) ([0-9]+(?:\.[0-9]+)?) ([0-9]+(?:\.[0-9]+)?) ([0-9]+(?:\.[0-9]+)?) (\w+) (\w+)")
            pattern = re.compile("edge (\d+) (\d+)")
            
            matches = pattern.findall(ddot_tmp)[0]
            startId = matches[0]
            endId = matches[1]
            
            xy0 = layersInfo[startId]['outter_bbox']
            xy1 = layersInfo[endId]['outter_bbox']
            
            x0 = (xy0[2]-xy0[0])/2 + xy0[0]
            y0 = xy0[3]
            
            x1 = (xy1[2]-xy1[0])/2 + xy1[0]
            y1 = xy1[1]
            
            edgesLength.append(y1-y0)
            
            bezierSvg = '<path stroke-width="%i" d="M%f,%f C %f %f, %f %f, %f %f" stroke="black" fill="none" marker-end="url(#arrow)" />'%(stroke_width,x0,y0,x0,y1-13,x1,y0,x1,y1-13)
            
            # Now add the output_shape tag
            shapeTagSvg =  ""
            if (layersInfo[startId]['output_shape'] != [] and display_shapes):
                shapeTagSvg = '<text x="%f" y="%f" text-anchor="start" alignment-baseline="hanging" fill="#000000" font-size="20px" font-family="Ubuntu Light" dy=".3em">%s</text>'%(np.abs((x0+x1)/2)+5,y0+10,str(layersInfo[startId]['output_shape'][1:]))

            SvgTag += bezierSvg + shapeTagSvg
        
        i += 1
        ddot_tmp = ddot[i]  
        #print(ddot_tmp)
        
    # add output boxes & final shape in case we need it
    L = 40
    if (edgesLength != []):
        L = np.median(edgesLength)
    
    output_layers = []
    if hasattr(model,'output_layers'):
        output_layers = model.output_layers
    elif hasattr(model,'outputs'):
        output_layers = model.outputs
    
    print(layersInfo)
    for out in output_layers:
        if str(id(out)) in layersInfo:
            xy0 = layersInfo[str(id(out))]['outter_bbox']
            # Get the position of the box
            tagSize = font.getsize("Output")
            X0 = (xy0[0]+xy0[2])/2
            Y0 = xy0[3] + L 
            bbox[3] += L
            _H = np.max((_H,Y0))
            
            tag_color = "#acacace2"
            border_color = "#4d4d4da9"
            font_color = "#ffffffff"
            
            outter_rectangle_svg = '<rect x="%f" y="%f" width="%f" height="%f" rx="%f" ry="%f" fill="%s" fill-opacity="%1.2f" />'%(X0-tagSize[0]//2 - h_border-border,Y0-border,tagSize[0] + 2*h_border+2*border,H0+2*border,1.2*bradius,1.2*bradius,border_color[0:-2],_hex2RGB(border_color)[3]/255.)
            inner_rectangle_svg = '<rect x="%f" y="%f" width="%f" height="%f" rx="%f" ry="%f" fill="%s" fill-opacity="%1.2f" />'%(X0-tagSize[0]//2 - h_border,Y0,tagSize[0] + 2*h_border,H0,bradius,bradius,tag_color[0:-2],_hex2RGB(tag_color)[3]/255.)
            text_svg = '<text x="%f" y="%f" text-anchor="middle" alignment-baseline="middle" fill="%s" fill-opacity="%1.2f" font-size="30px" font-family="Ubuntu Light" font-weight="normal" dy=".3em">Output</text>'%(X0,Y0+(tagSize[1] - tagSize[1]/2 + border),font_color[0:-2],_hex2RGB(font_color)[3]/255.)
        
            tagSvg = "<g>" + outter_rectangle_svg + inner_rectangle_svg + text_svg + "</g>"    
    
            # Now add edge (line)
            bezierSvg = '<path stroke-width="%i" d="M%f,%f C %f %f, %f %f, %f %f" stroke="black" fill="none" marker-end="url(#arrow)" />'%(stroke_width,X0,xy0[3],X0,Y0-13-border,X0,xy0[3],X0,Y0-13-border)
    
            SvgTag += tagSvg + bezierSvg
        
            if (display_shapes):
                SvgTag += '<text x="%f" y="%f" text-anchor="middle" alignment-baseline="middle" fill="#000000" font-size="20px" font-family="Ubuntu Light" dy=".3em">%s</text>'%(X0,Y0+H0+border+10,str(out.output_shape[1:]))

    if (verbose):
        t9 = time.time()
        sys.stdout.write("%f (s)\n"%(t9-t8))
    
    SvgTag = '<svg viewBox="%i %i %i %i">'%(bbox[0],_H-bbox[3],bbox[2]-bbox[0]+30,_H) + SvgTag + '</svg>'

    # write to file
    if filename is None:
        filename = "model.svg"
    else:
        if not ".svg" in filename:
            filename += ".svg"
    svgFile = open(filename,"w")
    svgFile.write(SvgTag)
    svgFile.close()

