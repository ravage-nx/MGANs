-- a script for synthesising images with MGAN
require 'torch'
require 'nn'
require 'optim'
require 'cutorch'
require 'cunn'
require 'image'

--camera--
require 'sys'
require 'camera'
require 'xlua'

loadcaffe_wrap = paths.dofile('lib/loadcaffe_wrapper.lua')
util = paths.dofile('lib/util.lua')
pl = require('pl.import_into')()


local opt = {}

--opt.model_name = '../model/VG_Alpilles_ImageNet100_epoch20.t7'
opt.model_name = '../model/Picasso.t7'
--opt.model_name = '../model/Kandinsky_Complex.t7'
opt.output_folder_name = '../Dataset/VG_Alpilles_ImageNet100/syn_VG_Alpilles_ImageNet100_epoch5/'

opt.max_length = 512 -- change this value for image size. Larger images needs more gpu memory.
opt.stand_atom = 8 -- make sure the image size can be divided by stand_atom
opt.noise_weight = 0.0 -- change this weight for balancing between style and content. Stronger noise makes the synthesis more stylish. 
opt.noise_name = 'noise.jpg' -- low frequency noise image
opt.gpu = 1

-- camera
dev = 0
width = 640
height = 480
fps = 30


camera1 = image.Camera{idx=dev, width=width, height=height, fps=fps}
local image_input_ori = camera1:forward()
local image_syn = camera1:forward()
win1 = image.display{win=win1, image={image_input_ori}}
win2 = image.display{win=win2, image={image_syn}}

f = 1

-- load data
---------------
local noise_image = image.load(opt.noise_name, 3)
-------------
local net = util.load(opt.model_name, opt.gpu)
net:cuda()
print(net)

print('*****************************************************')
print('Testing: ');
print('*****************************************************') 
local counter = 0
while true  do
    -- resize the image image
    image_input_ori = camera1:forward()
    local image_input = image_input_ori
    local max_dim = math.max(image_input:size()[2], image_input:size()[3])
    local scale = opt.max_length / max_dim
    local new_dim_x = math.floor((image_input:size()[3] * scale) / opt.stand_atom) * opt.stand_atom
    local new_dim_y = math.floor((image_input:size()[2] * scale) / opt.stand_atom) * opt.stand_atom
    image_input = image.scale(image_input, new_dim_x, new_dim_y, 'bilinear')
    
    -- add noise to the image (improve background quality)
    local noise_image_ = image.scale(noise_image, new_dim_x, new_dim_y, 'bilinear') 
    image_input:add(noise_image_:mul(opt.noise_weight))
    image_input:resize(1, image_input:size()[1], image_input:size()[2], image_input:size()[3])
    image_input:mul(2):add(-1)
    image_input = image_input:cuda()

    -- decode image with a single forward prop
	local tm = torch.Timer()
	image_syn = net:forward(image_input)  
    cutorch.synchronize()
	print(string.format('Image size: %d by %d, time: %f', image_input:size()[3], image_input:size()[4], tm:time().real))
    image_syn = image.toDisplayTensor{input = image_syn, nrow = math.ceil(math.sqrt(image_syn:size(1)))}

    -- save image
    image.display{win=win1, image={image_input_ori}}
    image.display{win=win2, image={image_syn}}
    
end

image_input = nil
image_syn = nil
noise_image_ = nil
net = nil
collectgarbage()
