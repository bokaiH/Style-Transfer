####################################
# git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# cd pytorch-CycleGAN-and-pix2pix
# bash ./scripts/download_cyclegan_model.sh horse2zebra
# make sure you git clone the needed library before you run this model.
# Here we just use cyclegan to compare with NST and LapStyle, so we didn't explore this model
####################################

import os
import torch
from options.test_options import TestOptions
from models import create_model
from util import util
from PIL import Image
import torchvision.transforms as transforms

# Instantiate TestOptions without parsing command-line arguments
opt = TestOptions()

# Manually set options for testing
opt.dataroot = './datasets/horse2zebra'  # Path to test data
opt.checkpoints_dir = './checkpoints'  # Directory to load models from
opt.results_dir = './results/'  # Directory to save results
opt.name = 'horse2zebra_pretrained'  # Pretrained model name
opt.model = 'test'  # Test mode
opt.isTrain = False  # Set to False for testing
opt.no_dropout = True  # No dropout
opt.num_threads = 0  # Single-threaded
opt.batch_size = 1  # Batch size of 1
opt.serial_batches = True  # Load data sequentially
opt.preprocess = 'resize_and_crop'  # Preprocessing method
opt.load_size = 2048  # Input image size
opt.crop_size = 2048  # Crop size
opt.gpu_ids = []  # Use CPU
opt.direction = 'AtoB'  # Direction of translation
opt.num_test = 50  # Number of test examples
opt.aspect_ratio = 1.0  # Aspect ratio
opt.dataset_mode = 'single'  # Dataset mode
opt.display_winsize = 256  # Display window size
opt.epoch = 'latest'  # Which epoch to load
opt.eval = False  # Evaluation mode
opt.init_gain = 0.02  # Initialization gain
opt.init_type = 'normal'  # Initialization type
opt.input_nc = 3  # Number of input channels
opt.load_iter = 0  # Which iteration to load
opt.max_dataset_size = float("inf")  # Maximum dataset size
opt.n_layers_D = 3  # Number of layers in discriminator
opt.ndf = 64  # Number of discriminator filters
opt.netD = 'basic'  # Type of discriminator
opt.netG = 'resnet_9blocks'  # Type of generator
opt.ngf = 64  # Number of generator filters
opt.norm = 'instance'  # Normalization layer
opt.output_nc = 3  # Number of output channels
opt.phase = 'test'  # Phase (train/test)
opt.suffix = ''  # Suffix for model name
opt.use_wandb = False  # Use Weights & Biases
opt.wandb_project_name = 'CycleGAN-and-pix2pix'  # W&B project name
opt.verbose = False  # Verbose output

# 添加缺失的属性
opt.model_suffix = ''  # Model suffix

# Load the model
model = create_model(opt)
model.setup(opt)

# Load input image
input_image_path = 'C:\\Users\86191\Downloads\ziyue.png'  # Input image path
output_image_path = 'C:/Users/86191/Downloads/ziyue2zebra.jpg'  # Output image path

# Ensure the input image exists
if not os.path.exists(input_image_path):
    raise FileNotFoundError(f"Input image not found at {input_image_path}")

input_image = Image.open(input_image_path).convert('RGB')

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((opt.load_size, opt.load_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
input_tensor = transform(input_image).unsqueeze(0)

# Inference to generate stylized image
model.eval()
with torch.no_grad():
    # Move tensor to appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() and len(opt.gpu_ids) > 0 else 'cpu')
    input_tensor = input_tensor.to(device)

    # Ensure the model is on the correct device
    model.netG.to(device)

    output_tensor = model.netG(input_tensor)  # Use generator to transform image

# Convert to image and save
output_image = util.tensor2im(output_tensor)
output_image = Image.fromarray(output_image)

# Ensure the output directory exists
output_dir = os.path.dirname(output_image_path)
os.makedirs(output_dir, exist_ok=True)

output_image.save(output_image_path)

print(f"风格迁移完成，图片已保存至: {output_image_path}")