import torch
from dalle2_pytorch import DALLE2, DiffusionPriorNetwork, DiffusionPrior, Unet, Decoder, CLIP
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os 

#import uuid
#from image import PIL
DATADIR = "S:\imagedataset\PetImages\Cat"
test_text = []
image_set = []
for filename in os.listdir(DATADIR):
    if filename.endswith(".jpg"):
        # Prints only text file present in My Folder
        #print(filename)
        test_text.append(filename)
        
def create_imageset():
    for image in os.listdir(DATADIR):
        IMG_SIZE = 512
        try:
            #reads in image. cv2 uses bgr color so you have to convert to rgb before printing
            image_array = cv2.imread(os.path.join(DATADIR,image), cv2.IMREAD_ANYCOLOR) #converts the image to an array on pixel values
            #rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            test_array = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))
            image_set.append(test_array)
            #plt.imshow(rgb)
            #plt.show()
            #print(image_array)
        except Exception as e:
            #might be useful in the future 
            pass
create_imageset()
print(len(test_text))
print(len(image_set))
#test_array = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))
#rgb = cv2.cvtColor(test_array, cv2.COLOR_BGR2RGB)
#plt.imshow(rgb)
#plt.show()
nksajnd.sp()
clip = CLIP(
    dim_text = 512,
    dim_image = 512,
    dim_latent = 512,
    num_text_tokens = 49408,
    text_enc_depth = 6,
    text_seq_len = 256,
    text_heads = 8,
    visual_enc_depth = 6,
    visual_image_size = 256,
    visual_patch_size = 32,
    visual_heads = 8
).cuda()

# mock data

text = torch.randint(0, 49408, (4, 256)).cuda()
images = torch.randn(4, 3, 256, 256).cuda()

# train

loss = clip(
    text,
    images,
    return_loss = True
)

loss.backward()

# do above for many steps ...

# prior networks (with transformer)

prior_network = DiffusionPriorNetwork(
    dim = 512,
    depth = 6,
    dim_head = 64,
    heads = 8
).cuda()

diffusion_prior = DiffusionPrior(
    net = prior_network,
    clip = clip,
    timesteps = 1000,
    sample_timesteps = 64, 
    cond_drop_prob = 0.2
).cuda()

loss = diffusion_prior(text, images)
loss.backward()

# do above for many steps ...

# decoder (with unet)

unet1 = Unet(
    dim = 128,
    image_embed_dim = 512,
    text_embed_dim = 512,
    cond_dim = 128,
    channels = 3,
    dim_mults=(1, 2, 4, 8),
    cond_on_text_encodings = True    # set to True for any unets that need to be conditioned on text encodings
).cuda()

unet2 = Unet(
    dim = 16,
    image_embed_dim = 512,
    cond_dim = 128,
    channels = 3,
    dim_mults = (1, 2, 4, 8, 16)
).cuda()

decoder = Decoder(
    unet = (unet1, unet2),
    image_sizes = (128, 256),
    clip = clip,
    timesteps = 100,
    image_cond_drop_prob = 0.1,
    text_cond_drop_prob = 0.5
).cuda()

for unet_number in (1, 2):
    loss = decoder(images, text = text, unet_number = unet_number) # this can optionally be decoder(images, text) if you wish to condition on the text encodings as well, though it was hinted in the paper it didn't do much
    loss.backward()

# do above for many steps

dalle2 = DALLE2(
    prior = diffusion_prior,
    decoder = decoder
)

images = dalle2(
    ['cute puppy chasing after a squirrel'],
    cond_scale = 2. # classifier free guidance strength (> 1 would strengthen the condition)
).detach().cpu().numpy()


images = images.squeeze(axis = 0)
images = np.transpose(images, (1,2,0)) 
plt.imsave("dalle2_output.png",images)

# save your image (in this example, of size 256x256)
