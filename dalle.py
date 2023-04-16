import token
import torch
from dalle2_pytorch import DALLE2, DiffusionPriorNetwork, DiffusionPrior, Unet, Decoder, CLIP
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os 
import torchvision.transforms as transforms
from tqdm.notebook import tqdm
from pkg_resources import packaging
import clip 
#import uuid
#from image import PIL
DATADIR = "S:\imagedataset\PetImages\catsample"
test_text = []
image_set = []
tester = []
tensors = []
token_text = []

transform = transforms.Compose([
    transforms.ToTensor()
])
imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

for filename in os.listdir(DATADIR):
    if filename.endswith(".jpg"):
        # Prints only text file present in My Folder
        test_text.append(filename[:-4])
        
def create_imageset():
    for image in os.listdir(DATADIR):
        IMG_SIZE = 256
        try:
            #reads in image. cv2 uses bgr color so you have to convert to rgb before printing
            image_array = cv2.imread(os.path.join(DATADIR,image), cv2.IMREAD_ANYCOLOR) #converts the image to an array on pixel values
            #rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            test_array = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))
            image_set.append(test_array)
        except Exception as e:
            #might be useful in the future 
            pass

def test_image():
    i=0
    for image in os.listdir(DATADIR):
        IMG_SIZE = 256
        image_array = cv2.imread(os.path.join(DATADIR, image), cv2.IMREAD_ANYCOLOR) #converts the image to an array on pixel values
        test_array = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))
        tester = cv2.cvtColor(test_array, cv2.COLOR_BGR2RGB)
        i+=1 
        if i == 1:
            return tester

def transform_tensors(list):
    for tensor in list:
        tensor = transform(tensor)
        tensor = tensor.unsqueeze(0).cuda()
        tensors.append(tensor)

def tokenize_text(list):
    for text in list:
        text = clip.tokenize(text).cuda()
        token_text.append(text)

create_imageset()
transform_tensors(image_set)
tokenize_text(test_text)
#tester = test_image()
#print(tester.dtype)
#print(len(test_text))
#print(len(image_set))

#tensor = transform(tester)
#tensor = transform()
print(tensors[0].size()) #3, 256, 256
#tensor = torch.stack(tensors) #didnt work because each tensor has a different size at entry 0 
#print(tensor[4644].size()) 1, 25

#test_array = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))
#rgb = cv2.cvtColor(image_set[1], cv2.COLOR_BGR2RGB)
#plt.imshow(tester)
#plt.show()
model, preprocess = clip.load("ViT-B/32")
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

#print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}") #151,277,313
#print("Input resolution:", input_resolution) #224
#print("Context length:", context_length) #77
#print("Vocab size:", vocab_size) #49408

def zeroshot_classifier(classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


#zeroshot_weights = zeroshot_classifier(test_text, imagenet_templates)
#print(test_text[0])
#print(clip.tokenize(test_text[0]))

#nksajnd.sp() #rest here 

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
#text = torch.randint(0, 49408, (4, 256)).cuda()
#images = torch.randn(4, 3, 256, 256).cuda()

#oijfds.pdo()
# real data 
text = token_text[1]
images = tensors[1]

#print(test_text[1])
#oijfds.pdo()
#images = torch.tensor(tensors).to(torch.int64)
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
    ['black and orange cat'],
    cond_scale = 2. # classifier free guidance strength (> 1 would strengthen the condition)
).detach().cpu().numpy()

images = images.squeeze(axis = 0)
images = np.transpose(images, (1,2,0)) 
plt.imsave("dalle2_output.png",images)

# save your image (in this example, of size 256x256)
