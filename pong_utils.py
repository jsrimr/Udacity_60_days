import numpy as np
import torch

device   = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# preprocess a single frame
# crop image and downsample to 80x80
# stack two frames together as input
# def preprocess_single(image, bkg_color = np.array([144, 72, 17])):
#     img = np.mean(image[34:-16:2,::2]-bkg_color, axis=-1)/255.
#     return img


def preprocess_single(image, bkg_color = np.array([144, 72, 17])):
    img = np.mean(image[34:-16:2,::2]-bkg_color, axis=-1)/255.
    return torch.from_numpy(img).float().to(device)
# convert outputs of parallelEnv to inputs to pytorch neural net
# this is useful for batch processing especially on the GPU
def preprocess_batch(images, bkg_color = np.array([144, 72, 17])):
    list_of_images = np.asarray(images)
    if len(list_of_images.shape) < 5:
        list_of_images = np.expand_dims(list_of_images, 1)
    # subtract bkg and crop
    list_of_images_prepro = np.mean(list_of_images[:,:,34:-16:2,::2]-bkg_color,
                                    axis=-1)/255.
    
    # batch_input = np.swapaxes(list_of_images_prepro,0,1)
    # return torch.from_numpy(batch_input).float().to(device)
    
    return torch.from_numpy(list_of_images_prepro).float().to(device)
