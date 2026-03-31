import torch
import cv2
import numpy as np
from models.e2vid.model.model import *
from models.e2vid.utils.inference_utils import CropParameters, EventPreprocessor, IntensityRescaler, ImageFilter, ImageDisplay, \
    ImageWriter, UnsharpMaskFilter
from models.e2vid.utils.inference_utils import upsample_color_image, \
    merge_channels_into_color_image  # for color reconstruction
from models.e2vid.utils.util import robust_min, robust_max
from models.e2vid.utils.timers import CudaTimer, cuda_timers
from os.path import join
from collections import deque
import torchvision.transforms as transforms
from PIL import Image


class ImageReconstructor:
    def __init__(self, model, height, width, num_bins, device, augmentation=False, standardization=False):

        self.model = model
        self.device = device
        self.height = height
        self.width = width
        self.num_bins = num_bins

        self.standardization = standardization
        self.augmentation = augmentation
    
        self.initialize()

    def initialize(self):
        self.crop = CropParameters(self.width, self.height, self.model.num_encoders)
        self.last_states_for_each_channel = {'grayscale': None}

    def update_reconstruction(self, event_tensor, event_tensor_id=None, stamp=None):
        with torch.no_grad():

            with CudaTimer('Reconstruction'):

                with CudaTimer('NumPy (CPU) -> Tensor (GPU)'):
                    events = event_tensor
                    events = events.to(self.device)

                events_for_each_channel = {'grayscale': self.crop.pad(events)}
                reconstructions_for_each_channel = {}

                for channel in events_for_each_channel.keys():
                    with CudaTimer('Inference'):
                        new_predicted_frame, states, latent = self.model(events_for_each_channel[channel],
                                                                         self.last_states_for_each_channel[channel])

                    self.last_states_for_each_channel[channel] = states

                    with CudaTimer('Tensor (GPU) -> NumPy (CPU)'):
                        reconstructions_for_each_channel[channel] = new_predicted_frame
    
                out = reconstructions_for_each_channel['grayscale']

        return out, states, latent


class PostProcessor:
    def __init__(self, device, options):
        self.device = device
        self.unsharp_mask_filter = UnsharpMaskFilter(options, device=self.device)
        self.intensity_rescaler = IntensityRescaler(options)
        self.image_filter = ImageFilter(options)

    def process(self, new_predicted_frame):
        with torch.no_grad():
            # Unsharp mask (on GPU)
            new_predicted_frame = self.unsharp_mask_filter(new_predicted_frame)

            # Intensity rescaler (on GPU)
            new_predicted_frame = self.intensity_rescaler(new_predicted_frame)

            out = new_predicted_frame.cpu().numpy()

            # Post-processing, e.g bilateral filter (on CPU)
            out = torch.from_numpy(self.image_filter(out)).to(self.device)
        return out
