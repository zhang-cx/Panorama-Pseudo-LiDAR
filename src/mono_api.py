from __future__ import absolute_import, division, print_function
import sys
sys.path.append("./monodepth2")
import os
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import monodepth2.networks as networks
from monodepth2.utils import download_model_if_doesnt_exist
import cv2

class monodepth2():
    def __init__(self,model_name="mono_1024x320"):
        self.model_name = model_name
        download_model_if_doesnt_exist(model_name)
        encoder_path = os.path.join("./monodepth2/models", model_name, "encoder.pth")
        depth_decoder_path = os.path.join("./monodepth2/models", model_name, "depth.pth")
        
        # LOADING PRETRAINED MODEL
        self.encoder = networks.ResnetEncoder(18, False)
        self.depth_decoder = networks.DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))

        loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)

        loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
        self.depth_decoder.load_state_dict(loaded_dict)

        self.encoder.eval()
        self.depth_decoder.eval();

        self.feed_height = loaded_dict_enc['height']
        self.feed_width = loaded_dict_enc['width']

    def pil_to_tensor(self,image_path):
        original_width, original_height = input_image.size
        feed_height = self.feed_height
        feed_width = self.feed_width
        input_image_resized = input_image.resize((feed_width, feed_height), pil.LANCZOS)
        input_image_pytorch = transforms.ToTensor()(input_image_resized).unsqueeze(0)
        return input_image_tensor, (original_width,original_height)

    def np_to_tensor(self,input_image):
        feed_height = self.feed_height
        feed_width = self.feed_width
        size = input_image.shape
        input_image_resized = cv2.resize(input_image,(feed_width, feed_height))
        input_image_pytorch = transforms.ToTensor()(input_image_resized).unsqueeze(0)
        return input_image_pytorch,size[:2]

    def get_disp(self,tensor,size):
        with torch.no_grad():
            features = self.encoder(tensor)
            outputs = self.depth_decoder(features)
        disp = outputs[("disp",0)]
        disp_resized = torch.nn.functional.interpolate(disp,size, mode="bilinear", align_corners=False)
        return disp_resized

    def inference_from_path(self,image_path):
        input_image = pil.open(image_path).convert('RGB')
        return self.get_disp(self.pil_to_tensor(input_image))

    def inference_sequence(self,imgs):
        disps = []
        for idx,img in enumerate(imgs):
            disps.append(self.inference(img))
        return disps

    def inference(self,img):
        tensor,size = self.np_to_tensor(img)
        return self.get_disp(tensor,size).squeeze().cpu().numpy()



