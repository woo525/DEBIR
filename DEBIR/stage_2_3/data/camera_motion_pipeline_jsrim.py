# Copyright (c) 2021 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# The file contains code from the unprocessing repo (http://timothybrooks.com/tech/unprocessing )
# Copyright (c) 2021 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# The file contains code from the unprocessing repo (http://timothybrooks.com/tech/unprocessing )

""" Based on http://timothybrooks.com/tech/unprocessing 
Functions for forward and inverse camera pipeline. All functions input a torch float tensor of shape (c, h, w). 
Additionally, some also support batch operations, i.e. inputs of shape (b, c, h, w)
"""

import torch
import random
import math
import glob
import numpy as np
import scipy.io


# lin2xyz
M_lin2xyz = scipy.io.loadmat('mat_collections/M_lin2xyz.mat')['M'].astype('float32')

# xyz2lin
M_xyz2lin = scipy.io.loadmat('mat_collections/M_xyz2lin.mat')['M'].astype('float32')

# ccm collections
cam2xyz_path = glob.glob('mat_collections/ccm*.mat')
cam2lin_list = []
lin2cam_list = []
for cam2xyz_path in cam2xyz_path:
    key = cam2xyz_path.split('/')[-2]
    cam2xyz_matrix = scipy.io.loadmat(cam2xyz_path)['colorCorrectionMatrix'][0, 1].astype('float32')
    cam2lin_transpose_matrix = np.matmul(M_xyz2lin.transpose(), cam2xyz_matrix[0].transpose())
    cam2lin_list.append(torch.tensor(cam2lin_transpose_matrix))

    lin2cam_transpose_matrix = np.matmul(np.linalg.inv(cam2xyz_matrix).transpose(), M_lin2xyz.transpose())
    lin2cam_list.append(torch.tensor(lin2cam_transpose_matrix))

def random_ccm():
    """Generates random RGB -> Camera color correction matrices."""
    # Takes a random convex combination of XYZ -> Camera CCMs.

    index = np.random.randint(0, 2)
    rgb2cam = lin2cam_list[index]

    return rgb2cam

def random_gains():
    """Generates random gains for brightening and white balance."""
    # RGB gain represents brightening.
    rgb_gain = 1.0 / random.gauss(mu=0.8, sigma=0.1)

    # Red and blue gains represent white balance.
    red_gain = random.uniform(1.9, 2.4)
    blue_gain = random.uniform(1.5, 1.9)

    return rgb_gain, red_gain, blue_gain

def apply_smoothstep(image):
    """Apply global tone mapping curve."""
    image_out = 3 * image**2 - 2 * image**3
    return image_out

def invert_smoothstep(image):
    """Approximately inverts a global tone mapping curve."""
    image = image.clamp(0.0, 1.0)
    return 0.5 - torch.sin(torch.asin(1.0 - 2.0 * image) / 3.0)

def gamma_expansion(image):

    x = image

    gamma = 2.4
    a = 1 / 1.055
    b = 0.055 / 1.055
    c = 1 / 12.92
    d = 0.04045

    in_sign = -2 * torch.lt(x, 0) + 1
    abs_x = torch.abs(x)

    lin_range = torch.lt(abs_x ,d)
    gamma_range = torch.logical_not(lin_range)

    lin_value = (c * abs_x)
    gamma_value = torch.exp(gamma * torch.log(a * abs_x + b))

    new_x = (lin_value * lin_range) + (gamma_value * gamma_range)
    new_x = new_x * in_sign

    return new_x

def gamma_compression(image):

    x = image

    gamma = 1/2.4
    a = 1.055
    b = -0.055
    c = 12.92
    d = 0.0031308

    in_sign = -2 * torch.lt(x, 0) + 1
    abs_x = torch.abs(x)

    lin_range = torch.lt(abs_x ,d)
    gamma_range = torch.logical_not(lin_range)

    lin_range = lin_range
    gamma_range = gamma_range

    lin_value = (c * abs_x)
    gamma_value = a * torch.exp(gamma * torch.log(abs_x)) + b
    new_x = (lin_value * lin_range) + (gamma_value * gamma_range)
    new_x = new_x * in_sign

    return new_x

def apply_ccm(image, ccm):
    """Applies a color correction matrix."""
    assert image.dim() == 3 and image.shape[0] == 3
    
    shape = image.shape
    image = image.reshape(3, -1)
    ccm = ccm.to(image.device).type_as(image)
    
    image = torch.mm(ccm, image)

    return image.view(shape)

def apply_gains(image, rgb_gain, red_gain, blue_gain, clamp=True):
    """Inverts gains while safely handling saturated pixels."""
    assert image.dim() == 3 and image.shape[0] in [3, 4]
    
    if image.shape[0] == 3:
        gains = torch.tensor([red_gain, 1.0, blue_gain]).to(image.device) * rgb_gain
    else:
        gains = torch.tensor([red_gain, 1.0, 1.0, blue_gain]).to(image.device) * rgb_gain
    gains = gains.view(-1, 1, 1)
    gains = gains.to(image.device).type_as(image)

    if clamp:
        return (image * gains).clamp(0.0, 1.0)
    else:
        return (image * gains)

def safe_invert_gains(image, rgb_gain, red_gain, blue_gain):
    """Inverts gains while safely handling saturated pixels."""
    assert image.dim() == 3 and image.shape[0] == 3

    gains = torch.tensor([1.0 / red_gain, 1.0, 1.0 / blue_gain]) / rgb_gain
    gains = gains.view(-1, 1, 1)

    # Prevents dimming of saturated pixels by smoothly masking gains near white.
    gray = image.mean(dim=0, keepdims=True)
    inflection = 0.9
    mask = ((gray - inflection).clamp(0.0) / (1.0 - inflection)) ** 2.0

    safe_gains = torch.max(mask + (1.0 - mask) * gains, gains)
    return image * safe_gains

def mosaic(image, mode='rggb'):
    """Extracts RGGB Bayer planes from an RGB image."""
    shape = image.shape
    if image.dim() == 3:
        image = image.unsqueeze(0)

    if mode == 'rggb':
        red = image[:, 0, 0::2, 0::2]
        green_red = image[:, 1, 0::2, 1::2]
        green_blue = image[:, 1, 1::2, 0::2]
        blue = image[:, 2, 1::2, 1::2]
        image = torch.stack((red, green_red, green_blue, blue), dim=1)
    elif mode == 'grbg':
        green_red = image[:, 1, 0::2, 0::2]
        red = image[:, 0, 0::2, 1::2]
        blue = image[:, 2, 0::2, 1::2]
        green_blue = image[:, 1, 1::2, 1::2]

        image = torch.stack((green_red, red, blue, green_blue), dim=1)

    if len(shape) == 3:
        return image.view((4, shape[-2] // 2, shape[-1] // 2))
    else:
        return image.view((-1, 4, shape[-2] // 2, shape[-1] // 2))

def random_noise_levels(iso):
    """Generates random noise levels from a log-log linear distribution."""

    iso2shot = lambda x: 9.2857e-07 * x + 8.1006e-05
    shot_value = iso2shot(iso)
    shot_noise = shot_value + torch.normal(mean=0.0, std=5e-05, size=iso_value.size())

    while shot_noise <= 0:
        iso_value = iso2shot(iso)
        shot_noise = iso_value + torch.normal(mean=0.0, std=5e-05, size=iso_value.size())

    log_shot_noise = torch.log(shot_noise)
    logshot2logread = lambda x: 2.2282 * x + 0.45982
    logread_value = logshot2logread(log_shot_noise)
    log_read_noise = logread_value + torch.normal(mean=0.0, std=0.25, size=logread_value.size())
    read_noise = torch.exp(log_read_noise)

    while read_noise <= 0:
        logread_value = logshot2logread(log_shot_noise)
        log_read_noise = logread_value + torch.normal(mean=0.0, std=0.25, size=logread_value.size())
        read_noise = torch.exp(log_read_noise)

    return shot_noise, read_noise

def add_noise(image, shot_noise=0.01, read_noise=0.0005):

    noisy_img = torch.poisson(image / (shot_noise))
    noisy_img = noisy_img * (shot_noise)

    noisy_img = noisy_img + (torch.normal(torch.zeros_like(noisy_img), std=1) * math.sqrt(read_noise))

    return noisy_img

operation_seed_counter = 0

def get_generator():
	global operation_seed_counter
	operation_seed_counter += 1
	g_cpu_generator = torch.Generator(device="cpu")
	g_cpu_generator.manual_seed(operation_seed_counter)
	return g_cpu_generator
