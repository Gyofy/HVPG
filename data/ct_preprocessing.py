""" os """
import os

"""glob"""
import glob

""" tqdm """
from tqdm import tqdm


"""numpy"""
import numpy as np

"""natsort"""
from natsort import natsorted

"""Medical Image"""
import SimpleITK as sitk



"""Function"""
def rgb2gray(img):
    # img [B, C, H, W]
    return  ((0.3 * img[:,0]) + (0.59 * img[:,1]) + (0.11 * img[:,2]))[:, None]
    # return  ((0.33 * img[:,0]) + (0.33 * img[:,1]) + (0.33 * img[:,2]))[:, None]

def normalize(img):
    # img =  torch.stack([b.clamp(torch.quantile(b, 0.001), torch.quantile(b, 0.999)) for b in img])
    return torch.stack([(b-b.min())/(b.max()-b.min()) for b in img])

def win_scale(data, wl, ww, dtype, out_range):
    """
    Scale pixel intensity data using specified window level, width, and intensity range.
    """
    
    data_new = np.empty(data.shape, dtype=np.double)
    data_new.fill(out_range[1]-1)
    
    data_new[data <= (wl-ww/2.0)] = out_range[0]
    
    data_new[(data>(wl-ww/2.0))&(data<=(wl+ww/2.0))] = ((data[(data>(wl-ww/2.0))&(data<=(wl+ww/2.0))]-(wl-0.5))/(ww-1.0)+0.5)*(out_range[1]-out_range[0])+out_range[0]
    
    data_new[data > (wl+ww/2.0)] = out_range[1]-1
    
    return data_new.astype(dtype)


def winset_type(data, winset_name, dytpe, out_range):

    if winset_name == 'Bone':

        wl, ww = 400, 1800

    elif winset_name == 'Liver':

        wl, ww = 50, 200

    elif winset_name == 'Mediastinum':

        wl, ww = 50, 350
    
    win_data = win_scale(data, wl, ww, dytpe, out_range)

    return win_data
    
def resampling_image(image, new_spacing = [1, 1, 1]):
   
    # 리사이즈를 위한 리샘플러 설정
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetInterpolator(sitk.sitkLinear)


    # 이미지의 원래 spacing 및 size 계산
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [int(round(osz*ospc/nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]
    resampler.SetSize(new_size)

    # 리샘플링된 이미지 생성
    resampled_image = resampler.Execute(image)

    return resampled_image


def resize_image(image, new_size=(256, 256, 256)):
    # 현재 이미지의 크기와 복셀 크기를 얻기
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    
    # 새로운 복셀 크기 계산
    new_spacing = [osz*osp/nsp for osz, osp, nsp in zip(original_size, original_spacing, new_size)]

    # 리사이즈를 위한 리샘플러 설정
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetInterpolator(sitk.sitkLinear)

    # 이미지 리샘플링 실행
    resized_image = resampler.Execute(image)

    return resized_image

def final_spacing(resampling_image, resize_image):

    resampling_image = sitk.GetArrayFromImage(resampling_image)
    resize_image = sitk.GetArrayFromImage(resize_image)

    fn_spacing = (resampling_image.shape[0]/resize_image.shape[0], resampling_image.shape[1]/resize_image.shape[1], resampling_image.shape[1]/resize_image.shape[1])

    return fn_spacing


"""Main"""

#Load nii.gz file
data_list = natsorted(glob.glob('../data/*.nii.gz'))

#Process image

for i in tqdm(range(len(data_list))):

    image = sitk.ReadImage(data_list[i])

    image_array = sitk.GetArrayFromImage(image)

    win_array = winset_type(image_array, 'Mediastinum', np.uint8, (0,255))

    win_image = sitk.GetImageFromArray(win_array)

    resam_image = resampling_image(win_image)

    resized_image = resize_image(resam_image, new_size=(256, 256, 96))

    resized_array = sitk.GetArrayFromImage(resized_image)

    sitk.WriteImage(resized_image, '../data/saved_data/' + data_list[i].split('\\')[-1])