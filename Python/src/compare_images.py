import torch
import lpips
import os
import random
from PIL import Image
import torchvision.transforms as transforms
from pytorch_msssim import ssim
import statistics
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def get_image_tensor(image):
    # create pipline for converting an image into a PyTorch tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # changes its shape from '[C,H,W]' to '[1,C,H,W]'
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def get_score(tensor1, tensor2):
    # loss function
    loss = lpips.LPIPS(net='alex')

    with torch.no_grad():
        # calculate LPIPS
        perceptual_distance = loss(tensor1, tensor2)
        # calculate SSIM
        ssim_value = ssim(tensor1, tensor2, data_range=1.0, size_average=True)

        # covert to % and return results
        lpips_score = (1 - perceptual_distance.item()) * 100
        ssim_score = ssim_value.item() * 100
        return lpips_score, ssim_score

def print_results(lpips_scores, ssim_scores):
    if len(lpips_scores) > 1:
        # get min, max. mean, and median of the results
        lpips_min, lpips_max, lpips_mean, lpips_median = min(lpips_scores), max(lpips_scores), statistics.mean(lpips_scores), statistics.median(lpips_scores)
        ssim_min, ssim_max, ssim_mean, ssim_median = min(ssim_scores), max(ssim_scores), statistics.mean(ssim_scores), statistics.median(ssim_scores)

        # get an overall accuracy of the two calculations
        accuracy_min = (lpips_min + ssim_min) / 2
        accuracy_max = (lpips_max + ssim_max) / 2
        accuracy_mean = (lpips_mean + ssim_mean) / 2
        accuracy_median = (lpips_median + ssim_median) / 2

        # shows how perceptually diverse the images in the dataset are
        # scores reflect the visual differences perceived by human eyes
        print('LPIPS Scores - Min: {:.2f}%, Mean: {:.2f}%, Median: {:.2f}%, Max: {:.2f}%'.format(lpips_min, lpips_mean, lpips_median, lpips_max))

        # shows the structural alignment of images
        # scores reflect how similar or different an image structure is for texture, brightness, and contrast
        print('SSIM Scores - Min: {:.2f}%, Mean: {:.2f}%, Median: {:.2f}%, Max: {:.2f}%'.format(ssim_min, ssim_mean, ssim_median, ssim_max))

        # scores reflect the average of LPIPS and SSIM
        print('Accuracy Percentage - Min: {:.2f}%, Mean: {:.2f}%, Median: {:.2f}%, Max: {:.2f}%'.format(accuracy_min, accuracy_mean, accuracy_median, accuracy_max))
    else:
        print('LPIPS Score - {:.2f}%'.format(lpips_scores))
        print('SSIM Score - {:.2f}%'.format(ssim_scores))


def plot_iter(obj, output_path, n=3):
    augment_images = []
    fig, ax = plt.subplots(n, n)
    fig.set_size_inches(6, 6)
    for i in range(n):
        for j in range(n):
            batch = obj.next()
            image = batch[0].astype('uint8')
            augment_images.append(image)
            ax[i][j].set_axis_off()
            ax[i][j].imshow(image)
            plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'Augmentations.png'))
    plt.close()
    return augment_images

def augment_compare(real_image_path, output_path):
    # open image and get image tensor
    real_image = Image.open(real_image_path).convert('RGB')
    real_tensor = get_image_tensor(real_image)
    real_array = np.expand_dims(np.array(real_image), axis=0)

    lpips_scores = []
    ssim_scores = []
    datagen = ImageDataGenerator(rotation_range=10, width_shift_range=[-10,10], height_shift_range=[-10, 10], zoom_range=[0.9, 1.1], fill_mode='nearest')
    it = datagen.flow(real_array, batch_size=1)
    augment_images = plot_iter(it, output_path)
    for augment in augment_images:
        augment_image = Image.fromarray(augment)
        augment_tensor = get_image_tensor(augment_image)

        # get scores
        lpips_score, ssim_score = get_score(real_tensor, augment_tensor)
        lpips_scores.append(lpips_score)
        ssim_scores.append(ssim_score)

    # print result
    print_results(lpips_scores, ssim_scores)

def png_to_png_compare(image1_path, image2_path):
    # open image and get image tensor
    image1 = Image.open(image1_path).convert('RGB')
    image2 = Image.open(image2_path).convert('RGB')

    image1_tensor = get_image_tensor(image1)
    image2_tensor = get_image_tensor(image2)

    # get scores
    lpips_score, ssim_score = get_score(image1_tensor, image2_tensor)

    # print result
    print_results(lpips_score, ssim_score)
    return lpips_score, ssim_score

def png_to_numpy_compare(png_path, numpy_image):
    # open image and get image tensor
    image1 = Image.open(png_path).convert('RGB')
    image2 = Image.fromarray(numpy_image)

    image1_tensor = get_image_tensor(image1)
    image2_tensor = get_image_tensor(image2)

    # get scores
    lpips_score, ssim_score = get_score(image1_tensor, image2_tensor)

    # print result
    print_results(lpips_score, ssim_score)
    return lpips_score, ssim_score

def numpy_to_numpy_compare(numpy1, numpy2):
    # open image and get image tensor
    image1 = Image.fromarray(numpy1)
    image2 = Image.fromarray(numpy2)

    image1_tensor = get_image_tensor(image1)
    image2_tensor = get_image_tensor(image2)

    # get scores
    lpips_score, ssim_score = get_score(image1_tensor, image2_tensor)

    # print result
    print_results(lpips_score, ssim_score)
    return lpips_score, ssim_score

def training_compare(training_path, generated_path):
    # open image and get image tensor
    generated_image = Image.open(generated_path).convert('RGB')
    generated_tensor = get_image_tensor(generated_image)

    # get all training images
    training_images = [os.path.join(training_path, f) for f in os.listdir(training_path) if f.endswith('.png')]

    # sample 1/4 of the images to compare the generated image against
    sample_size = len(training_images) // 4
    sampled_images = random.sample(training_images, sample_size)

    # get compare results
    lpips_scores = []
    ssim_scores = []

    count = 1
    for image_path in sampled_images:
        # for each sampled image
        print(f"Comparing Sample Image {count}/{sample_size}")
        count += 1
        # open train sample image
        train_img = Image.open(image_path).convert('RGB')
        # convert to pytorch tensor
        train_tensor = get_image_tensor(train_img)
        # get scores
        lpips_score, ssim_score = get_score(generated_tensor, train_tensor)
        lpips_scores.append(lpips_score)
        ssim_scores.append(ssim_score)
    
    # print results
    print_results(lpips_scores, ssim_scores)


if __name__ == '__main__':
    training_path = 'data/128x128/'
    generated_path = 'data/128x128/00000.png'
    real_path = 'data/sample.png'
    output_path = 'output/'

    #training_compare(training_path, generated_path)

    augment_compare(real_path, output_path)

