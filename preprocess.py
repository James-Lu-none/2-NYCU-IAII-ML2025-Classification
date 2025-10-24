import os
import torchvision.transforms.v2 as T
from PIL import Image
import torch
from models import *

# Custom transform to add Gaussian noise
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

# Custom transform to add Speckle noise
class AddSpeckleNoise(object):
    """
    Add speckle noise to the image.
    """
    def __init__(self, noise_level=0.1):
        """
        :param noise_level: Standard deviation of the noise distribution
        """
        self.noise_level = noise_level

    def __call__(self, tensor):
        """
        :param tensor: PyTorch tensor, the image on which noise is added
        :return: PyTorch tensor, image with speckle noise
        """
        # Generate speckle noise
        noise = torch.randn_like(tensor) * self.noise_level

        # Add speckle noise to the image
        noisy_tensor = tensor * (1 + noise)

        # Clip the values to be between 0 and 1
        noisy_tensor = torch.clamp(noisy_tensor, 0, 1)

        return noisy_tensor

class AddPoissonNoise(object):
    """
    Add Poisson noise to the image.
    """
    def __init__(self, lam=1.0):
        """
        :param lam: Lambda parameter for Poisson distribution
        """
        self.lam = lam

    def __call__(self, tensor):
        """
        :param tensor: PyTorch tensor, the image to which noise is added
        :return: PyTorch tensor, image with Poisson noise
        """
        # Generate Poisson noise
        noise = torch.poisson(self.lam * torch.ones(tensor.shape))

        # Add Poisson noise to the image
        noisy_tensor = tensor + noise / 255.0  # Assuming the image is scaled between 0 and 1

        # Clip the values to be between 0 and 1
        noisy_tensor = torch.clamp(noisy_tensor, 0, 1)

        return noisy_tensor

# Custom transform to add Salt and Pepper noise
class AddSaltPepperNoise(object):
    def __init__(self, salt_prob=0.05, pepper_prob=0.05):
        self.salt_prob = salt_prob
        self.pepper_prob = pepper_prob

    def __call__(self, tensor):
        tensor = tensor.clone()
        noise = torch.rand(tensor.size())
        tensor[noise < self.salt_prob] = 1  # Salt noise: setting some pixels to 1
        tensor[noise > 1 - self.pepper_prob] = 0  # Pepper noise: setting some pixels to 0
        return tensor

def augment_data(src_folder, dst_folder, transform, n_per_class=10000, transform_per_image=4):
    for c in os.listdir(src_folder):
        i = 0
        src_class_folder = os.path.join(src_folder, c)
        dst_class_folder = os.path.join(dst_folder, c)
        os.makedirs(dst_class_folder, exist_ok=True)
        print(f"{len(os.listdir(src_class_folder))} files in class: {c}")
        for f in os.listdir(src_class_folder):
            if i >= n_per_class:
                break
            src_path = os.path.join(src_class_folder, f)
            img = Image.open(src_path).convert("RGB")
            aug_imgs = []
            for k in range(transform_per_image):
                aug_img = transform(img)
                aug_imgs.append(aug_img)

            # before_path = os.path.join(dst_class_folder, f"{i}.jpg")
            # img.save(before_path)
            for j, aug_img in enumerate(aug_imgs):
                dst_path = os.path.join(dst_class_folder, f"{i}_{j}.jpg")
                aug_img.save(dst_path)
            i += 1
        print(f"{i*transform_per_image} file generated")
        
transform = T.Compose([
    T.ToTensor(),  # Convert PIL image to tensor

    T.RandomApply([T.RandomHorizontalFlip()], p=0.1),
    T.RandomApply([T.RandomVerticalFlip()], p=0.1),
    T.RandomApply([T.RandomRotation(10)], p=0.1),

    T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.1),
    T.RandomGrayscale(p=0.1),
    T.RandomInvert(p=0.1),
    T.RandomPosterize(bits=2, p=0.1),
    T.RandomApply([T.RandomSolarize(threshold=1.0)], p=0.05),
    T.RandomApply([T.RandomAdjustSharpness(sharpness_factor=2)], p=0.1),

    T.RandomApply([AddGaussianNoise(0., 0.05)], p=0.1),  # mean and std
    T.RandomApply([AddPoissonNoise(lam=0.1)], p=0.1),  # mean and std
    T.RandomApply([AddSpeckleNoise(noise_level=0.1)], p=0.1),
    T.RandomApply([AddSaltPepperNoise(salt_prob=0.05, pepper_prob=0.05)], p=0.1),

    T.RandomApply([T.RandomPerspective(distortion_scale=0.6, p=1.0)], p=0.1),
    T.RandomApply([T.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))], p=0.1),
    T.RandomApply([T.ElasticTransform(alpha=250.0)], p=0.1),

    T.RandomApply([T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))], p=0.1),

    T.RandomApply([AddGaussianNoise(0., 0.001)], p=1.0),  # mean and std
    T.ToPILImage()  # Convert tensor back to PIL image for saving
])

elastic_transform = T.Compose([
    T.ToTensor(),  # Convert PIL image to tensor

    T.RandomApply([T.RandomHorizontalFlip()], p=0.1),
    T.RandomApply([T.RandomVerticalFlip()], p=0.1),
    T.RandomApply([T.RandomRotation(10)], p=0.1),

    T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.1),
    T.RandomGrayscale(p=0.1),
    T.RandomInvert(p=0.1),
    T.RandomPosterize(bits=2, p=0.1),
    T.RandomApply([T.RandomSolarize(threshold=1.0)], p=0.05),
    T.RandomApply([T.RandomAdjustSharpness(sharpness_factor=2)], p=0.1),

    T.RandomApply([AddGaussianNoise(0., 0.05)], p=0.1),  # mean and std
    T.RandomApply([AddPoissonNoise(lam=0.1)], p=0.1),  # mean and std
    T.RandomApply([AddSpeckleNoise(noise_level=0.1)], p=0.1),
    T.RandomApply([AddSaltPepperNoise(salt_prob=0.05, pepper_prob=0.05)], p=0.1),

    T.RandomApply([T.RandomPerspective(distortion_scale=0.6, p=1.0)], p=0.1),
    T.RandomApply([T.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))], p=0.1),
    T.ElasticTransform(alpha=250.0),

    T.RandomApply([T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))], p=0.1),

    T.RandomApply([AddGaussianNoise(0., 0.001)], p=1.0),  # mean and std
    T.ToPILImage()  # Convert tensor back to PIL image for saving
])

src_folder = "./data/train"
dst_folder = "./data/pre"
elastic_folder = "./data/elastic"
if not os.path.exists(dst_folder):
    os.makedirs(dst_folder)

if not os.path.exists(elastic_folder):
    os.makedirs(elastic_folder)

# augment_data(src_folder, dst_folder, transform, n_per_class=10000, transform_per_image=4)
augment_data(src_folder, elastic_folder, elastic_transform, n_per_class=10000, transform_per_image=1)

print("Image augmentation completed.")
