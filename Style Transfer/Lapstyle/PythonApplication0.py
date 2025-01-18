import paddle
import paddle.nn as nn
import paddle.vision.transforms as transforms
from PIL import Image
import numpy as np
import os
from skimage import exposure
def load_image(image_path, max_size=400, batch_size=1):
    try:
        print(f"Loading image from {image_path}...")
        image = Image.open(image_path).convert('RGB')
        size = min(max_size, max(image.size))
        transform = transforms.Compose([
            transforms.Resize(size, interpolation='bilinear'),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        transformed_image = transform(image).unsqueeze(0)  
        transformed_image = transformed_image.repeat_interleave(batch_size, axis=0)
        print(f"Image loaded and transformed: {transformed_image.shape}")
        return transformed_image
    except Exception as e:
        print(f"Error loading image: {e}")
        print(f"Image path: {image_path}")
        raise

def build_laplacian_pyramid(image, levels):
    print(f"Building Laplacian pyramid with {levels} levels...")
    pyramid = [image]
    for level in range(levels):
        image = nn.functional.interpolate(image, scale_factor=0.5, mode='bilinear', align_corners=False)
        pyramid.append(image)
        print(f"Level {level + 1} added to pyramid: {image.shape}")
    return pyramid

def reconstruct_from_pyramid(pyramid):
    print("Reconstructing image from pyramid...")
    image = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        image = nn.functional.interpolate(image, scale_factor=2, mode='bilinear', align_corners=False)
        if image.shape != pyramid[i].shape:
            image = nn.functional.interpolate(image, size=pyramid[i].shape[2:], mode='bilinear', align_corners=False)
        image += pyramid[i]
        print(f"Reconstructed level {i + 1}: {image.shape}")
    return image

class LapStyleModel(nn.Layer):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2D(6, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2D(64),
            nn.ReLU(),
            nn.Conv2D(64, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2D(128),
            nn.ReLU(),
            nn.Conv2D(128, 256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2D(256),
            nn.ReLU(),
            nn.Conv2D(256, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2D(128),
            nn.ReLU(),
            nn.Conv2D(128, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2D(64),
            nn.ReLU(),
            nn.Conv2D(64, 3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x, style, content_weight=1.0, style_weight=0.5):
        x = paddle.concat([x, style], axis=1)
        output = self.model(x)
        combined = content_weight * x[:, :3, :, :] + style_weight * output
        combined = paddle.clip(combined, 0, 1)
        return combined
def match_histograms(source, reference, blend_ratio=0.8):
    print("Matching histograms...")
    source = source.cpu().numpy().transpose(0, 2, 3, 1)
    reference = reference.cpu().numpy().transpose(0, 2, 3, 1)
    matched = np.empty_like(source)
    for i in range(source.shape[0]):
        matched[i] = exposure.match_histograms(source[i], reference[i], channel_axis=-1)
    matched = paddle.to_tensor(matched.transpose(0, 3, 1, 2))
    blended = blend_ratio * matched + (1 - blend_ratio) * paddle.to_tensor(source.transpose(0, 3, 1, 2))
    return blended
def lapstyle_transfer(content_img, style_img, style_model, levels=3, iterations=500, content_weight=1.0, style_weight=0.5):
    print("Starting Laplacian style transfer...")
    content_pyramid = build_laplacian_pyramid(content_img, levels)
    style_pyramid = build_laplacian_pyramid(style_img, levels)
    stylized_pyramid = []
    for level, (content_level, style_level) in enumerate(zip(content_pyramid, style_pyramid)):
        if content_level.shape != style_level.shape:
            style_level = nn.functional.interpolate(style_level, size=content_level.shape[2:], mode='bilinear', align_corners=False)
        print(f"Processing level {level + 1}...")
        for iteration in range(iterations):
            content_level = style_model(content_level, style_level, content_weight, style_weight)
            content_level = paddle.clip(content_level, 0, 1)
            if iteration % 100 == 0:
                print(f"Iteration {iteration}/{iterations} at level {level + 1}")
        stylized_pyramid.append(content_level)
    stylized_img = reconstruct_from_pyramid(stylized_pyramid)
    return match_histograms(stylized_img, style_img)

def save_image(tensor, path):
    print(f"Saving image to {path}...")
    unnormalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    tensor = tensor.clone().detach().cpu()
    image = unnormalize(tensor.squeeze(0))
    image = image.numpy().transpose(1, 2, 0)
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    image.save(path)
    print("Image saved.")

def main():
    content_img_path = './content/content3.jpg'
    style_img_path = './style/style3.jpg'
    output_path = './output/output12.jpg'
    levels = 3
    iterations = 10
    batch_size = 1
    content_weight = 1.0
    style_weight = 0.02
    blend_ratio=0.1

    content_img = load_image(content_img_path, batch_size=batch_size)
    style_img = load_image(style_img_path, batch_size=batch_size)
    style_model = LapStyleModel().to("gpu" if paddle.is_compiled_with_cuda() else "cpu")
    content_img = content_img.to("gpu" if paddle.is_compiled_with_cuda() else "cpu")
    style_img = style_img.to("gpu" if paddle.is_compiled_with_cuda() else "cpu")
    target_img = lapstyle_transfer(content_img, style_img, style_model, levels, iterations, content_weight, style_weight)
    save_image(target_img, output_path)
    print(f"Saved output image to {output_path}")

if __name__ == "__main__":
    main()