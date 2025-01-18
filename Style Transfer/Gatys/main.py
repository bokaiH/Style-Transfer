import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models

# Load the images and preprocess
def load_image(image_path, max_size=400):
    image = Image.open(image_path).convert('RGB')

    size = min(max_size, max(image.size))

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    return transform(image).unsqueeze(0)

# Extract features using vgg
class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.vgg19(pretrained=True).features[:22]
        for param in self.net.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i in {0, 5, 10, 19, 21}:
                features.append(x)
        return features

# Define the loss functions
def get_content_loss(content_features, target_features):
    loss = 0
    for content_feature, target_feature in zip(content_features, target_features):
        loss += torch.mean((content_feature - target_feature) ** 2)
    return loss

def get_style_loss(style_features, target_features):
    loss = 0
    weights = [1, 0.8, 0.5, 0.3, 0.1]
    def gram_matrix(input):
        batch_size, channels, height, width = input.size()
        features = input.view(batch_size*channels, height*width)
        G = torch.mm(features, features.t())
        return G.div(batch_size*channels*height*width)

    for weight, style_feature, target_feature in zip(weights, style_features, target_features):
        style_gram = gram_matrix(style_feature)
        target_gram = gram_matrix(target_feature)
        loss += weight * torch.mean((style_gram - target_gram)**2)

    return loss

def get_tv_loss(tensor):
    return 0.5 * (torch.abs(tensor[:, :, 1:, :] - tensor[:, :, :-1, :]).mean() +
                  torch.abs(tensor[:, :, :, 1:] - tensor[:, :, :, :-1]).mean())
# Train the model
def train(content_img, style_img, steps, content_weight=1, style_weight=1e9, tv_weight=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    content_img, style_img = content_img.to(device), style_img.to(device)
    target_img = content_img.clone().requires_grad_(True).to(device)

    model = VGG().to(device)

    style_features = model(style_img)
    content_features = model(content_img)

    optimizer = optim.LBFGS([target_img])

    step = 0
    while step <= steps:
        def closure():
            nonlocal step
            optimizer.zero_grad()

            target_features = model(target_img)

            style_loss = get_style_loss(style_features, target_features)
            content_loss = get_content_loss(content_features, target_features)
            tv_loss = get_tv_loss(target_img)

            total_loss = style_weight * style_loss + content_weight * content_loss + tv_weight * tv_loss
            total_loss.backward()

            step += 1
            if step % 50 == 0:
                print(f"Step {step}, Style Loss: {style_loss.item()}, Content Loss: {content_loss.item()}, Total variation loss: {tv_loss.item()}")
            return total_loss
        optimizer.step(closure)


    return target_img
# Save the images
def save_image(tensor, path):
    unnormalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    tensor = tensor.clone().detach().cpu()
    image = unnormalize(tensor.squeeze(0))
    image = transforms.ToPILImage()(image.cpu())
    image.save(path)
    '''
    image = tensor.clone().detach().cpu()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    image.save(path)
    '''
# Main function
if __name__ == "__main__":
    content_num = 3
    style_num = 7
    for i in range(1, content_num + 1):
        for j in range(1, style_num + 1):
            print(f"Processing content{i}.jpg and style{j}.jpg")
            content_path = f"content/content{i}.jpg"
            style_path = f"style/style{j}.jpg"
            content_img = load_image(content_path)
            style_img = load_image(style_path)
            target_img = train(content_img, style_img, 1000)
            save_image(target_img, f"output/output{i}_{j}.jpg")
            print(f"Saved output{i}_{j}.jpg\n")