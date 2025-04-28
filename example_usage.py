import torch
from torchvision import datasets, transforms
from timm.models.vision_transformer import vit_small_patch16_224, vit_base_patch16_224

path_to_data = "imagefolder-20"
path_to_weights = "https://huggingface.co/IverMartinsen/scampi-dino-vits16/resolve/main/vit_small_backbone.pth?download=true"

if __name__ == "__main__":

    # load the model
    model = vit_small_patch16_224(pretrained=False, num_classes=0)
    # load the pretrained weights
    state_dict = torch.hub.load_state_dict_from_url(path_to_weights, map_location="cpu", weights_only=True)
    # load the state dict into the model
    model.load_state_dict(state_dict)
    # set the model to evaluation mode
    model.eval()

    # preprocess the data
    # center crop and normalization will improve the performance
    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    # load the dataset
    dataset = datasets.ImageFolder(path_to_data, transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=128)

    # extract features
    for samples, labels in data_loader:
        features = model(samples).detach().numpy()
        labels = labels.detach().numpy()
        print(f'Extracted {features.shape[0]} features of shape {features.shape[1:]}')
