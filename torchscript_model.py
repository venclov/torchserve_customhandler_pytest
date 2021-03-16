import torch
from torchvision.models import resnet34


def main():
    model = resnet34(pretrained=True)
    model.eval()
    traced_model = torch.jit.script(model)
    traced_model.save("./resnet34.pt")


if __name__ == '__main__':
    main()
