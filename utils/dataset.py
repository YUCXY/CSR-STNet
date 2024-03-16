from PIL import Image
from torch.utils.data import Dataset



def pil_loader(path):
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


class ColoDataset(Dataset):
    def __init__(self, file_paths, file_labels, transform=None, loader=pil_loader):
        self.file_paths = file_paths
        self.file_labels_phase = file_labels
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_names = self.file_paths[index]
        labels_phase = self.file_labels_phase[index]
        imgs = self.loader(img_names)
        if self.transform is not None:
            imgs = self.transform(imgs)

        return imgs, labels_phase

    def __len__(self):
        return len(self.file_paths)