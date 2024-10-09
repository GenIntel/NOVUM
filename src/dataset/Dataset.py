import os
from torch.utils.data import Dataset
from abc import abstractmethod
from PIL import Image
import numpy as np


class ParentDataset(Dataset):
    
    @property
    @abstractmethod
    def classes(self):
        """Return the class list of the dataset"""
        return []

    def __len__(self):
        return len(self.file_list)
    
    def get_image_size(self):
        name_img = self.file_list[0]
        img = Image.open(os.path.join(self.image_path, name_img))
        return np.array(img).shape[0:2]
    
    def __init__(self, transforms, enable_cache=True, **kwargs):
        self.root_path = kwargs["rootpath"]
        self.img_class = kwargs["imgclass"] if "imgclass" in kwargs else ""
        self.max_n = kwargs["max_n"] if "max_n" in kwargs else 0
        self.for_test = kwargs["for_test"] if "for_test" in kwargs else False
        self.data_pendix = kwargs["data_pendix"] if "data_pendix" in kwargs else ""
        self.weighted = kwargs["weighted"] if "weighted" in kwargs else False
        self.sub_dir = "test" if self.for_test else "train"
        self.corruptions = kwargs["corruptions"] if "corruptions" in kwargs else None
        list_path = kwargs["list_path"] if "list_path" in kwargs else "lists"
        anno_path = kwargs["anno_path"] if "anno_path" in kwargs else "annotations"
        img_path = kwargs["img_path"] if "img_path" in kwargs else "images"
        self.image_path = os.path.join(self.root_path, img_path)
        self.annotation_path = os.path.join(self.root_path, anno_path)
        list_path = os.path.join(self.root_path, list_path)

        self.transforms = transforms
        self.classes = os.listdir(list_path)
        self.label_list = []
        self.file_list = []
        if self.data_pendix == "":
            for class_ in self.classes:
                deeper_dir = os.path.join(list_path, class_)
                self.file_list = [
                    os.path.join(class_, l.strip())
                    for l in open(
                        os.path.join(deeper_dir, "mesh01.txt"),
                    ).readlines()
                ] + self.file_list
            for f_dir in self.file_list:
                img_class = f_dir.split("/")[0]
                label = self.classes.index(img_class)
                self.label_list.append(label)
        else:
            for class_ in self.classes:
                if class_.count(self.data_pendix):
                    deeper_dir = os.path.join(list_path, class_)
                    self.file_list = [
                        os.path.join(class_, l.strip())
                        for l in open(
                            os.path.join(deeper_dir, "mesh01.txt"),
                        ).readlines()
                    ] + self.file_list
                else:
                    continue
            for f_dir in self.file_list:
                img_class = f_dir.split("/")[0].strip(self.data_pendix)
                label = self.classes.index(img_class)
                self.label_list.append(label)

        if "weighted" in kwargs:
            self.weighted = kwargs["weighted"]
        else:
            self.weighted = False

        self.enable_cache = enable_cache
        self.cache_img = dict()
        self.cache_anno = dict()
