from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):

    def returnclasses(self):
        return 3

    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')
        
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, split)
        self.targets = [s[1] for s in samples]

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''
        
    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:

        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        
        classes = [d.name for d in os.scandir(dir) if d.is_dir() and d.name != "BACKGROUND_Google"]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


    def __getitem__(self, index):
      
        img, target = self.samples[index]

        return img, target
    
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        splits) -> List[Tuple[str, int]]:
        instances = list()
        directory = os.path.expanduser(directory)
        name = '{0}.txt'.format(splits)

        with open(name, 'r') as f:
          for line in f:
            classname = line.split('/')[0]
            img = pil_loader(line)
            instances.append((img, class_to_idx.get(index)))





        return instances

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.samples) # Provide a way to get the length (number of elements) of the dataset
        return length
