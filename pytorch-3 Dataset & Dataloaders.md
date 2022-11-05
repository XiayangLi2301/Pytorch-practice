# pytorch-3 Dataset & Dataloaders

### Dataset 函数

构建自己的数据集需要继承官方的Dataset的类。自定义的函数中必须要实现三个函数：\_\_init\_\_,\_\_len\_\_,\_\_getitem\_\_

```python
    def __getitem__(self, index) -> T_co:
        raise NotImplementedError
```

在 Dataset源码中, \*  \***getitem**函数并没有实现

getitem 基于索引来返回一个训练对 $(x,y)$。如果训练集中有10000个样本，那么index的取值范围就是 0到9999。

下面看一下官网实现的例子：

```python
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        ## init 函数需要传入 annotations_file 也就是文件名， img_dirs 是图片的路径名。
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        ## img_path记录了照片的地址
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        ## 比如对数据进行预处理
        return image, label
        ## 根据指定的索引返回样本
```

在官网中写了, The labels.csv files looks like.

### Dataloader 函数

dataset 函数每次只处理一个样本, 但是 Dataloader 函数每次处理多个样本。

Dataloader 函数可以直接调用。

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
### test_dataloader 一般不需要设置 shuffle=True
```

DataLoader 函数的源码:

```python
def __init__(self, dataset: Dataset[T_co], batch_size: Optional[int] = 1,
                 shuffle: bool = False, sampler: Union[Sampler, Iterable, None] = None,
                 batch_sampler: Union[Sampler[Sequence], Iterable[Sequence], None] = None,
                 num_workers: int = 0, collate_fn: Optional[_collate_fn_t] = None,
                 ## collate_fn 函数对一个批次的样本进行处理。
                 pin_memory: bool = False, drop_last: bool = False,
                 timeout: float = 0, worker_init_fn: Optional[_worker_init_fn_t] = None,
                 multiprocessing_context=None, generator=None,
                 *, prefetch_factor: int = 2,
                 persistent_workers: bool = False):
"""
Dataset 传入数据集, batch_size 设置批次大小。Optionat[int] 要不要把数据集打乱, Sampler与 batch 
sampler 是采样的方式。num_workers: 设置多线程的数目,取决于 CPU的个数。pin_memory*
drop_last 表示把最后一个小批次丢掉，适用于不是数据集不是 batch_size 整数倍的情况.

"""
```

\*关于pin\_memory的详解, 可看 [文章](https://zhuanlan.zhihu.com/p/477870660 "文章")

然后就是利用 Data\_loader 喂给模型样本

```python
# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
```
