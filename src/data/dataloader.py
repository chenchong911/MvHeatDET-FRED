import torch 
import torch.utils.data as data
from torch.utils.data import Dataset

from src.core import register


__all__ = ['DataLoader']


@register
class DataLoader(data.DataLoader):
    __inject__ = ['dataset', 'collate_fn']

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for n in ['dataset', 'batch_size', 'num_workers', 'drop_last', 'collate_fn']:
            format_string += "\n"
            format_string += "    {0}: {1}".format(n, getattr(self, n))
        format_string += "\n)"
        return format_string


@register
class DatasetSubset(Dataset):
    __inject__ = ['dataset']

    def __init__(self, dataset, ratio=1.0, max_samples=None, seed=42, shuffle=True):
        self.dataset = dataset
        dataset_len = len(dataset)

        target_len = dataset_len
        if ratio is not None:
            target_len = min(target_len, max(1, int(dataset_len * float(ratio))))

        if max_samples is not None:
            target_len = min(target_len, int(max_samples))

        if target_len >= dataset_len:
            self.indices = list(range(dataset_len))
        else:
            generator = torch.Generator()
            generator.manual_seed(int(seed))
            if shuffle:
                self.indices = torch.randperm(dataset_len, generator=generator)[:target_len].tolist()
            else:
                self.indices = list(range(target_len))

        print(
            f"DatasetSubset initialized: {len(self.indices)}/{dataset_len} samples "
            f"(ratio={ratio}, max_samples={max_samples}, shuffle={shuffle})"
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


@register
class SequenceDatasetSubset(Dataset):
    __inject__ = ['dataset']

    def __init__(self, dataset, ratio=1.0, max_sequences=None, seed=42, shuffle=True, seq_field='seq_name'):
        self.dataset = dataset
        self.seq_field = seq_field

        if not hasattr(dataset, 'ids') or not hasattr(dataset, 'coco'):
            raise ValueError('SequenceDatasetSubset requires a COCO-style dataset with `ids` and `coco` attributes.')

        sequence_to_indices = {}
        for dataset_idx, image_id in enumerate(dataset.ids):
            img_info = dataset.coco.loadImgs(image_id)[0]
            seq_name = img_info.get(seq_field)
            if seq_name is None:
                file_name = img_info.get('file_name', '')
                parts = file_name.split('/')
                seq_name = parts[1] if len(parts) > 1 else 'unknown'

            sequence_to_indices.setdefault(seq_name, []).append(dataset_idx)

        sequence_names = sorted(sequence_to_indices.keys())
        total_sequences = len(sequence_names)

        target_sequences = total_sequences
        if ratio is not None:
            target_sequences = min(target_sequences, max(1, int(total_sequences * float(ratio))))

        if max_sequences is not None:
            target_sequences = min(target_sequences, int(max_sequences))

        if target_sequences >= total_sequences:
            selected_sequences = sequence_names
        else:
            generator = torch.Generator()
            generator.manual_seed(int(seed))
            if shuffle:
                perm = torch.randperm(total_sequences, generator=generator)[:target_sequences].tolist()
                selected_sequences = [sequence_names[i] for i in perm]
            else:
                selected_sequences = sequence_names[:target_sequences]

        self.selected_sequences = sorted(selected_sequences)
        self.indices = []
        for seq_name in self.selected_sequences:
            self.indices.extend(sequence_to_indices[seq_name])

        print(
            f"SequenceDatasetSubset initialized: {len(self.selected_sequences)}/{total_sequences} sequences, "
            f"{len(self.indices)}/{len(dataset)} samples "
            f"(ratio={ratio}, max_sequences={max_sequences}, shuffle={shuffle})"
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]



@register
def default_collate_fn(items):
    '''default collate_fn
    '''    
    return torch.cat([x[0][None] for x in items], dim=0), [x[1] for x in items]

@register
def eso_collate_fn(batch):
    imgs, density, targets = zip(*batch)  # 解压批次数据

    # 将imgs、density和targets转换为适当的tensor
    imgs = torch.stack(imgs,dim=0)  # 假设 imgs 是torch.Tensor
    density = torch.stack(density, dim=0)  # 假设 density 是torch.Tensor
    # targets = torch.stack(targets)  # 假设 targets 是torch.Tensor

    return imgs, density, targets
