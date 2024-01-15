import dgl
from dgl.data.utils import save_graphs, load_graphs
from torch.utils.data import Dataset


class TwitterDataset(dgl.data.DGLDataset):
    def __init__(self, num_weeks, stride=1, use_week_count=57):
        self.num_weeks = num_weeks
        self.stride = stride
        self.use_week_count = use_week_count
        super().__init__(name='TwitterDataset')

    def process(self):
        self.graphs = []
        self.unfollow_splits = []

        start_from_week = self.num_weeks - self.use_week_count
        for week in range(start_from_week, self.num_weeks):
            unfollows, _ = load_graphs(f'./data/week_{week + 1}_unfollow_split.bin', list(range(2)))

            self.unfollow_splits.append(unfollows)

        self.graphs, _ = load_graphs('./data/graphs1-58.bin', list(range(self.num_weeks)))
        self.graphs = self.graphs[start_from_week:]

        self.graphs = [dgl.add_self_loop(graph) for graph in self.graphs]
    def __getitem__(self, item):
        return self.graphs[item], self.unfollow_splits[item]

    def __len__(self):
        return len(self.graphs)

class DynamicDataset(Dataset):
    def __init__(self, data, window_size, stride):
        self.data = data
        self.stride = stride
        self.window_size = window_size


    def __getitem__(self, item):

        if isinstance(item, slice):
            x = []
            for index in range(item.start, item.stop):
                x.append(self.data[index:index + self.stride*self.window_size:self.stride])
            return x
        else:
            x = self.data[item:item + self.stride*self.window_size:self.stride]
            return x

    def __len__(self):
        return len(self.data) - self.stride * (self.window_size-1) -1