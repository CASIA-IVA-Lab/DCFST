import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList


def VOT19Dataset():
    return VOT19DatasetClass().get_sequence_list()


class VOT19DatasetClass(BaseDataset):
    """VOT2018 dataset

    Publication:
        The sixth Visual Object Tracking VOT2018 challenge results.
        Matej Kristan, Ales Leonardis, Jiri Matas, Michael Felsberg, Roman Pfugfelder, Luka Cehovin Zajc, Tomas Vojir,
        Goutam Bhat, Alan Lukezic et al.
        ECCV, 2018
        https://prints.vicos.si/publications/365

    Download the dataset from http://www.votchallenge.net/vot2018/dataset.html"""
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.vot19_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        sequence_path = sequence_name
        nz = 8
        ext = 'jpg'
        start_frame = 1

        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)
        try:
            ground_truth_rect = np.loadtxt(str(anno_path), dtype=np.float64)
        except:
            ground_truth_rect = np.loadtxt(str(anno_path), delimiter=',', dtype=np.float64)

        end_frame = ground_truth_rect.shape[0]

        frames = ['{base_path}/{sequence_path}/color/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
                  sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext)
                  for frame_num in range(start_frame, end_frame+1)]

        # Convert gt
        if ground_truth_rect.shape[1] > 4:
            gt_x_all = ground_truth_rect[:, [0, 2, 4, 6]]
            gt_y_all = ground_truth_rect[:, [1, 3, 5, 7]]
            
            '''
            x1 = np.amin(gt_x_all, 1).reshape(-1,1)
            y1 = np.amin(gt_y_all, 1).reshape(-1,1)
            x2 = np.amax(gt_x_all, 1).reshape(-1,1)
            y2 = np.amax(gt_y_all, 1).reshape(-1,1)

            ground_truth_rect = np.concatenate((x1, y1, x2-x1, y2-y1), 1)
            '''
        return Sequence(sequence_name, frames, ground_truth_rect)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list= ['agility',
                        'book',
                        'zebrafish1',
                        'singer3',
                        'soccer1',
                        'road',
                        'shaking',
                        'tiger',
                        'gymnastics2',
                        'glove',
                        'dinosaur',
                        'helicopter',
                        'marathon',
                        'surfing',
                        'matrix',
                        'polo',
                        'fish1',
                        'crabs1',
                        'flamingo1',
                        'singer2',
                        'gymnastics3',
                        'godfather',
                        'hand2',
                        'handball1',
                        'fish2',
                        'pedestrian1',
                        'fernando',
                        'nature',
                        'dribble',
                        'car1',
                        'rowing',
                        'monkey',
                        'iceskater2',
                        'rabbit',
                        'butterfly',
                        'girl',
                        'graduate',
                        'wiper',
                        'motocross1',
                        'handball2',
                        'basketball',
                        'wheel',
                        'hand',
                        'frisbee',
                        'iceskater1',
                        'birds1',
                        'lamb',
                        'conduction1',
                        'soccer2',
                        'soldier',
                        'bolt1',
                        'ants1',
                        'leaves',
                        'rabbit2',
                        'ball2',
                        'drone_across',
                        'gymnastics1',
                        'drone_flip',
                        'drone1',
                        'ball3']

        return sequence_list
