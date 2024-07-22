from multiprocessing import Pool
from batchgenerators.utilities.file_and_folder_operations import *


def parse_dataset_trainer_plans_configuration_from_path(path: str):
    folders = split_path(path)
    fold_x_present = [i.startswith('fold_') for i in folders]
    if any(fold_x_present):
        idx = fold_x_present.index(True)
        assert len(folders[:idx]) >= 2, 'Bad path, cannot extract what I need. Your path needs to be at least ' \
                                        'DatasetXXX/MODULE__PLANS__CONFIGURATION for this to work'
        if folders[idx - 2].startswith('Dataset'):
            splitted = folders[idx - 1].split('__')
            assert len(splitted) == 3, 'Bad path, cannot extract what I need. Your path needs to be at least ' \
                                        'DatasetXXX/MODULE__PLANS__CONFIGURATION for this to work'
            return folders[idx - 2], *splitted
    else:
        dataset_folder = [i.startswith('Dataset') for i in folders]
        if any(dataset_folder):
            idx = dataset_folder.index(True)
            assert len(folders) >= (idx + 1), 'Bad path, cannot extract what I need. Your path needs to be at least ' \
                                        'DatasetXXX/MODULE__PLANS__CONFIGURATION for this to work'
            splitted = folders[idx + 1].split('__')
            assert len(splitted) == 3, 'Bad path, cannot extract what I need. Your path needs to be at least ' \
                                       'DatasetXXX/MODULE__PLANS__CONFIGURATION for this to work'
            return folders[idx], *splitted


def check_workers_alive_and_busy(export_pool: Pool, worker_list: List, results_list: List, allowed_num_queued: int = 0):
    """

    returns True if the number of results that are not ready is greater than the number of available workers + allowed_num_queued
    """
    alive = [i.is_alive() for i in worker_list]
    if not all(alive):
        raise RuntimeError('Some background workers are no longer alive')

    not_ready = [not i.ready() for i in results_list]
    if sum(not_ready) >= (len(export_pool._pool) + allowed_num_queued):
        return True
    return False

