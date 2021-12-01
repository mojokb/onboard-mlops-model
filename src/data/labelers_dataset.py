from multiprocessing import freeze_support
from eimmo_sdk_inhouse.eimmo import initialize, Phase
from eimmo_sdk_inhouse.eimmo import get_dataset
from eimmo_sdk_inhouse.eimmo import download_files
from eimmo_sdk_inhouse.eimmo import iter_gt

initialize(Phase.inhouse_dev, query_log=False)
project_id = "618a23969eab02839d20f3ae"
dataset = get_dataset(project_id)

iter_gt = iter_gt(dataset.id, done_stage_id=None, from_date=None,
                  to_date=None, parent_path="None", batch_size=100,
                  recursive=False)

file_paths = ((gt['parent_path'], gt['filename']) for gt in iter_gt)

CLASSES = ['Shirt', 'Bag', 'Trouser', 'Dress', 'Sandal', 'Ankelboot', 'Coat', 'Sneaker', 'Tshirt', 'Pullover']

from eimmo_sdk_inhouse.eimmo import get_dataset
print(get_dataset(project_id))

from eimmo_sdk_inhouse.eimmo import get_directories_file_count
dataset_dir = get_directories_file_count(dataset.id)


if __name__ == '__main__':
    freeze_support()

    download_files(project_id=project_id, file_paths=file_paths, output_root="~/download",
                   process_count=2, overwrite=True)

    train_f = open("~/download/train/labels.txt", "w")
    test_f = open("~/download/test/labels.txt", "w")

    # fit filename class_id
    # ex) fashion00522.jpg 8
    for gt in iter_gt:
        class_name = gt['annotations'][0]['label']
        if gt['parent_path'] == '/train':
            train_f.write(f"{gt['filename']} {CLASSES.index(class_name)}\n")
        else:
            test_f.write(f"{gt['filename']} {CLASSES.index(class_name)}\n")
    