# -*- coding: utf-8 -*-
import click
import logging
import os
import cv2
import numpy as np
import random
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('train_filepath', type=click.Path(exists=True))
@click.argument('label_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('transform')
def main(train_filepath, label_filepath, output_filepath, transform=False):
    """ idx 타입의 파일을 raw image(png)로 변환
    """
    
    for f in [train_filepath, label_filepath]:
        os.system('gunzip %s.gz' % (f,))   
    
    logger = logging.getLogger(__name__)
    logger.info('making raw images from idx file')
    
    with open(train_filepath, 'rb') as f:
        images = f.read()
    with open(label_filepath, 'rb') as f:
        labels = f.read()
    images = [d for d in images[16:]]
    images = np.array(images, dtype=np.uint8)
    images = images.reshape((-1, 28, 28))
    images = images
    if not os.path.exists(output_filepath):
        os.mkdir(output_filepath)
    for k, image in enumerate(images):
        if transform:
            variation = random.randint(-10, 10)
            if variation % 4 == 1:
                image = move_image(image)
            image = rotate_image(image, variation)
        cv2.imwrite(os.path.join(output_filepath, '%05d.png' % (k,)), image)

    labels = ['%05d.png %d' % (k, l) for k, l in enumerate(labels[8:])]
    labels = labels
    with open(os.path.join(output_filepath, 'labels.txt'), 'w') as f:
        f.write(os.linesep.join(labels))
    total += len(images)    
    
def rotate_image(img, angle):
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def move_image(img):
    move_amount = random.randint(-10, 10)
    m = [[1, 0, move_amount], [0, 1, move_amount]]
    h, w = img.shape[:2]
    m = np.float32(m)
    return cv2.warpAffine(img, m, (w, h))    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
    '''
    python save_raw_from_idx.py /data/FMNIST/raw/train-images-idx3-ubyte.gz /data/FMNIST/raw/train-labels-idx3-ubyte.gz /data/raw/train, False
    '''
