from glob import glob
import os.path as osp
import subprocess
from pathlib import Path
from tqdm import tqdm
import argparse

BUFFER_SIZE_TRAIN = 64
CLIP_LIMIT_TRAIN = 32

BUFFER_SIZE_VAL = 64 * 4
CLIP_LIMIT_VAL = 32 * 4


def main(cas_path, dest_path):
    for mode in ['val']:
        neg_frames = sorted(glob(osp.join(cas_path, mode, '0', '*.jpg')))
        pos_frames = sorted(glob(osp.join(cas_path, mode, '1', '*.jpg')))

        neg_split = make_splits(neg_frames)
        pos_split = make_splits(pos_frames)

        if mode == 'train':
            gen_annotation(neg_split, pos_split, 'mma_train_list.txt')
        if mode == 'val':
            gen_annotation(neg_split, pos_split, 'mma_val_list.txt')

        copy_frames(neg_split, dest_path)
        copy_frames(pos_split, dest_path)


def gen_annotation(neg_split, pos_split, dest_file_name):
    with open(dest_file_name, 'w') as f:
        for idx, split in enumerate([neg_split, pos_split]):
            for clip_name, frames in split.items():
                f.write(clip_name + ' ' + str(len(frames)) + ' ' + str(idx) + '\n')


def copy_frames(split, dest_dir):
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    for clip_idx, frames in tqdm(split.items()):
        dir_name = osp.join(dest_dir, clip_idx)
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        for idx, frame in enumerate(frames):
            file_name = frame.split('/')[-1]
            mma_name = 'img_' + str(idx).zfill(5) + '.jpg'
            subprocess.call(['cp', frame, osp.join(dir_name, mma_name)])


def make_splits(frames):
    splits = {}
    frame_buffer = []
    clip_counter = 0

    prev_video_id = None
    prev_part_id = None
    prev_frame_id = None

    for frame in frames:
        # print(frame)
        filename = frame.split('/')[-1]
        unique_frame_id = filename.split('.')[0]
        assert len(unique_frame_id) == 10, "The frame id is not 10 in length!"

        video_id = unique_frame_id[:2]
        part_id = unique_frame_id[2:4]
        frame_id = unique_frame_id[4:4+6]
        # print(filename)
        # print(video_id, part_id, frame_id)

        if prev_video_id is None:
            prev_video_id = video_id
        if prev_part_id is None:
            prev_part_id = part_id
        if prev_frame_id is None:
            prev_frame_id = frame_id

        if video_id == prev_video_id and part_id == prev_part_id and (int(frame_id) - int(prev_frame_id)) < 5:
            # print('case 1')
            frame_buffer.append(frame)
            if len(frame_buffer) == BUFFER_SIZE_TRAIN:
                splits[video_id+part_id+'_'+str(clip_counter)] = frame_buffer
                clip_counter += 1
                frame_buffer = []
        elif len(frame_buffer) >= BUFFER_SIZE_TRAIN:
            # print('case 2')
            splits[video_id + part_id + '_' + str(clip_counter)] = frame_buffer
            clip_counter += 1
            frame_buffer = []
        else:
            # print('case 3')
            frame_buffer = []

        prev_video_id = video_id
        prev_part_id = part_id
        prev_frame_id = frame_id

    return splits


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='get the source and destination path')
    parser.add_argument('--cas_path', type=str, help='path to cas dataset')
    parser.add_argument('--dest_path', type=str, help='path to destination dataset')

    args = parser.parse_args()

    main(args.cas_path, args.dest_path)

