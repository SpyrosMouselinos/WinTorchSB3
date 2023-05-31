import argparse
import os
import tarfile
import wget
import os
import patoolib
import shutil
from huggingface_hub import hf_hub_download


def main(mode='subset'):
    if mode == 'subset':
        hf_dataset_identifier = "sayakpaul/ucf101-subset"
        filename = "UCF101_subset.tar.gz"
        file_path = hf_hub_download(repo_id=hf_dataset_identifier, filename=filename, repo_type="dataset")
        with tarfile.open(file_path) as t:
            t.extractall(".")
    elif mode == 'full':
        ## Get the data ###
        wget.download('https://www.crcv.ucf.edu/data/UCF101/UCF101.rar')
        file_path = './UCF101.rar'
        patoolib.extract_archive(file_path, outdir=".")
        os.remove(file_path)

        # Fix folders #
        if not os.path.exists('./UCF101_full'):
            os.mkdir('./UCF101_full')
            os.mkdir('./UCF101_full/val')
            for f in os.listdir('./UCF-101/'):
                os.mkdir('./UCF101_full/val/' + f)
            with open('ufc_testlist.txt', 'r') as fin:
                test_files = fin.readlines()

            for filepath in test_files:
                filepath = filepath.strip()
                shutil.copy('./UCF-101/' + filepath, './UCF101_full/val/' + filepath)
            shutil.move('./UCF-101/', './UCF101_full/train/')
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='subset', help='subset/full')
    args = parser.parse_args()
    main(args.mode)
