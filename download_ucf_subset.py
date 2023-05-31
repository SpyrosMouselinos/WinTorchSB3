import argparse
import os
import tarfile
import wget
import os
import patoolib
from huggingface_hub import hf_hub_download


def main(mode='subset'):
    if mode == 'subset':
        hf_dataset_identifier = "sayakpaul/ucf101-subset"
        filename = "UCF101_subset.tar.gz"
        file_path = hf_hub_download(repo_id=hf_dataset_identifier, filename=filename, repo_type="dataset")
        with tarfile.open(file_path) as t:
            t.extractall(".")
    elif mode == 'full':
        ### Get the data ###
        file_path = './UCF101.rar'
        # patoolib.extract_archive(file_path, outdir=".")
        os.remove(file_path)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='full', help='subset/full')
    args = parser.parse_args()
    main(args.mode)
