# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import wget

BASE_URL = "https://dl.fbaipublicfiles.com/atlas"


import os
import urllib.request
import time
import sys

def download_with_resume(url, target, max_retries=5):
    """Download a file with resume capability"""
    
    os.makedirs(os.path.dirname(target), exist_ok=True)
    temp_file = target + '.partial'
    
    # Get file size if exists
    initial_pos = os.path.getsize(temp_file) if os.path.exists(temp_file) else 0
    
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries}")
            
            # Configure the request with resume capability
            req = urllib.request.Request(url)
            if initial_pos:
                req.add_header('Range', f'bytes={initial_pos}-')
            
            # Open the remote file
            with urllib.request.urlopen(req) as response:
                total_size = int(response.headers.get('content-length', 0)) + initial_pos
                
                # Open our local file
                with open(temp_file, 'ab' if initial_pos else 'wb') as f:
                    downloaded = initial_pos
                    block_size = 8192
                    
                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                            
                        downloaded += len(buffer)
                        f.write(buffer)
                        
                        # Update progress
                        progress = int(50 * downloaded / total_size)
                        sys.stdout.write('\r[%s%s] %d%%' % 
                            ('=' * progress, ' ' * (50-progress), 
                             int(downloaded/total_size*100)))
                        sys.stdout.flush()
            
            # If we get here, the download was successful
            os.rename(temp_file, target)
            print(f"\nDownload completed successfully: {target}")
            return True
            
        except Exception as e:
            print(f"\nError occurred: {str(e)}")
            print("Retrying in 5 seconds...")
            time.sleep(5)
            continue
    
    print(f"Failed to download after {max_retries} attempts")
    return False

def maybe_download_file(source, target):
    if not os.path.exists(target):
        print(f"Downloading {source} to {target}")
        return download_with_resume(source, target)
    return True


def get_s3_path(path):
    return f"{BASE_URL}/{path}"


def get_download_path(output_dir, path):
    return os.path.join(output_dir, path)


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse

AVAILABLE_CORPORA = {
    "corpora/wiki/enwiki-dec2017": {
        "corpus": "corpora/wiki/enwiki-dec2017",
        "description": "Wikipedia dump from Dec 2017, preprocessed into passages",
        "files": ["text-list-100-sec.jsonl", "infobox.jsonl"],
    },
    "corpora/wiki/enwiki-dec2018": {
        "corpus": "corpora/wiki/enwiki-dec2018",
        "description": "Wikipedia dump from Dec 2018, preprocessed into passages",
        "files": ["text-list-100-sec.jsonl", "infobox.jsonl"],
    },
    "corpora/wiki/enwiki-aug2019": {
        "corpus": "corpora/wiki/enwiki-aug2019",
        "description": "Wikipedia dump from Aug 2019, preprocessed into passages",
        "files": ["text-list-100-sec.jsonl", "infobox.jsonl"],
    },
    "corpora/wiki/enwiki-dec2020": {
        "corpus": "corpora/wiki/enwiki-dec2020",
        "description": "Wikipedia dump from Dec 2020, preprocessed into passages",
        "files": ["text-list-100-sec.jsonl", "infobox.jsonl"],
    },
    "corpora/wiki/enwiki-dec2021": {
        "corpus": "corpora/wiki/enwiki-dec2021",
        "description": "Wikipedia dump from Dec 2021, preprocessed into passages",
        "files": ["text-list-100-sec.jsonl", "infobox.jsonl"],
    },
}


def _helpstr():
    helpstr = "The following corpora are available for download: "
    for m in AVAILABLE_CORPORA.values():
        helpstr += f'\nCorpus name: {m["corpus"]:<30} Description: {m["description"]}'
    helpstr += "\ndownload by passing --corpus {corpus name}"
    return helpstr


def main(output_directory, requested_corpus):
    AVAILABLE_CORPORA[requested_corpus]
    for filename in AVAILABLE_CORPORA[requested_corpus]["files"]:
        path = f"{requested_corpus}/{filename}"
        source = get_s3_path(path)
        target = get_download_path(output_directory, path)
        maybe_download_file(source, target)


if __name__ == "__main__":
    help_str = _helpstr()
    choices = list(AVAILABLE_CORPORA.keys())
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--output_directory",
        type=str,
        default="./data",
        help="Path to the file to which the dataset is written.",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        choices=choices,
        help=help_str,
    )
    args = parser.parse_args()
    main(args.output_directory, args.corpus)