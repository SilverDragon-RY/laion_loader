from PIL import Image
from tqdm import tqdm
from urllib.request import urlopen
from urllib.error import HTTPError, URLError
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor,as_completed
import random
import os

def build_dataset(thread_cum, urls, correct_sizes, texts, save_path, timeout):
    desc_text = "Thread:" + str(thread_cum)
    total = len(urls)
    success = 0

    for i in tqdm(range(0, total), desc=desc_text):
        url = urls[i]
        correct_w, correct_h = correct_sizes[i]
        name = str( thread_cum*total + i )
        text = texts[i]

        try:
            # Get Web Image+size
            IMG = Image.open(urlopen(url,timeout=timeout))
            res_w, res_h = IMG.size
            # Filter Blocked Images
            if res_w == correct_w and res_h == correct_h:
                # save image
                pth = save_path+name+"/"
                os.makedirs(pth)
                IMG.save(pth+"image.png")
                # save prompt
                with open (pth+"text.txt", "w") as f:
                    f.write(text)
                success += 1
        except:
            pass
    return success

class LaionBuilder():
    def __init__(self, token="", dataset_name="laion/laion2B-en-aesthetic", tgt_url="URL", tgt_txt="TEXT", shuffle=True):
        # Load Meta
        print("loading meta")
        self.ds = load_dataset(dataset_name, token=token, split="train", streaming=True)
        if shuffle==True:
            seed=random.randint(0, 1000)
            self.ds = self.ds.shuffle(seed=seed)
            print("shuffling with seed:", seed)
        self.default_path = "./" + dataset_name + "/"
        self.tgt_url = tgt_url
        self.tgt_txt = tgt_txt
        print("LAION - meta data loaded successfully! ")
        print("DATASET:", dataset_name, "Image-URL col heading:", tgt_url, "Text col heading:", tgt_txt)

    def load(self, num_data, save_path=".", num_workers=1, timeout=10):
        if save_path == ".":
            save_path = self.default_path
        # Check path
        if not os.path.exists(save_path):
            # Create the directory
            os.makedirs(save_path)
        # Init
        thread_cum = 0
        # Download
        while num_data != 0:
            if num_data > 100:
                # parallel
                num_per_thread = int(num_data / num_workers)
            else:
                # single
                num_per_thread = num_data
                num_workers = 1
            # start
            threads = []
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                for i in range(0, num_workers):
                    this_thread_id = thread_cum
                    # Get shard
                    if i==num_workers-1:
                        this_ds = list(self.ds.take(num_data - num_per_thread * (num_workers-1) ))
                    else:
                        this_ds = list(self.ds.take(num_per_thread))
                    # Calc
                    urls = [ent[self.tgt_url] for ent in this_ds]
                    correct_sizes = [ [ent["WIDTH"], ent["HEIGHT"]] for ent in this_ds]
                    texts = [ent[self.tgt_txt] for ent in this_ds]
                    # Start Thread
                    threads.append(pool.submit(build_dataset, this_thread_id, urls, correct_sizes, texts, save_path, timeout))
                    # Skip Used
                    self.ds = self.ds.skip(num_per_thread)
                    thread_cum += 1
                # Summarize
                for thread in as_completed(threads):
                    num_data -= thread.result()
                    pool.shutdown(wait=True)
            # batch summarize
            print("cycle complete. remaining tasks: ", num_data)
