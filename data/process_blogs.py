"""
Run this as a python module.

python -m data.process_blogs
"""

from bs4 import BeautifulSoup
from data import vocab
import re as ree
import tqdm
import glob
import os

def extract_posts(blog_file, vocab_set):
    """
    Extract posts as a list from a blog_file (XML) format.
    Does some basic preprocessing by excluding posts with
    characters not found in the provided vocab_set.

    Also changes spaces to underscores because vocab.
    """

    with open(blog_file) as f:
        try:
            soup = BeautifulSoup(f)
        except UnicodeDecodeError:
            return []
    
    posts = soup.find_all('post')
    
    out = []
    for post in posts:
        
        valid = True
        
        # quick preprocess
        post_text = post.get_text().strip()
        post_text = post_text.replace("\n", "")
        post_text = post_text.replace("urlLink", "")
        post_text = ree.sub("\s{4,}"," ",post_text)
        post_text = post_text.lower()
        
        post_text = post_text.replace(" ", "_")
        
        for char in post_text:
            if char not in vocab_set:
                valid = False
                
        if valid and len(post_text) > 0:
            out.append(post_text)
    
    return out

def extract_all_posts(blogs, vocab_set):
    
    out = []
    for blog in tqdm.tqdm(blogs):
        out.extend(extract_posts(blog, vocab_set))
        
    return out

if __name__ == "__main__":

    BLOG_FILE_DIR = "./data/datasets/blogs"

    vocab = vocab.AdvancedVocab([])
    vocab_set = set(vocab.stoi.keys())

    blogs = glob.glob(os.path.join(BLOG_FILE_DIR, "*.xml"))
    print(f"Found {len(blogs)} blog posts at that directory.")

    out = extract_all_posts(blogs, vocab_set)

    print(f"{len(out)} blog posts survived the preprocess steps")

    with open('blog.txt', 'w') as f:
        f.writelines("%s\n" % post for post in out)