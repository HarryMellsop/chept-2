# chept-2
Second iteration of the ChePT Neural Chess Engine - Refined and Improved

## Enter the Virtual Environment

    $ python -m venv venv
    $ source venv/bin/activate
    $ pip install -r requirements.txt

## Install and Preprocess the Datasets

    $ python data/get_datasets.py --download --preprocess
    
## Pretrain

    $ python run.py pretrain

Then accept the default presets when prompted.  You may need to use the --batch_size flag depending on your hardware constraints, as the model is extremely video memory intensive when using CUDA.