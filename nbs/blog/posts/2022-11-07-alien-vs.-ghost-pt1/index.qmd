---
title: Alien vs. Ghost Pt.1 Training
author: Ga
date: 'November 7, 2022'
toc: true
jupyter: python3
---

# Training the model

In this article, we will make a image classification model that attempts to differentiate between aliens and ghosts. Although it is possible to only use CPU to train the model, it is faster to use GPU. I am on Google colab, which provides free GPU. After training, we will save the model so that we can use it for deployment on Hugging Face spaces.

First, we import FastAI library.

```{python}
!pip install -Uqq fastai
!pip install -Uqq fastbook
import fastbook
fastbook.setup_book()
from fastbook import *
from fastai.vision.widgets import *
```

## Collect Data

We use duckduckgo to search and download images. FastAI provides `search_images_ddg` so that we do not have to go to a search engine and download an image one by one!

```{python}
search_results = search_images_ddg('alien images')
```

```{python}
ims = search_results
len(ims)
```

```{python}
dest = 'images/aliens'
download_url(ims[2], dest)
```

We can check the image and see if it looks right. It seems like an alien.

```{python}
im = Image.open(dest)
im.to_thumb(128, 128)
```

Now we download images to our path. We create two directories, alien and ghost, and download each category of images into each directory.

```{python}
category_types = 'alien', 'ghost'
path = Path('alien_ghost')
```

```{python}
if not path.exists():
    path.mkdir()
    for o in category_types:
        dest = (path/o)
        dest.mkdir(exist_ok=True)
        results = search_images_ddg(f'{o} images')
        download_images(dest, urls=results)
```

```{python}
fns = get_image_files(path)
fns
```

Failed one are ones that cannot be opened, so we unlink them from our path.

```{python}
failed = verify_images(fns)
failed
```

```{python}
failed.map(Path.unlink)
```

## DataLoaders

We now create DataBlocks. Simply put, it is a bridge between raw data and a model. We specify input and output for the model, how to get the input, how to split train data from validation data, how data are labelled, and what transfroms are needed for the input. 

```{python}
dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),               # We take images and try to classify them based on categories.
    get_items=get_image_files,                        # inputs are images.
    splitter=RandomSplitter(valid_pct=0.2, seed=42),  # randomly pick 20% of the input for validation.
    get_y=parent_label,                               # parent directory name is the label for each image.
    item_tfms=Resize(128))                            # Resize each image to 128.
```

```{python}
dls = dblock.dataloaders(path)
dls
```

Look at the images and labels to make sure everything looks right. Seems like images are correctly labelled. At this point, it is okay to have some wrong images or labels. We will fix that later.

```{python}
dls.valid.show_batch(max_n=4, nrows=1)
```

We can add data augmentation (transforms) into images so that we can train more efficiently with less data.

```{python}
dblock = dblock.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms())
dls = dblock.dataloaders(path)
```

Now we train the model.

```{python}
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4)
```

There are some errors. We can look at the errors by looking at the confusion matrix and top losses.

```{python}
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
```

```{python}
# Check the high losses
interp.plot_top_losses(10, nrows=2)
```

Some images are not labelled correctly or wrong images. We can easily fix it by using the cleaner. 

```{python}
# Pick which ones to delete or to move to another category
cleaner = ImageClassifierCleaner(learn)
cleaner
```

```{python}
# make cleaner take effect
for idx in cleaner.delete(): cleaner.fns[idx].unlink()
for idx,cat in cleaner.change(): shutil.move(str(cleaner.fns[idx]), path/cat)
```

After cleaning, we train again for the cleaned data. So, we go back up to DataLoaders and come back down here. After that, we can export the model. We can then download the model so that we can deploy it into Hugging Face Spaces.

```{python}
learn.export()
```

```{python}
# check if file is pickled
path = Path()
path.ls(file_exts='.pkl')
```

That's it. If this notebook was run locally, there will be the the model in the same directory as the notebook. If this notebook was run on Google colab, the model has to be downloaded from the directory from the panel on the left. 


