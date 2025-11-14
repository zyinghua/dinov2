# DiT linear probing for SSL project

## download imagenet 1k test set (13gb)
```
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_var.tar
tar -xvf ILSVRC2012_img_var.tar
```


## download dit xl model
```
wget https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt
```

## How to extract features
* The only file I added in `.DiT` is `./DiT/linprobing`
* path to dataset and checkpoint can be set in argparser @ `./DiT/linprobing`. Please check other arguments as well. 
* Then run:

```
python DiT/linprobing.py
```