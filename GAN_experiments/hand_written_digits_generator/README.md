# Hand-written digits generator
Inspired from [Diego Gomez Mosquera's tutorial](https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f), huge thanks to him, and featuring [VanillaGAN](https://arxiv.org/abs/1406.2661).  
For visualization:  
![numbers as generated over epoch](images/numbers.gif)

## Good data habits for GANs, based on [this github](https://github.com/soumith/ganhacks)
Note: It would worth to double check all of them, to understand them better
- tanh as last layer of generator? Why that?
- Used to be `min (log 1-D)`, but in practice we now use `max log D`, more intuitive nowadays
- Replay (reshow old tracks to current model) seems to work sometimes, worth a random try now and then
- Use [Adam optimizer](https://arxiv.org/pdf/1412.6980.pdf) --- TO READ
- Statistically balancing the trainings of D and G is rarely (never) a good idea

## Set up
```shell
pip3 install torch torchvision tensorboardx jupyter matplotlib numpy
apt-get install python3-tk
```
 Also command line to generate the gif ẁith Imagemagick at the end, if needed:
 ```shell
 convert -delay 5 -resize 50% -loop 0 `ls -v | grep -v hori` numbers.gif
 ```