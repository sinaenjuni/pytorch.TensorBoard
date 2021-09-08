# pytorch.TensorBoard
Guide to Using TensorBoard with PyTorch

# Requirements
```shell
pytorch
tensorboard

optional(with GPU):
torchvision
cudatoolkit

Anaconda Installation Command:
conda insatll pytorch torchvision cudatoolkit=[According to your cuda version]
conda install -c conda-forge tensorboard

```

# Using
1. import tensorboard class:
    ```python
    from torch.utils.tensorboard import SummaryWriter
    ```
   
2. Displaying Images And Graphs with TensorBoard:
   ```python
   tb = SummaryWriter(log_dir='./runs', comment=[INDEXING])
   model = CNN()
   images, labels = next(iter(train_loader))
   grid = torchvision.utils.make_grid(images)
   tb.add_image("images", grid)
   tb.add_graph(model, images)
   tb.close()
   ```
   
   ####TIP
   Easy to view image list if use global_step parameter 
   ```python
   tb.add_image("images", grid, global_step=idx)
      
   ```
   
3. Start Tensorboard sever
   ```shell
   !tensorboard --logdir runs --host [REMOTE SERVER IP] --port [REMOTE SERVER PORT]
   ```