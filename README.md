# DeepColour

Colours in images.

Sketch Lines             |  Coloured Images
:-------------------------:|:-------------------------:
![alt text](samples/sample_edges.jpg?raw=True "Edge Images")  |  ![alt text](samples/sample_gen.jpg?raw=True "Coloured Images")


## Getting Started

You can clone the repo as is and create two folders: "./imgs" and "./results".  The images you want to train on or sample from go in "./imgs".

```
DeepColour
|-- results
|-- imgs
|   |-- img1.jpg
|   |-- ...
|   `-- img3.jpg
|-- main.py
|-- utils.py
`-- get_images.py
```

### Prerequisites

Tensorflow
```
pip install tensorflow==1.12.0
```

### Work Still to be Done

Model doesn't handle images with backgrounds very well.  Maybe due to the image dataset primarily consisting of simple/white backgrounds.
The generated images are also still somewhat blurry compared to their sharp originals, this is amplified in images with complex or busy backgrounds.
Below shows an example of the blurriness.

![alt text](samples/sample_rough.jpg?raw=True "Blurry and Rough Edges")

Also, edge detection is rough right now.  Many small artifacts are picked up when it'd be preferable to avoid them.  

## Acknowledgments

Code based off of Kevin Frans' original implementation [here](https://github.com/kvfrans/deepcolor).
