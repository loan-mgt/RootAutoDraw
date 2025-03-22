# RootAutoDraw
It is very simple to run RootAutoDraw. We will run it on Google Colaboratory which is free and only requires a Gmail account to be able to use.
First, go to this Colaboratory: https://colab.research.google.com/drive/1vK4R281YmL8AcE7KefxBUc72nclIrzuK?usp=sharing.

At the same time, follow this video for instruction: https://youtu.be/DLyUSksOkwI

Feel free to create issues for questions.
Thank you!

## Docker

```bash
docker run -v ./input:/app/input:ro -v ./output:/app/output root-auto-draw ghcr.io/loan-mgt/rootautodraw:latest
```

### For local build

build:
```bash
docker build . -t root-auto-draw
```

run:
```bash
docker run -v ./input:/app/input:ro -v ./output:/app/output root-auto-draw
```