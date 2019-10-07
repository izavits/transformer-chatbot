# chatbot

> A transformer chatbot.

This is a chatbot based on a neural network architecture, called transformer. The transformer model is
entirely built on the self attention mechanisms without using sequence-aligned recurrent architecture.

## Table of Contents

- [Install](#install)
- [Usage](#usage)
- [Tests](#tests)


## Install
### Install locally
The local installation concerns the creation of the virtual environment with the needed libraries and the
assumes the existence of Python3 language.

- Create your virtual environment and install the required dependencies:

```
virtualenv -p `which python3` venv
source venv/bin/activate
pip install requirements
``` 

### Use docker
For concenience purposes, the application comes Dockerized. All you need is to have Docker installed 
on your machine in order to build the image and run the container.

- Build image:

```
docker build -t <user>/chatbort
```

- Run the image:

```
docker run -it --rm <user>/chatbot
```


## Usage
If you chose to run the application using the Docker image, then all you need to do is to follow the 
Docker installation steps described above. When you run the docker image, the application will start 
automatically. The model comes pre-trained using the `ALARM_SET` dataset in order to quickly start using
it and do some discussions with the chatbot.

If you have the application installed on your local system, then some basic commands are provided:

- Run the pre-trained model:

```
cd src
python main.py
```

- See usage:

```
cd src
python main.py --help
```

You will get the following output:

```
usage: main.py [-h] [--train]

Welcome to chatbot.
Use the config.ini file to setup the required parameters and input dataset.
Start chatting with the bot. Type 'bye' or Ctrl^C to exit

optional arguments:
  -h, --help   show this help message and exit
  --train, -t  train the model before using for chatting
```

- Train the model and then use it for discussions:

```
cd src
python main.py --train
```

- Just train the model:

```
cd src
python train.py
```

For your convenience the dataset is also provided in the project directory. Thus, by just editing the
`config.ini` file you can select different train datasets and different model parameters. The trained 
models are stored in the `models` directory of the project.


## Tests
Unit tests are included. Install `pytest` if needed and run it to execute them:

```
pip install pytest
pytest
```
