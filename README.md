# DigiGait

An interactive dashboard to easily use 3D Human Pose Estimation for analyzing
the knee angle trajectories in gait analysis. At the moment [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose) and [LPN](https://github.com/zhang943/lpn-pytorch) are supported for 2D pose estimation and [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) is used for 2D to 3D inference. Using automatic gait event detection the knee angle will be split into seperate strides and time normalized to 101 time points (0-100%). In addition several common metrics (e.g. joint range and max peak at loading response) are derived from this data.


## Running locally

To run a development instance locally, create a virtualenv, install the 
requirements from `requirements.txt` and launch `app.py` using the 
Python executable from the virtualenv.

## Deploy on Heroku

The app can be easily deployed to Heroku and used with their free tier.
[![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy)

## Deploying on ECS

Use `make image` to create a Docker image. Then, follow [these 
instructions](https://www.chrisvoncsefalvay.com/2019/08/28/deploying-dash-on-amazon-ecs/) 
to deploy the image on ECS.