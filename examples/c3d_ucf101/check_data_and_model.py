#!/usr/bin/env python

# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
caffe_root = '../../'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe
import glob
import matplotlib.pyplot as plt
import numpy as np
import os

# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.

model_weights = './c3d_ucf101_iter_20000.caffemodel'
if not os.path.isfile(model_weights):
    print "[Error] model weights can't be found."
    sys.exit(-1)

print 'model found.'
caffe.set_mode_gpu()
caffe.set_device(1)
model_def = './c3d_ucf101_deploy.prototxt'

classes = [
    "ApplyEyeMakeup", "ApplyLipstick", "Archery", "BabyCrawling", "BalanceBeam", "BandMarching",
    "BaseballPitch", "Basketball", "BasketballDunk", "BenchPress", "Biking", "Billiards",
    "BlowDryHair", "BlowingCandles", "BodyWeightSquats", "Bowling", "BoxingPunchingBag",
    "BoxingSpeedBag", "BreastStroke", "BrushingTeeth", "CleanAndJerk", "CliffDiving",
    "CricketBowling", "CricketShot", "CuttingInKitchen", "Diving", "Drumming", "Fencing",
    "FieldHockeyPenalty", "FloorGymnastics", "FrisbeeCatch", "FrontCrawl", "GolfSwing",
    "Haircut", "HammerThrow", "Hammering", "HandstandPushups", "HandstandWalking",
    "HeadMassage", "HighJump", "HorseRace", "HorseRiding", "HulaHoop", "IceDancing",
    "JavelinThrow", "JugglingBalls", "JumpRope", "JumpingJack", "Kayaking", "Knitting",
    "LongJump", "Lunges", "MilitaryParade", "Mixing", "MoppingFloor", "Nunchucks", "ParallelBars",
    "PizzaTossing", "PlayingCello", "PlayingDaf", "PlayingDhol", "PlayingFlute", "PlayingGuitar",
    "PlayingPiano", "PlayingSitar", "PlayingTabla", "PlayingViolin", "PoleVault", "PommelHorse",
    "PullUps", "Punch", "PushUps", "Rafting", "RockClimbingIndoor", "RopeClimbing", "Rowing",
    "SalsaSpin", "ShavingBeard", "Shotput", "SkateBoarding", "Skiing", "Skijet", "SkyDiving",
    "SoccerJuggling", "SoccerPenalty", "StillRings", "SumoWrestling", "Surfing", "Swing",
    "TableTennisShot", "TaiChi", "TennisSwing", "ThrowDiscus", "TrampolineJumping", "Typing",
    "UnevenBars", "VolleyballSpiking", "WalkingWithDog", "WallPushups", "WritingOnBoard", "YoYo"
]

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

# convert from HxWxCxL to CxLxHxW (L=temporal length)
length = 16
transformer.set_transpose('data', (2,3,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

# set the size of the input (we can skip this if we're happy
# with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(1,         # batch size
                          3,         # 3-channel (BGR) images
                          length,    # length of a clip
                          112, 112)  # image size

#frame_files = sorted(glob.glob(caffe_root + 'examples/videos/youtube_objects_dog_v0002_s006/*.jpg'))
frame_files = sorted(glob.glob(caffe_root + 'examples/videos/UCF-101_Rowing_g16_c03/*.jpg'))[:16]

frame_img = caffe.io.load_image(frame_files[0])
clip = np.empty((16,)+ frame_img.shape, dtype=np.uint8)
for frame_num, frame_file in enumerate(frame_files):
    frame_img = caffe.io.load_image(frame_file)
    clip[frame_num,:,:,:] = frame_img

#clip = np.tile(
#       caffe.io.load_image(caffe_root + 'examples/images/cat.jpg'),
#        (16,1,1,1)
#        )
clip = np.transpose(clip, (1,2,3,0))
print "clip.shape={}".format(clip.shape)

transformed_image = transformer.preprocess('data', clip)

#plt.imshow(image)

# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image
for l in range(0, length):
    print "net.blobs['data'].data[0,:,{},:,:]={}".format(
            l,
            net.blobs['data'].data[0,:,l,:,:]
            )

### perform classification
output = net.forward()

output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

print 'predicted class is:', classes[output_prob.argmax()]
