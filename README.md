# point-to-define

Optical character recognition and feature detection are two important aspects of image analysis, which I combined in this project. My goal was to make an an application with which a human can directly interact. This application allows a user to hold a paper containing text in a foreign language in front of a camera, and then the user can point at any of the words on the piece of paper. The application will recognize whichever word is being pointed at, translate this word into English and display this translation on the video output. This application would be used if someone easily wants to translate words from a foreign language into English without having to type every word into an online dictionary. A user can save a significant amount of time just by pointing at the word, instead of typing. I wrote this application in Python and used OpenCV.

## Features
1. Detect location at which user points
  - Use skin color histogram
  - Find finger by looking for contours in skin color region
2. Detect paper region
  - Use paper color histogram
  - OCR text on paper
3. Translate word at which user points
  - Use Google Translate to find translation

##### Libraries
- OpenCV: image analysis, feature detection
- Tesseract: optical character recognition
- goslate: Python package for Google Translate

##### How it works
###### Train paper histogram
![alt tag](http://i.imgur.com/pv8p1mD.png)

###### Train hand histogram
![alt tag](http://i.imgur.com/Vo4XqpD.png)

###### Find contours
![alt tag](http://i.imgur.com/ct16Y8W.png)

###### Find defects of largest contour
![alt tag](http://i.imgur.com/6oxoJu5.png)

###### Find farthest point from center of hand
![alt tag](http://i.imgur.com/JDSKAfT.png)

###### Result showing user pointing at word which is translated
![alt tag](http://i.imgur.com/q7nL0xz.png)

##### Videos (click on each image to go to YouTube)
<a href="http://www.youtube.com/watch?feature=player_embedded&v=rwpYMnfHBKc
" target="_blank"><img src="http://img.youtube.com/vi/rwpYMnfHBKc/0.jpg" 
alt="point-to-define 1" width="420" height="315" border="10" /></a>

<a href="http://www.youtube.com/watch?feature=player_embedded&v=_LGGYTyVZwE
" target="_blank"><img src="http://img.youtube.com/vi/_LGGYTyVZwE/0.jpg" 
alt="point-to-define 2" width="420" height="315" border="10" /></a>

## Requirements
OpenCV and Tesseract need to be installed.

Install OpenCV (on OS X):
```
pip install numpy
brew tap homebrew/science
brew install opencv
```
Install Tesseract (on OS X):
```
brew install tesseract
```

## Installation
Set up virtualenv (optional but recommended):
```
virtualenv --no-site-packages venv
```
Install packages (if using virtualenv, source it beforehand):
```
pip install -r requirements.txt
```
If using virtualenv, you need to copy the OpenCV site-package files to the virtualenv site-package directory. You can do it like this:
```
cp /usr/local/lib/python2.7/site-packages/cv* ./venv/lib/python2.7/site-packages
```
If that didn't work you can find the location of the cv files by opening a python console and typing:
```
import cv2
print cv2.__file__
```

## Usage
1. From inside project directory on the command line run `python main.py`, or optionally `python main.py –v
<output_video>` where output_video is the path to store the video output.
2. Next the application needs to be trained to recognize a hand and paper.
  - Hold a piece of paper with text inside the green rectangle and then press the `P` key.
  - Hold your hand, so it is inside the green rectangles, and then press the `H` key.
3. Point at any word on the paper for about a second and then a translation will be displayed on screen.
4. To quit press the `Q` key.
 
## To do
- Add tests
- Add support for more languages, currently only German->English is included
- Test for support for smaller and larger camera resolutions
- Use constants when finding regions of interest
- Only apply HSV to regions of interest
- Handle case where there is no internet connection
