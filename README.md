# iihe_sklearn_tutorial
Tutorial on scikit-learn given at the IIHE. 

The goal of this tutuorial is to introduce you to the [Scikit-Learn](http://scikit-learn.org/stable/) machine learning software and some of its features that can be used in Multivariate Analysis (MVA) techniques. This tutorial uses a charm-tagger as an example to go over some interesting features. The code in this tutorial, together with links to the relevant information, should serve as a basis for your own analysis and can be adapted accordingly.

Slides from Kevins talk can be found [here](http://mon.iihe.ac.be/~kderoove/MachineLearning/MachineLearningInHEP.pdf)

## Minimal installation
In order to follow this tutorial, **this minimal installation is required**. The extensions below will allow you to also test specified features that will be revised during the tutorial, but alternatives are in place for those who do not which to (or do not manage to) install these extensions. **This installation should be done locally on your laptop (not on the m-machines for example)**

This tutorial uses [Jupyter](http://jupyter.org/) notebook to compile python code on the go. Jupyter, along with other scientific python packages that are needed, is included in the [Anaconda](https://www.continuum.io/downloads) python distribution. Therefore it is required to dowload Anaconda:

For Linux: <a href="https://3230d63b5fc54e62148e-c95ac804525aac4b6dba79b00b39d1d3.ssl.cf1.rackcdn.com/Anaconda2-2.4.0-Linux-x86_64.sh"> With python 2.7, 64-bit installation</a>

For Mac: <a href="https://3230d63b5fc54e62148e-c95ac804525aac4b6dba79b00b39d1d3.ssl.cf1.rackcdn.com/Anaconda2-2.4.0-MacOSX-x86_64.sh"> With python 2.7, 64-bit installation</a>

Once you downloaded the installer, open a terminal and move to the directory where the .sh file is located. Then do the following (note that this is for the Anaconda installation with python 2.7, 64-bit. If you installed another release, replace the name of the .sh file accordingly):

For Linux: `bash Anaconda2-2.4.0-Linux-x86_64.sh`

For Mac: `bash Anaconda2-2.4.0-MacOSX-x86_64.sh`

*NOTE: During the installation Anaconda will ask you to read and accept the license agreement. You will need to press enter a few times carefully untill the yes/no acceptance question appears. If you click enter too fast you will select 'no' by default and the installation will terminate! Also at the end of the installation you will have to answer 'yes' once more.*

After the installation is finished, there should be a new directory in your home folder named anaconda(2). Test if Anaconda is indeed installed by typing `conda` in your command line. If this does not say something like `conda: command not found`, you are good. 

*NOTE: you might need to restart your terminal after the installation, since anaconda adds a line to your .bash_profile or .bashrc that sets the correct PATH. If this is not the case, please set: `PATH="/YOUR/HOME/DIRECTORY/anaconda(2)/bin:$PATH`*

Finally you can install the necessary scientific python packages and Jupyter by doing:

`conda install numpy scipy scikit-learn jupyter matplotlib`

You can test if this succeeded by opening python (`python`) and importing the packages (for example: `import sklearn`).

*NOTE: If you already have a python distribution installed locally on your laptop, it might be that you need to activate the Anaconda environment:*

`source activate /YOUR/HOME/DIRECTORY/anaconda(2)`

**You can now clone this git repository to your laptop (locally) and run the notebook:**

`git clone https://github.com/smoortga/iihe_sklearn_tutorial.git`

`cd iihe_sklearn_tutorial`

`jupyter notebook`

This will open in your default browser a window with the content of this repository. Click on the CharmTagger_SkLearn.ipynb file to open the tutorial.

## Extension 1: get the latest developers version of Scikit-Learn
Anaconda comes with the latest stable version of Scikit-learn (version 0.17 at this time of writing). However some features in the tutorial are only available in the developers version (0.18dev). Therefore you might want to download this latest version from [github](https://github.com/scikit-learn/scikit-learn) and install it:

`git clone git://github.com/scikit-learn/scikit-learn.git`

`cd scikit-learn`

`python setup.py install`

`make`

`export PYTHONPATH="/DIRECTORY/WHERE/YOU/CLONED/GITREPO/scikit-learn:$PYTHONPATH"` (or put this in your .bash_profile)

You can now test if this worked by doing the following, which should now say something like '0.18.dev0':

`python`

`import sklearn`

`sklearn.__version__`

## Extension 2: Installing root_numpy
In this tutorial (and in most research activities at IIHE) we will work with data that are stored in .root files. In order to read ROOT files within python, you will need the root_numpy package. This of course assumes you have an installation of ROOT locally on your laptop (if not, you can install from [here](https://root.cern.ch/downloading-root)). If you do not wish to do this, the tutorial can also be executed without the ROOT files, as also the numpy-style arrays are stored on this github repository.

Once you have a local installation of ROOT, do the following:

`conda install --channel https://conda.anaconda.org/NLeSC root_numpy`

EXTRA: you can also download *rootpy*, but this is not needed in this tutorial:

`conda install --channel https://conda.anaconda.org/NLeSC rootpy`
