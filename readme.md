

# LIBRARY DEPENDENCIES

These are the libraries needed for the GS.py:
Numpy
Scipy
Matplotlib
Liac-arff 
Sklearn
Pandas

# HOW TO SET UP

First, please put the test arousal and valence annotation under folder ratings_individual accordingly. The annotations from train and dev datasets have already been put in the folders. That means pleas put test_*.csv directly in the folder ratings_individual/arousal/ and ratings_individual/valence.

If the program is running directly in this TestAVEC folder,
you do not need to change the path in the Config.py but the number of threads you would like to use.
If you are using windows, please change the slash in the path into backslash in Config.py and GlobalsVars.py.

# HOW TO RUN

To generate the Gold Standard from the individual ratings(TestAVEC/ratings_individual/):

Go in the TestAVEC folder
Type "python GS.py"
Results will be generated in the TestAVEC folder.

# Output

The generated gold standards are in folder TestAVEC named gs_1, gs_2 to gs_5.
The arousal and valence are stored separately as gs_*/arousal and gs_*/valence

