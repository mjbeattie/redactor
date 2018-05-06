"""
unredactor.py
This set of functions unredeacts text files from a directory and allows for storage
of the redacted files in another location.  It can create a feature/label set or it
can use one previously created and stored in a pickle file.  The program creates
three classifiers, a Naive Bayes, Decision Tree, and SVM.  It uses the most accurate
one to run the unredaction.  The program takes in a file whose name is of the format
of the Stanford IMDB set and attempts to replace names that have been redacted with
lower case thorn characters.
Created on Sun Apr 8 19:57:34 2018

@author: Matthew J. Beattie, DSA5970
coding = utf-8
"""

import glob
import re
from random import shuffle
import joblib
import sys
import argparse
import os

from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
from nltk import ne_chunk
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn import svm

import logging
import importlib

importlib.reload(logging)  # To stop repeated outputs in iPython

# Setup logging for program
log = logging.getLogger("UNREDACTOR")
log.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s %(levelname)s:%(name)s %(message)s")
ch.setFormatter(formatter)

log.addHandler(ch)


def get_entity(text):
    """
    get_entity() prints the PERSON entities within a text stream
    :param text: text stream containing PERSON entities
    :return:  none
    """
    for sent in sent_tokenize(text):
        for chunk in ne_chunk(pos_tag(word_tokenize(sent))):
            if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                print(chunk.label(), ' '.join(c[0] for c in chunk.leaves()))


def extract_features(text, fname):
    """
    extract_features() finds PERSON chunks and creates a feature set that includes different
    attributes.  It takes the previous two chunks (a chunk bigram) and also takes the
    length of the PERSON chunk and number of spaces within it.
    :param text: input text string, filename of text string
    :return: list of dictionary features and list of PERSON values
    """
    r = re.compile(r'\d*.txt', )
    review = r.findall(fname)
    review = re.sub(r'\.txt', '', review[0])
    feature_list = []
    label_list = []
    for sent in sent_tokenize(text):
        chunks = ne_chunk(pos_tag(word_tokenize(sent)))
        for i in range(0, len(chunks)):
            if hasattr(chunks[i], 'label') and chunks[i].label() == 'PERSON':
                pnoun = ' '.join(c[0] for c in chunks[i].leaves())
                features = [
                    int(review),
                    len(pnoun)
                ]
                feature_list.append(features)
                label_list.append(pnoun)
    return (feature_list, label_list)


def setup_data(traincount, testcount):
    """
    setup_data() creates training and test data from the imdb dataset.  This dataset is assumed to
    exist in the imdb folder within this project.  The routine takes in parameters that set the
    number of files to be used for feature extraction.  Because feature extraction is a lengthy
    process, the routine saves the feature and label sets as pickle files to the local directory.
    :param traincount: number of training files to be used
    :param testcount:  number of test files to be used
    :return: features and labels list for training and testing.  Saves as pickle files.
    """
    # Collect training files into a list and randomly shuffle
    log.info("Accessing imdb files for training data...")
    trainf1 = glob.glob("/projects/imdb/aclImdb/train/neg/*.txt")
    trainf2 = glob.glob("/projects/imdb/aclImdb/train/pos/*.txt")
    alltrain = trainf1 + trainf2
    shuffle(alltrain)
    log.info("Number of shuffled training files:  " + str(len(alltrain)))
    trainlist = alltrain[:traincount]

    # Collect testing files into a list and randomly shuffle
    log.info("Accessing imdb files for testing data...")
    testf1 = glob.glob("/projects/imdb/aclImdb/test/neg/*.txt")
    testf2 = glob.glob("/projects/imdb/aclImdb/test/pos/*.txt")
    alltest = testf1 + testf2
    shuffle(alltest)
    log.info("Number of shuffled testing files:  " + str(len(alltest)))
    testlist = alltest[:testcount]

    # Create a training set of data for the classifiers
    log.info("Extracting features and labels from training data...")
    trainFeatures = []
    trainLabels = []
    counter = 0
    for file in trainlist:
        f = open(file)
        raw = f.read()
        f.close()
        if counter % 10 == 0:
            log.info("Evaluating training file number " + str(counter) + " of " + str(len(trainlist)))
        newdata = extract_features(raw, file)
        trainFeatures += newdata[0]
        trainLabels += newdata[1]
        counter += 1
    log.info("Completed creation of training features and labels")

    # Create a testing set of data for the classifiers
    log.info("Extracting features and labels from testing data")
    testFeatures = []
    testLabels = []
    counter = 0
    for file in testlist:
        f = open(file)
        raw = f.read()
        f.close()
        if counter % 10 == 0:
            log.info("Evaluating testing file number " + str(counter) + " of " + str(len(testlist)))
        newdata = extract_features(raw, file)
        testFeatures += newdata[0]
        testLabels += newdata[1]
        counter += 1
    log.info("Completed creation of testing features and labels")

    return trainFeatures, trainLabels, testFeatures, testLabels


def build_models(trainFeatures, trainLabels, testFeatures, testLabels):
    """
    build_models() takes as input test and training data and uses the scikit packages
    to build three classifiers.  One Naive Bayes, and one Decision Tree, SVM performance was
    too poor to include.  The routine prints out the progress and accuracy of each
    of the models.
    :param trainFeatures: A list of either integer or vectorized features for training
    :param trainLabels: A list of dependent variable classifications for training
    :param testFeatures: A list of either interger or vectorized features for testing
    :param testLabels: A list of dependent variable classifications for testing
    :return: classifiers and accuracy scores in a dictionary object
    """
    # Run Naive Bayes classification and analyze results
    # Train Naive Bayes classifier
    log.info("Building Bayes classifier...")
    bayesclf = MultinomialNB().fit(trainFeatures, trainLabels)
    log.info("Bayes classifier complete")

    # Predict results based upon Naive Bayes classifier and check accuracy
    bayespredict = bayesclf.predict(testFeatures)
    bayesscore = bayesclf.score(testFeatures, testLabels)
    print("The accuracy of the Naive Bayes classifier is: " + str(bayesscore))

    # Run Decision Tree classification and analyze results
    # Train Decision Tree classifier
    log.info("Building Decision Tree classifier...")
    dtclf = tree.DecisionTreeClassifier().fit(trainFeatures, trainLabels)
    log.info("Decision Tree classifier complete")

    # Predict results based upon Decision Tree classifier and check accuracy
    dtpredict = dtclf.predict(testFeatures)
    dtscore = dtclf.score(testFeatures, testLabels)
    print("The accuracy of the Decision Tree classifier is: " + str(dtscore))

    # Select best model
    if bayesscore > dtscore:
        bestmodel = bayesclf
    else:
        bestmodel = dtclf

    # Return the classifiers and accuracy scores in a dictionary object, including the
    # best model choice
    return dict(BAYESCLF=bayesclf, BAYESSCORE=bayesscore, DTCLF=dtclf, DTSCORE=dtscore, BESTMODEL=bestmodel)


def unredact_file(text, fname, clf):
    """
    unredact_file() takes an IMDB file with a redacted names (as lower case
    thorns) and replaces the redactions with predictions made by the
    classifier sent to the routine
    :param text: input text string
    :param fname: filename of text string
    :param clf: classifier
    :return: unredacted text stream
    """
    # Extract review value from filename
    r = re.compile(r'\d*.txt', )
    review = r.findall(fname)
    review = re.sub(r'\.txt', '', review[0])

    # Build list of redacted names
    log.info("Identifying redacted words...")
    redactlist = []
    for sent in sent_tokenize(text):
        for word in word_tokenize(sent):
            if u'\xfe' in word:
                redactlist.append(word)

    # Determine replacements for redacted words a store in tuples
    replacelist = []
    log.info("Generating predictions...")
    for word in redactlist:
        predict = clf.predict([[review, len(word)]])
        replacelist.append([word, predict[0]])

    # Replace thorns in string with replacement value
    newtext = text
    for tuple in replacelist:
        newtext = re.sub(tuple[0], tuple[1], newtext)

    log.info("Returning redacted string...")
    return newtext


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # The --input flag defaults to the current directory unless another one is specified
    parser.add_argument("--input", type=str, help="The directory of *.txt files, e.g. ./ or " \
                                                  "/<directory>/  The directory must exist and the files must include " \
                                                  "the rating value as the last character before .txt extension.",
                        default='./')
    parser.add_argument("--pickle", type=str, help="Name of optional pickle file containing pre-built classifier")
    parser.add_argument("--output", type=str, help="Target directory for output files' \
                        '(must exist, enter as ./<dirname>/)")
    parser.add_argument("--traincount", type=int, help="Integer value for number of " \
                                                       "training files, default=1000", default=1000)
    parser.add_argument("--testcount", type=int, help="Integer value for number of " \
                                                      "testing files, default=1000", default=1000)
    args = parser.parse_args()

    #Create a list of files to redact.
    redfilelist = glob.glob(args.input + "*.txt.redacted")
    if redfilelist == []:
        log.info("No redacted files found, exiting...")
        sys.exit(0)
    else:
        print("Unredacting files: " + str(redfilelist))

    # Select a classifier
    # If using pickled classifier, load it
    if args.pickle:
        log.info("Accessing pickled classifier...")
        try:
            clf = joblib.load(args.pickle)
        except:
            print("Could not open pickle files in given directory, exiting...")
            sys.exit(1)
    # If not using pickled classifier extract features from IMDB and build models
    else:
        log.info("Extracting features and labels from IMDB...")
        trainFeatures, trainLabels, testFeatures, testLabels = setup_data(args.traincount, args.testcount)

        # Generate classifier models, including best selection
        log.info("Building models...")
        clfdict = build_models(trainFeatures, trainLabels, testFeatures, testLabels)
        clf = clfdict['BESTMODEL']

    # Run the redaction routines for the files in the list
    for redfile in redfilelist:
        # Read in document to be redacted into a string
        try:
            f = open(redfile)
            raw = f.read()
            f.close()
        except:
            print("File read error, exiting...")
            sys.exit(1)

        # Create output file name
        outname = re.sub(args.input, "", redfile) + ".unredacted"
        outname = re.sub('.redacted.', '.', outname)

        # Open output file
        try:
            if args.output:
                fileOut = open(os.path.join(args.output, outname), 'w', encoding='utf-8')
            else:
                fileOut = open(outname, 'w', encoding='utf-8')
        except:
            print("Could not open output file, was directory name in form of ./<name>?  Exiting...")
            sys.exit(1)

        # Perform unredaction
        log.info("Unredacting string from file...")
        newstring = unredact_file(raw, redfile, clf)

        # Write redacted string to the output file
        log.info("Writing string to file...")
        print(newstring, file=fileOut)
        fileOut.close()

    log.info("\nUnredactor complete\n")
