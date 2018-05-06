Module:  redactor
Author:  Matthew J. Beattie
Author email:  mjbeattie@ou.edu
Version Date:  April 9, 2018
Class:  Univ of Oklahoma DSA5970 (Dr. Grant)

**OVERVIEW**
redactor is a module that contains a set of functions and two programs that can be used
to find, redact, and unredact sensitive information from text files.  Redaction can be done
on several different types of information, including names, addresses, concepts, gender,
dates, phone numbers, and email addresses.  Unredaction on the other hand is limited to
names of people only.


**INSTALLATION**
The files for the module are archived in a tar ball called beat0000_project1.tar.gz (
or if on github, the tarball is redactor.tar.gz).  To install the file, copy it to 
the /projects/ directory and unpack.  The module can then be installed via pip with 
the following command:

     pip3 install --editable .

Depending on your permissions, you may need to sudo the install.  The unredactor portion of
the module uses data files from the Stanford IMDB sentiment classification project to
generate classifers.  unredactor.py has a hard coded reference to the location for these files,
so they must be in the correct location.  To use unredactor.py without modification, create
a directory /projects/imdb.  From this directory, get the Stanford tarball with the
following command:

     wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz

Unpack the tarball into the current directory.  If you need to use a different directory
structure, you will need to modify the code in unredactor.py.  The location of the reference
is evident.


**TESTING**
The redactor module comes with a set of test files that test the main redaction and
unredaction routines.  Note that the unredactor test does NOT create a new classifier.
Instead, it uses an already created classifier that has been storeed in the redactor root
directory as a pickle file.  (More on this in the --pickle option for unredactory.py).
To run the test routines, enter the following into the command line from the /projects/redactor
directory:

     python3 -m pytest -v

Pytest will test the functions for the module.  Do not expect the unredact_file() routine
to pass.  The accuracy of the classifer is only a couple percentage points.


**RUNNING REDACTOR.PY**
redactor.py is the program that performs redaction on a glob of text files in a location
supplied by the user.  This program performs very well, correctly identifying and redacting
most entities.  There are some option specific limitations to the program that deal with
expecting some entities in a specific format, but these limitations exist for even
production packages such as NLTK and spaCy.

Invocation:  redactor is called from the linux command line as:  python3 redactor.py
Options:  --input <directory>   This is the directory containing the files to be redacted.
                                The format is specific.  The directory should be similar to
                                this example:  ./tests/
                                The module reads files with the *.txt extension.  Files that
                                do not have this extension will not be read.  The default value
                                for --input is ./

          --output <directory>  This is the directory to which the redacted files will be
                                written.  As with input, the directory should be similar
                                to the example ./docs/

          --stats <filename>    This flag requires a filename to which a summary of the redaction
                                will be written.  If the flag is not present the stats will be
                                written to stdout.

          --names               The presence of this flag will result in a redaction of formal
                                names from the documents.  Unlike the other options, which
                                result in redactions of the form <gender redacted>, --names
                                replaces a person's proper name with a string of lower
                                case thorns.  This is to support unredaction.

          --genders             This flag causes the redaction of gender-specific nouns, 
                                pronouns, and possessives.

          --dates               Causes the redaction of dates in several formats

          --addresses           Causes the redaction of standard format addresses

          --phones              Causes the redaction of phones of the format (xxx) xxx-xxxx

          --email               Causes the redaction of email addresses

          --concepts <concept>  Causes the redaction of sentences containing stems of words that
                                match the given concept

**HOW IT WORKS:**
The module includes several functions that perform the redactions above.  Additionally, there
are functions that return the redaction matches.  These functions are not part of the DSA5970
assignment, but should be of use for other purposes.  The module also includes a py.test
procedure that checks the redaction functions.

While testing, the program works pretty well, but there remain some bugs that could be ironed
out if going to production.  For example, if a concept sentence is the first sentence of a
letter, all prior content is redacted as well.  Since this content is a set of names and
addresses, it doesn't hinder the usefulness of the module, but nevertheless this should be
fixed in a future version.  Also, I did not write the module to redact ALL forms of addresses,
and in particular does not redact international addresses.

For concept redaction, the module takes the document, converts it to a set of sentence tokens,
and takes each sentence and word_tokenizes it.  It then takes these word tokens, stems them,
and compares those stems to synonyms of the concept to redact.  If there is a match of any
word in the sentence, that sentence is redacted.

**PERFORMANCE:**
In general, the performance of redactor.py is very good, and this program could serve as
a base for a production level redactor.  In particular, the concept and gender redaction,
perhaps the most difficult redactors, work very well.  The --names redactor, which relies
on the recognition of a PERSON nltk chunk works well as long as the person's name begins
with a capital letter and is followed by lower case letters.  This is a shortcoming of both
nltk and spaCy.  I wrote a routine that instead matched names to a names corpus, but opted
not to use it because it often missed last names.


**RUNNING UNREDACTOR.PY**
unredactor.py takes a glob of files in a directory specified by the user and attempts to
unredact proper names that have been replaced by lower case thorns.  The files that are
read must end in .txt.redacted so that the program recognizes them.  

Invocation:  redactor is called from the linux command line as:  python3 unredactor.py
Options:  --input <directory>   This is the directory containing the files to be redacted.
                                The format is specific.  The directory should be similar to
                                this example:  ./tests/
                                The module reads files with the *.txt extension.  Files that
                                do not have this extension will not be read.  The default value
                                for --input is ./  The files provided must have a name
                                of the format *d.txt.redacted, where 'd' is a digit.

          --output <directory>  This is the directory to which the redacted files will be
                                written.  As with input, the directory should be similar
                                to the example ./docs/

          --pickle <filename>   This flag enables the user to bypass classifier creation and
                                use one created previously and stored as a pickle file.  There
                                are two included in this module:  dclf.pkl (Naive Bayes),
                                dtclf.pkl (Decision Tree).  These classifiers were created
                                using the SciKit package and 3000 training and 3000 test
                                IMDB files.

          --traincount <int>    The number of training IMDB files to use when creating a new
                                classifier.
                                
          --testcount <int>     The number of testing IMDB files to use when creating a new
                                classifier.
                                
**HOW IT WORKS:**
unredactor.py seeks to unredact proper person names by using a classifier that is either provided
by the user or built by unredactor.py.  The program uses the Stanford IMDB files as data
for building such a classifier.  In brief, the program reads in a redacted file, extracts the
features used by the classifier, and predicts the missing names based upon those features.
The program then writes unredacted files to a directory chosen by the user.

**CLASSIFICATION:**
There are two classifiers built in the program.  One is a Naive Bayes classifier, and the other
is a Decision Tree classifier.  In earlier versions of the program, I also built a Support
Vector Machines classifier, but building it was so CPU and memory intense that I had to give
up that effort.  The program chooses which one performed best against the test
set of data and uses that classifier for prediction during the unredaction step.  Both
classifiers use two features:  the "review" associated with the redacted file and the length
in letters and spaces of the redacted name.  The review is obtained from the name of the
redacted file and is represented as the digit 'd':  d.txt.redacted.

I wanted to use more feautures for classification, but found that just getting this very simple
model to work was difficult enough.  I believe that the difficulty comes not from feature
extraction, but from the fact that the dependent variables in the classification are in a set
of over 5000 possibilities.  So there are more possible names than feature sets!

**PERFORMANCE:**
Unfortunately, the unredactor does not perform very well.  This is no surprise since there are
only two features per redacted word and thousands of potential names.  The best performance
I saw was 0.0108 by the Decision Tree model at 3000 training and 3000 test files.  However,
while this performance was low, the probability of randomly choosing the right name is about
.000168 (6,000 names), so we did see an almost hundredfold improvement in our chances.  The 
performance of the unredactor classifiers is given below:

     # of test/tng files   |     Model       |   Accuracy   | Time to build classifier (sec)
     ---------------------------------------------------------------------------------------
          1000             |  Bayes          |  0.00378     |  0.252
                           |  Decision Tree  |  0.00597     |  0.029
     ---------------------------------------------------------------------------------------
          3000             |  Bayes          |  0.00564     |  1.816
                           |  Decision Tree  |  0.0108      |  0.0850
     ---------------------------------------------------------------------------------------
          5000             |  Bayes          |  0.00420     |  4.506
                           |  Decision Tree  |  0.00894     |  0.139
     ---------------------------------------------------------------------------------------

To run the program, I increased the capacity of the Google Compute Instance to 4 CPUs and
26GB of RAM.  I required this to run the SVM models.  I did not experiment with lower speed
configuration, but the routines should run okay on 2 CPUs and 16 GB of RAM.

**SUGGESTIONS FOR IMPROVEMENT:**
Performance of the unredactor can be made by building models with more features.  Some would
include the presence of other proper nouns as a vectorized dictionary or gender words, again
as a vectorized dictionary.  A more robust feature list can particulary improve the Decision
Tree model, as it is intended to find critical features among a large list.
     

**REFERENCES:**
The data used for classification was from Mass, Andrew L., et al.  Learning Word
Vectors for Sentiment Analysis.  Proceedings of the 49th Annual Meeting of the
Association for Computational Linguistics:  Human Language Technologies.
June 2001, Portland, OR.  p142 - 150.
