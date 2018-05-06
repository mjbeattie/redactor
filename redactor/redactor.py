"""
redactor.py
This set of functions redeacts text files from a directory and allows for storage
of the redacted files in another location.  It redacts several types of information
that can be picked by the user via flags.
Created on Mon Mar 26 19:57:34 2018

@author: Matthew J. Beattie, DSA5970
coding = utf-8
"""

import nltk
import argparse
import re
import sys
import os, glob

from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk import ne_chunk

stop = stopwords.words('english')

# Note:  If we import stem.* it overrides corpus.wordnet, so we only import porter
from nltk.stem.porter import *


def find_phone_numbers(string):
    r = re.compile(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})', )
    phone_numbers = r.findall(string)
    return [re.sub(r'\D', '', number) for number in phone_numbers]


def find_email_addresses(string):
    r = re.compile(r'[\w\.-]+@[\w\.-]+')
    return r.findall(string)


def find_addresses(string):
    r = re.compile(r'[0-9]+ .+, .+, [A-Z]{2} [0-9]{5}')
    return r.findall(string)


def find_names(document):
    names = []
    sentences = ie_preprocess(document)
    for tagged_sentence in sentences:
        for chunk in nltk.ne_chunk(tagged_sentence):
            if type(chunk) == nltk.tree.Tree:
                if chunk.label() == 'PERSON':
                    names.append(' '.join([c[0] for c in chunk]))
    return names


def ie_preprocess(stringin):
    """
    ie_preprocess
    :param document:
    :return: tokenized sentence list
    This function takes a string and returns a tagged and tokenized string
    back to the calling function.  It uses nltk to do the work.
    """
    stringin = ' '.join([i for i in stringin.split() if i not in stop])
    sentences = nltk.sent_tokenize(stringin)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences


def find_concept(document, concept):
    # Generate list of stemmed sentences from input document string
    stop = stopwords.words('english')
    stemmer = PorterStemmer()
    sentences = nltk.sent_tokenize(document)
    stemsentlist = []
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        synonyms = []
        docstems = []
        for word in words:
            if word not in stop:
                for syn in wordnet.synsets(word):  # Generate synonyms for words in document
                    for l in syn.lemmas():
                        synonyms.append(l.name())
                for syn in synonyms:
                    docstems.append(stemmer.stem(syn))
        stemsentlist.append(docstems)

    # Generate list of stemmed synonyms for input concept
    constems = []
    synonyms = []
    for syn in wordnet.synsets(concept):  # Generate synonyms
        for l in syn.lemmas():
            synonyms.append(l.name())
    for syn in synonyms:
        constems.append(stemmer.stem(syn))

    # Check for set intersection between stemmed document sentences and
    # concept stems.  If a match, add the original sentence to the list
    i = 0
    matches = []
    for sent in stemsentlist:
        if (set(sent) & set(constems)):
            matches.append(sentences[i])
        i += 1

    # Return list of sentences that match the concept
    return matches


def redact_phone_numbers(string):
    """
    redact_phone_numbers
    :param string:
    :return: newstring
    This function redacts all phone numbers of the standard format from a
    string extracted from a document.  It uses regular expressions to
    do the redaction.
    """
    newstring = re.sub(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})',
                       '<phone redacted>', string)
    return newstring


def redact_dates(stringin):
    """
    redact_dates
    :param string:
    :return: newstring
    This function redacts dates of several different types of formats from a
    string extracted from a document.  It uses regular expressions to do the redaction.
    """
    newstring = re.sub('[0-1]?\d[- /.][0-3]?\d[- /.][1-2]?\d?\d\d', "<date redacted>", stringin)
    newstring = re.sub('(18|19|20)\d\d[- /.][0-1]?\d[- /.][0-3]?\d', "<date redacted>", newstring)
    newstring = re.sub('[0-1]?d[- /.][0-3]?d[- /.][1-2]?\d?\d\d', "<date redacted>", newstring)
    newstring = re.sub(
        '(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)[- ][0-2]?[1-9][- ,][- ][1-2]\d\d\d',
        "<date redacted>", newstring)
    return newstring


def redact_email_addresses(string):
    """
    redact_email_addresses
    :param string:
    :return: newstring
    This function redacts all email addresses from a string extracted from
    a document.  It uses regular expresssions to do the redaction.
    """
    newstring = re.sub(r'[\w\.-]+@[\w\.-]+', '<email redacted>', string)
    return newstring


def redact_addresses(string):
    """
    redact_addresses
    :param string:
    :return: newstring
    This function redacts standard format addresses from a string that has been
    extracted from a document. It uses regular expressions to do the redaction.
    """
    newstring = re.sub(r'[0-9]+ .+, .+, [A-Z]{2} [0-9]{5}', '<address redacted>', string)
    return newstring


def redact_names(stringin):
    """redact_names
    This function passes through a string once to identify all formal
    names.  During a second pass, it replaces patterns that match the
    strings with a redaction text.
    """
    # Pass one:  create a list of formal names using nltk and PERSON chunk
    names = []
    sentences = ie_preprocess(stringin)
    for tagged_sentence in sentences:
        for chunk in nltk.ne_chunk(tagged_sentence):
            if type(chunk) == nltk.tree.Tree:
                if chunk.label() == 'PERSON':
                    names.append(' '.join([c[0] for c in chunk]))

    # Pass two:  use re to replace patterns that match the names in the names list
    newstring = stringin
    for name in names:
        replstr = ''
        for i in range(0,len(name)):
            replstr += u'\xfe'
        newstring = re.sub(name, replstr, newstring)

    # Return the redacted string
    return newstring


def redact_concept(stringin, concept):
    """redact_concept
    This function takes two arguments, a string to parse and a concept to look for.
    The function then generates a list of stemmed sentences from the string
    and generates a list of synomyms for the words in those sentences.  It next
    takes the input concept and similarly generates a list of stemmed synonyms.
    The function then compares these two sets, and for any sentence with overlap,
    adds it to a list.  Finally, the function uses re to replace the sentences with
    redaction blocks.
    """
    # Generate list of stemmed sentences from input document string
    stop = stopwords.words('english')
    stemmer = PorterStemmer()
    sentences = nltk.sent_tokenize(stringin)
    stemsentlist = []
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        synonyms = []
        docstems = []
        for word in words:
            if word not in stop:
                for syn in wordnet.synsets(word):  # Generate synonyms for words in document
                    for l in syn.lemmas():
                        synonyms.append(l.name())
                for syn in synonyms:
                    docstems.append(stemmer.stem(syn))
        stemsentlist.append(docstems)

    # Generate list of stemmed synonyms for input concept
    constems = []
    synonyms = []
    for syn in wordnet.synsets(concept):  # Generate synonyms
        for l in syn.lemmas():
            synonyms.append(l.name())
    for syn in synonyms:
        constems.append(stemmer.stem(syn))

    # Check for set intersection between stemmed document sentences and
    # concept stems.  If a match, add the original sentence to the list
    i = 0
    matches = []
    for sent in stemsentlist:
        if (set(sent) & set(constems)):
            matches.append(sentences[i])
        i += 1

    # Pass two:  use re to replace patterns that match the sentences in the matches list
    newstring = stringin
    for match in matches:
        newstring = re.sub(match, '<concept sentence redacted>', newstring)

    # Return the redacted string
    return newstring


def redact_gender(stringin):
    """redact_gender
    This function redacts gender specific nouns and possessives from a string
    by comparing patterns in the string to a list of defined words.
    """
    malelist = ['he', 'him', 'his', 'man', 'boy', 'lad', 'bloke', 'chap', 'gentleman', 'men', 'boys',
                'lads', 'blokes', 'chaps', 'gentlemen']
    femalelist = ['she', 'her', 'hers', 'woman', 'girl', 'chick', 'lady', 'women', 'girls', 'chicks',
                  'ladies']
    words = nltk.word_tokenize(stringin)
    newstr = stringin

    # Strip each word in stringin of possessive forms and check against gender lists
    # then replace the word in the tokenized list
    i = 0
    redactlist = []
    for word in words:
        wordroot = word.lower()
        wordroot = wordroot.replace("'s", "")
        wordroot = wordroot.replace("'", "")
        if (wordroot in malelist) or (wordroot in femalelist):
            redactlist.append(words[i])
            words[i] = '<gender redacted>'
        i += 1

    # Untokenize and recombine
    matches = []
    for word in redactlist:
        matches.append("\s" + word + "\s")
        matches.append("\s" + word + ",")
        matches.append("\s" + word + "\.")
        matches.append("\s" + word + "\?")
        matches.append("s" + word + "\!")
        for match in matches:
            newstr = re.sub(match, ' <gender redacted> ', newstr)

    return newstr


def word_counter(stringin):
    """
    word_counter
    :param stringin:
    :return: integer
    Counts the words in the string and returns the count
    """
    tokens = nltk.word_tokenize(stringin)
    return len(tokens)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # The --input flag defaults to the current directory unless another one is specified
    parser.add_argument("--input", type=str, help="The directory of *.txt files, e.g. ./ or " \
                                                  "/<directory>/  The directory must exist.", default='./')
    parser.add_argument("--names", help="Redact formal names", action="store_true")
    parser.add_argument("--genders", help="Redact gender identifiers", action="store_true")
    parser.add_argument("--dates", help="Redact dates", action="store_true")
    parser.add_argument("--addresses", help="Redact standard format addresses", action="store_true")
    parser.add_argument("--phones", help="Redact standard format phone numbers", action="store_true")
    parser.add_argument("--emails", help="Redact standard format emails", action="store_true")
    parser.add_argument("--concepts", type=str, help="Redact sentences with a concept")
    parser.add_argument("--output", type=str, help="Target directory for output files' \
                        '(must exist, enter as ./<dirname>/)")
    parser.add_argument("--stats", type=str, help="File name for stats file.  If not included " \
                                                  "the stats are printed to standard output.  File " \
                                                    " is written to the input directory.")
    args = parser.parse_args()

    # Create a list of files to redact.
    redfilelist = glob.glob(args.input+"*.txt")
    print("Redacting files: ", redfilelist)

    # Initialize statistics file, which will track stats for the whole glob
    statstring = ""

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

        # Create output file
        outname = re.sub(args.input, "", redfile) + ".redacted"

        try:
            # Open output file
            try:
                if args.output:
                    fileOut = open(os.path.join(args.output, outname), 'w', encoding='utf-8')
                else:
                    fileOut = open(outname, 'w', encoding='utf-8')
            except:
                print("Could not open output file, was directory name in form of ./<name>?  Exiting...")
                sys.exit(1)

            # Redact elements from the input string
            newstring = raw
            summarystats = []
            if args.phones:
                newstring = redact_phone_numbers(newstring)
                phonestats = re.findall("<phone redacted>", newstring)
                summarystats.append(["Phone numbers", len(phonestats)])
            if args.emails:
                newstring = redact_email_addresses(newstring)
                emailstats = re.findall("<email redacted>", newstring)
                summarystats.append(["Email addresses", len(emailstats)])
            if args.addresses:
                newstring = redact_addresses(newstring)
                addrstats = re.findall("<address redacted>", newstring)
                summarystats.append(["Street addresses", len(addrstats)])
            if args.names:
                newstring = redact_names(newstring)
                namestats = re.findall("\xfe+", newstring)
                summarystats.append(["Names", len(namestats)])
            if args.concepts:
                newstring = redact_concept(newstring, args.concepts)
                conceptstats = re.findall("<concept sentence redacted>", newstring)
                summarystats.append(["Concept sentences", len(conceptstats)])
            if args.genders:
                newstring = redact_gender(newstring)
                genderstats = re.findall("<gender redacted>", newstring)
                summarystats.append(["Gender identifiers", len(genderstats)])
            if args.dates:
                newstring = redact_dates(newstring)
                datestats = re.findall("<date redacted>", newstring)
                summarystats.append(["Dates", len(phonestats)])

            # Write redacted string to the output file
            print(newstring, file=fileOut)
            fileOut.close()

            # Print out statistics
            statstring = statstring + "\n\nSummary statistics for the input file " + redfile + "\n"
            statstring = statstring + "Initial number of words was " + str(word_counter(raw)) + "\n"
            statstring += "Number of words in redacted file is " + str(word_counter(newstring)) + "\n"
            for tuple in summarystats:
                statstring += tuple[0] + ":  " + str(tuple[1]) + " redactions\n"

        except Exception as e:
            print("Error occurred during file redaction, exiting...", e.args)
            sys.exit(1)

    # Print out stats to either a defined file or standard out
    try:
        if args.stats:
            fileOut = open(args.input+args.stats, 'w', encoding='utf-8')
            print(statstring, file=fileOut)
            fileOut.close()
        else:
            print(statstring, file=sys.stdout)

    except Exception as e:
        print("Error writing stats, exiting...", e.args)
        sys.exit(1)
