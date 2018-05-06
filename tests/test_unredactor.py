import redactor
import joblib
from sklearn import tree

def test_extract_features():
    ef = redactor.extract_features("There is Matthew Beattie and Mary Beattie here.",'4356_10.txt')
    assert ef == ([[10,15],[10,12]],['Matthew Beattie', 'Mary Beattie'])


def test_unredact_file():
    clf = joblib.load('/projects/redactor/redactor/dtclf.pkl')
    newstr = redactor.unredact_file("A great actor is \xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfe I think.",'4356_10.txt',clf)
    assert newstr == "A great actor is John Wayne I think."
