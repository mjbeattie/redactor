import redactor

def test_word_counter():
	total = redactor.word_counter("This is a string")
	assert total == 4

def test_redact_phone_numbers():
	newstr = redactor.redact_phone_numbers("This, (301) 555-1212 is a number to remove")
	assert newstr == "This, <phone redacted> is a number to remove"

def test_redact_dates():
	newstr = redactor.redact_dates("Please remove 11/20/2001 and November 11, 2001")
	assert newstr == "Please remove <date redacted> and <date redacted>"

def test_redact_email_addresses():
	newstr = redactor.redact_email_addresses("This, mjbeattie@ou.edu is an email to remove")
	assert newstr == "This, <email redacted> is an email to remove"

def test_redact_addresses():
	newstr = redactor.redact_addresses("This, 1212 Mockingbird Lane, Los Angeles, CA 90001 is an address to remove")
	assert newstr == "This, <address redacted> is an address to remove"

def test_redact_names():
	newstr = redactor.redact_names("This, Steve Rogers is a name to remove")
	assert newstr == "This, \xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfe\xfe is a name to remove"

def test_redact_concept():
	newstr = redactor.redact_concept("This, child, is a number to remove.  This, dog, is not.", "child")
	assert newstr == "<concept sentence redacted>  This, dog, is not."

def test_redact_gender():
	newstr = redactor.redact_gender("This, woman, is a gender to remove")
	assert newstr == "This, <gender redacted>  is a gender to remove"
