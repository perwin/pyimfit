#!/usr/bin/env python3

# This script takes as input an HTML file generated by Sphinx using autodoc commands
# and generates a new HTML file with just the documentation-related HTML (i.e., without
# the header, sidebar, footer, etc.).
#
# The result can then be incorporated into a Sphinx api_ref directory as "raw HTML".
#
# The reason behind this silliness is that Sphinx autodoc generates nice API reference HTML
# files via autodoc commands for PyImfit on my laptop (because Sphinx can import the pyimfit
# package), but it *doesn't* work on readthedocs.org (because readthedocs.org prevents
# installation of Python extension modules, and so Sphinx can't import pyimfit and the
# autodoc generation fails).
#
# Intended use (e.g., as encoded in the make_docs.sh shell script):
# On my laptop:
#    1. Generate HTML docs using sphinx
#       $ cd ~/coding/pyimfit/docs
#       $ ./convert_md_to_rst.sh && make html  [FIXME: make *local HTML files]
#    2. Use this script to generate simplified HTML files
#       $ convert_apidoc_html.py ~/coding/pyimfit/docs/_build/html/api_ref_local/descriptions.html \
#         ~/coding/pyimfit/docs/api_ref/descriptions_base.html
#       etc.

import os, sys, optparse
from bs4 import BeautifulSoup


header = """
<!DOCTYPE html>

<html xmlns="http://www.w3.org/1999/xhtml">
  <head>
  </head>

  <body>

"""

trailer = """

  </body>
</html>
"""


def ExtractUsefulText( input ):
    """Given a blob of HTML text, this extracts whatever is inside
    the <div section ...></div> block and returns it, minus its H1 header
    element, as regular text.
    """
    soup = BeautifulSoup(input, 'html.parser')
    # extract the "section" tag, then remove the H1 element inside it
    # sectionTag = soup.find('div', {'class' :'section'})
    sectionTag = soup.find('div', {'class' :'body'})
    try:
        sectionTag.h1.decompose()
    except AttributeError:
        print("No h1 tag in!")
    # get the text *inside* the <div section ...></div>, return as Unicode string
    extractedBytes = sectionTag.encode_contents()
    return extractedBytes.decode()

    
def WriteSimplifiedHTML( text, outputFile ):
	with open(outputFile, 'w') as outf:
		outf.write(header)
		outf.write(text)
		outf.write(trailer)


def main( argv ):

    if (len(argv) < 3):
        print("ERROR: you must supply input file and output file names")
        return None
    inputFile = argv[1]
    outputFile = argv[2]
    if not os.path.exists(inputFile):
        print("ERROR: cannot fined input file \"%s\"!" % filinputFilee1)
        return None

    print("*** ", inputFile)
    with open(inputFile) as inf:
        inputHTML = inf.read()
    usefulHTML = ExtractUsefulText(inputHTML)
    WriteSimplifiedHTML(usefulHTML, outputFile)
    print("Done.")


if __name__ == "__main__":
	main(sys.argv)
