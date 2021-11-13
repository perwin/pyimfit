#!/usr/bin/env bash
#
# This is the current master script for generating PyImfit docs, suitable for commiting
# to GitHub and thus being processed on readthedocs.org

# For clarity/possible future use, I list what each Makefile invocation actually does
# (as a comment)

echo "** make clean..."
# sphinx-build -M clean "." "_build"
make clean
echo
echo "** converting Markdown files to ReST..."
./convert_md_to_rst.sh
echo
echo "** Making HTML (first round)..."
# sphinx-build -M html "." "_build"
make html

# generate simplified HTML files from the autodoc-generated HTML
#echo
echo "** Generating simplified API HTML files..."
./convert_apidoc_html.py ./_build/html/api_ref_local/descriptions.html ./api_ref/descriptions_base.html
./convert_apidoc_html.py ./_build/html/api_ref_local/fitting.html ./api_ref/fitting_base.html
./convert_apidoc_html.py ./_build/html/api_ref_local/useful.html ./api_ref/useful_base.html

echo
echo "** Making HTML (second round)..."
make html
