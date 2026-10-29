#!/bin/sh
# Copy the front end into docs/ for GitHub Pages. Run after editing static/index.html.
set -e
cp static/index.html docs/index.html
echo "docs/index.html updated from static/index.html"
