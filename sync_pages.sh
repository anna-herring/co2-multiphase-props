#!/bin/sh
# Copy the front end and its images into docs/ for GitHub Pages.
# Run after editing anything in static/.
set -e
cp static/index.html docs/index.html
cp static/*.png static/*.jpg docs/ 2>/dev/null || true
echo "docs/ updated from static/"
