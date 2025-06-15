#!/bin/bash
set -e  # Exit on error
apt-get update -q
apt-get install -y -qq espeak-ng libespeak-ng-dev ffmpeg
apt-get clean  # Clean up apt cache to reduce image size
pip install --no-cache-dir -r requirements.txt