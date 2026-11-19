#!/bin/sh
set -eu

mkdir -p /app/data/corpus /app/data/parsed /app/data/tmp_uploads /home/most/.cache/huggingface
chown -R most:most /app/data /home/most/.cache

exec runuser -u most -- "$@"
