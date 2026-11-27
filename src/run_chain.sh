#!/bin/zsh
# wait for stages A,B, then run the rest (energy stages wait for the download internally)
cd "$(dirname "$0")"
while pgrep -f "journal.py A B" >/dev/null; do sleep 30; done
../.venv/bin/python -u journal.py H G F E C > ../logs/journal_rest.log 2>&1
