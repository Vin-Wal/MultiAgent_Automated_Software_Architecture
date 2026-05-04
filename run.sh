#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$DIR/.venv/bin/python"

NO_RAG=""
for arg in "$@"; do
  case $arg in
    --no-rag)  NO_RAG="--no-rag" ;;
    --reindex) echo "Clearing chroma_db ..."; rm -rf "$DIR/chroma_db" ;;
  esac
done

cd "$DIR"
"$VENV" run.py $NO_RAG
