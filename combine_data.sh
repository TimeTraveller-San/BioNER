#!/bin/bash

cat data/*/train.tsv >> train_all.tsv
cat data/*/devel.tsv >> devel_all.tsv
cat data/*/test.tsv >> test_all.tsv