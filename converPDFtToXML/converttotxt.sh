#!/bin/bash
#echo $1
first=$1
output=${first%.pdf}
txt='.txt'
#echo $output$txt
python pdf2txt.py -o $output$txt -t html $1


