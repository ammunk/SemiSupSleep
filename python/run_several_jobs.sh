#!/bin/sh

for i in {1..100}
do
	qsub main_batch.sh
done
