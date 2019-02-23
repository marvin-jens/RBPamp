bowtie-build nsRBNS_oligos_taliaferro_et_al.fa pool
bwa index nsRBNS_oligos_taliaferro_et_al.fa

#bowtie --fullref -p8 -m1 -l 16 --best -S pool --chunkmbs 512 $SRC/$READS |

export REF=nsRBNS_oligos_taliaferro_et_al.fa

export SRC1=/home/mjens/engaging/common/reads/nsRBNS_data/raw_reads/trial1/
export TRIAL1=$( find $SRC/*.fastq -printf "%f\n" )

export SRC2=/home/mjens/engaging/common/reads/nsRBNS_data/raw_reads/trial2/
export TRIAL2=$( find $SRC/*.fastq -printf "%f\n" )

for READS in $TRIAL1;
do {
    bwa aln -n 2 -t 8 -R 10 $REF $SRC1/$READS | bwa samse -n 1 -f /dev/stdout $REF /dev/stdin $SRC1/$READS | \
        samtools view -f 16 -buh /dev/stdin | \
        samtools sort -o bam/$READS.bam /dev/stdin
}
done;


for READS in $TRIAL2;
do {
    bwa aln -n 2 -t 8 -R 10 $REF $SRC2/$READS | bwa samse -n 1 -f /dev/stdout $REF /dev/stdin $SRC2/$READS | \
        samtools view -f 16 -buh /dev/stdin | \
        samtools sort -o bam/$READS.bam /dev/stdin
}
done;

for BAM in bam/*.bam;
do {
    samtools index $BAM
}
done;

# trial1 reads were 45nt long, trial2 reads 50nt
for READS in $TRIAL1;
do {
    python count.py --pos_min=87 --pos_max=90 bam/$READS.bam > $READS.counts &
}
done;

for READS in $TRIAL2;
do {
    python count.py --pos_min=82 --pos_max=85 bam/$READS.bam > $READS.counts
}
done;
```

for BAM in bam/*.bam;
do {
    echo $BAM
}
done;


