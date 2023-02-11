# ImplementingSearch

This part of the repository is an edit and extended version of the repository https://github.com/SGSSGene/ImplementingSearch 

### Cloning (very cool)
To checkout the code you can run:
  + `git clone --recurse-submodules https://git.imp.fu-berlin.de/herzlef98/ifa-2022.git`


## How to build the software
```
$ # We are assuming you are in the terminal/console inside the repository folder
$ cd ifa-2022/advanced_algorithms/week1/
$ mkdir build # creates a folder for our build system
$ cd build
$ cmake ..    # configures our build system
$ make        # builds our software, repeat this command to recompile your software
$ ./bin/naive_search --reference ../data/hg38_partial.fasta.gz --query ../data/illumina_reads_40.fasta.gz       # calls the code in src/naive_search.cpp
$ ./bin/suffixarray_search --reference ../data/hg38_partial.fasta.gz --query ../data/illumina_reads_40.fasta.gz # calls the code in src/suffixarray_search.cpp
$ ./bin/suffixarray_search --reference ../data/hg38_partial.fasta.gz --query ../data/illumina_reads_40.fasta.gz --out ../benchmarks/out.csv # calls the benchmark for src/suffixarray_search.cpp

$ ./bin/fmindex_construct --reference ../data/hg38_partial.fasta.gz --index myIndex.index # creates an index, see src/fmindex_construct.cpp
$ ./bin/fmindex_search --index myIndex.index --query ../data/illumina_reads_40.fasta.gz   # searches by using the fmindex, see src/fmindex_search.cpp

$ ./bin/fmindex_pigeon_search --index myIndex.index --query ../data/illumina_reads_40.fasta.gz   # searches by using the fmindex, see src/fmindex_pigeon_search.cpp
```

## How to run
1. For suffixarray_search:
````
$ ./bin/suffixarray_search --reference ../data/hg38_partial.fasta.gz --query ../data/illumina_reads_40.fasta.gz --out ../benchmarks/out.csv # calls the benchmark for src/suffixarray_search.cpp
````
   + -out <PATH/FILENAME> for saving the benchmark results as CSV 
   + -i <NUMEBER_OF_ITERATIONS> can be the number of iterations for benchmarking determined 
   + -n <NUMBER_OF_QUERIES> the number of queries
   + -l searching with lcp values is supported. (Be careful - Memory overhead! Not Optimized! Only recommend for small (>=1000) number of queries.)
   + --lcp-out <PATH/FILENAME> can be writen to disk and with 
   + --lcp-in <PATH/FILENAME> can be saved lcp-values loaded again.

2. For fmindex_search:
````
$ --index ../data/hg38_partial.index --40 ../data/illumina_reads_40.fasta --60 ../data/illumina_reads_60.fasta --80 ../data/illumina_reads_80.fasta --100 ../data/illumina_reads_100.fasta --out ../benchmarks/test.csv
````
   + -out <PATH/FILENAME> for saving the benchmark results as CSV
   + -i <NUMEBER_OF_ITERATIONS> can be the number of iterations for benchmarking determined
   + -n <NUMBER_OF_QUERIES> the number of queries
   + --selection <REGEX> you can select specific benchmarks for example
     + [a-zA-Z]\*60[a-zA-Z]*        select all benchmarks using queries with length 60
     + [a-zA-Z]\*no_error[a-zA-Z]*  select all benchmarks searching with no error
     + [a-zA-Z]\*error_2[a-zA-Z]*   select all benchmarks searching with two errors 
