.PHONY: all clean

all:
	gcc --std=gnu11 -O3 -funroll-loops -march=native -g -DEXTRALARGE_DATASET -DLAPTOP -Wall -fsanitize=address,undefined 3mm.c -o out

clean:
	rm out
