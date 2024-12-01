DUMPS=$(wildcard *.dump)
DUMPS_OBJ=$(patsubst %.dump,%.o,$(DUMPS))

.PHONY: all clean

all: out

out: 3mm.c 3mm.h $(DUMPS_OBJ)
	gcc --std=gnu11 -O3 -funroll-loops -march=native -g $(CFLAGS) -Wall 3mm.c $(DUMPS_OBJ) -o out

%.o: %.dump
	objcopy -I binary -O elf64-x86-64 --binary-architecture i386:x86-64 $< $@


clean:
	rm out $(DUMPS_OBJ)
