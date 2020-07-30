CC = gcc
C_OPTS = $(shell mlpx-config --cflags)
LINK_OPTS = -lm $(shell mlpx-config --libs)
sources = $(wildcard *.c)
objects = $(patsubst %.c,%.o,$(sources))

all: debug

debug: online_training

online_training: $(objects)
	$(CC) $(C_OPTS) -g $(objects) $(LINK_OPTS) -o online_training

%.o: %.c
	$(CC) $(C_OPTS) -g -c $(LINK_OPTS) $< -o $@

.PHONY: clean

clean:
	-rm *.o online_training
