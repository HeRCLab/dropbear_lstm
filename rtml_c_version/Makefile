CC = gcc
C_OPTS = $(shell mlpx-config --cflags)
LINK_OPTS = -lm $(shell mlpx-config --libs)
sources = $(wildcard *.c)
headers = $(wildcard *.h)
objects = $(patsubst %.c,%.o,$(sources))

all: debug

debug: online_training

online_training: $(objects)
	$(CC) $(C_OPTS) -g $(objects) $(LINK_OPTS) -o online_training

%.o: %.c $(headers)
	$(CC) $(C_OPTS) -g -c $(LINK_OPTS) $< -o $@

.PHONY: clean

clean:
	rm -f *.o online_training
