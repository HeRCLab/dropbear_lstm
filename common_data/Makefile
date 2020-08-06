all: signal1.wavegen
.PHONY: all

signal1.wavegen:
	wavegen generate -f 10 -f 37 -f 78 -p 0 -p 1 -p 2 -a 1 -a 2 -a 3  -d 2 -s 5000 > $@

clean:
	rm -f *.wavegen
.PHONY: clean
