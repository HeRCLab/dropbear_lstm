all: signal1.wavegen nn1.mlpx nn2.mlpx nn3.mlpx
.PHONY: all

signal1.wavegen:
	wavegen generate -f 10 -f 37 -f 78 -p 0 -p 1 -p 2 -a 1 -a 2 -a 3  -d 2 -s 5000 -o "$@"

nn1.mlpx:
	mlpx --seed 1 new --sizes 10,10,1 -A sigmoid --alpha 0.0001 -o "$@"

nn2.mlpx:
	mlpx --seed 1 new --sizes 10,10,10,1 -A sigmoid --alpha 0.0000001 -o "$@"

nn3.mlpx:
	mlpx --seed 1 new --sizes 10,10,10,1 -A identity --alpha 0.00001 -o "$@"

clean:
	rm -f *.wavegen *.mlpx
.PHONY: clean
