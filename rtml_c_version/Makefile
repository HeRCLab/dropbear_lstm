CC = gcc
C_OPTS = -Wall -Wextra -I $(CONNXR_PATH)/include -I $(CONNXR_PATH)/src/pb -I $(CONNXR_PATH)/src/operators
LINK_OPTS = -lm
sources = $(wildcard *.c)
objects = $(patsubst %.c,%.o,$(sources))

CONNXR_PATH = cONNXr
CONNXR_SRCS = $(CONNXR_PATH)/src/trace.c \
		$(CONNXR_PATH)/src/operators/operator_sets.c \
		$(CONNXR_PATH)/src/operators/operator_stub.c \
		$(CONNXR_PATH)/src/pb/protobuf-c.c \
		$(CONNXR_PATH)/src/pb/onnx.pb-c.c \
		$(CONNXR_PATH)/src/inference.c \
		$(CONNXR_PATH)/src/utils.c \
		$(wildcard $(CONNXR_PATH)/src/operators/resolve/onnx/*.c)
CONNXR_OBJS = $(patsubst %/,,$(patsubst %.c,%.o,$(CONNXR_SRCS)))

all: debug

$(CONNXR_PATH):
	git clone https://github.com/alrevuelta/cONNXr

debug: online_training

online_training: $(objects) $(CONNXR_OBJS)
	$(CC) $(C_OPTS) -g $(CONNXR_OBJS) main.c $(LINK_OPTS) -o online_training

%.o: %.c $(CONNXR_PATH)
	$(CC) $(C_OPTS) -g -c $(LINK_OPTS) $< -o $@

.PHONY: clean

clean:
	-rm *.o online_training
