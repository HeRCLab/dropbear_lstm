# How to Run

`go run main.go`

# Dependencies

Because to my knowledge, the only Golang library capable of live plotting is my
own work-in-progress bindings of implot, I'm using `go.mod` to override GIU
with my own development fork of it. The relevant `replace` can be removed once
the bindings are fully finished and merged into the upstream. Progress on the
bindings is tracked [here](https://github.com/AllenDang/giu/issues/53).
