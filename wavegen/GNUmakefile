include ./opinionated.mk

test:
> go test -timeout 30s ./pkg/...
.PHONY: test

install:
> go install ./cmd/hdr
.PHONY: install

fmt:
> go fmt ./pkg/...
.PHONY: fmt

lint:
> golint ./pkg/...
.PHONY: lint

coverage:
> go test -timeout 30s -coverprofile /dev/null ./pkg/...
.PHONY: coverage

viewcoverage:
> go test -timeout 30s -coverprofile cover.out ./pkg/...
> go tool cover -html=cover.out
.PHONY: viewcoverage

clean:
> rm -f cover.out
> sh smoketest/cleanup.sh

