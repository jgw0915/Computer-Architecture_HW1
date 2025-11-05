.PHONY: all clean

CC := gcc
CFLAGS := -Wall -O2

# dir/name for each program
PROGS := \
  vector/q1-vector \
  uf8/q1-uf8 \
  uf8/uf8_encode \
  uf8/uf8_decode \
  uf8/uf8_clz \
  bfloat16/q1-bfloat16

# dir/name -> dir/output/name
BINS := $(foreach p,$(PROGS),$(dir $(p))output/$(notdir $(p)))

all: $(BINS)

# Template to compile one binary from dir/name.c
define COMPILE_template
$(dir $(1))output/$(notdir $(1)): $(1).c | $(dir $(1))output/
	$$(CC) $$(CFLAGS) -o $$@ $$<
endef

# Instantiate one explicit rule per program
$(foreach p,$(PROGS),$(eval $(call COMPILE_template,$(p))))

# Ensure output directory exists
%/output/:
	mkdir -p $@

clean:
	rm -f $(BINS)
