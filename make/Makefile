src=$(wildcard *.c ./sub/*.c)
dir=$(notdir $(src))
obj=$(patsubst %.c,%.o,$(dir))
obj2=$(dir:%.c=%.o)

all:
	@echo $(src)
	@echo $(dir)
	@echo $(obj)
	@echo $(obj2)
	@echo "end"

prepare:
	touch l1_a.c l1_b.c
	mkdir -p sub
	touch sub/l2_a.c sub/l2_b.c

clean:
	rm -rf l1_a.c l1_b.c sub
