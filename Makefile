CC		= nvcc
CFLAGS  = -lcublas -g

SRCSC		= $(shell find . -name "*.cu")
OBJSC		= $(patsubst %.cu,%.out,$(SRCSC))

# All Phony Targets
.PHONY : everything clean

# Default starting position
everything : $(OBJSC)

clean :
	rm -f $(OBJSC) $(OSCARSKERNEL)

$(OBJSC) : %.out : %.cu
	$(CC) -o $@ $< $(CFLAGS) 