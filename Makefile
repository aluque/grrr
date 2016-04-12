OBJ = grrr.o misc.o

#CFLAGS    = -std=c99 -g -fPIC
CFLAGS    = -std=c99 -O3 -march=native -fPIC
DFLAGS    = -DCOMPILING_HOSTNAME=\"$(HOSTNAME)\"
LIBS = -lm
CC   = gcc
LINK = gcc

all:	libgrrr.so

%.o:	%.c
	$(CC) $(CFLAGS) $(DFLAGS) $(INCLUDE_DIRS) -o $@ -c $< 

grrr:   $(OBJ)
	$(LINK) $(OBJ) -o $@ $(LIBS)

libgrrr.so: $(OBJ)
	$(LINK) -shared $(OBJ) -o $@ $(LIBS)

clean:  
	rm *.o 2> /dev/null
