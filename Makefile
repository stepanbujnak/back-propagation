all:
	gcc -Wall -std=c99 main.c nn.c -lm -o nn

clean:
	rm -f nn
