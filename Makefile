my_app: symnmf.o matrix_util.h
    gcc -o my_app main.o foo.o bar.o

main.o: main.c
    gcc -c main.c

matrix_util.o: matrix_util.c
    gcc -c matrix_util.c

bar.o: bar.c
    gcc -c bar.c