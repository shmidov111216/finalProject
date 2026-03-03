my_app: main.o foo.o bar.o foo_bar.h
    gcc -o my_app main.o foo.o bar.o

main.o: main.c
    gcc -c main.c

foo.o: foo.c
    gcc -c foo.c

bar.o: bar.c
    gcc -c bar.c