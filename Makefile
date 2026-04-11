symnmf: symnmf.c matrix_util.c
	gcc -g -ansi -Wall -Wextra -Werror -pedantic-errors symnmf.c matrix_util.c -lm -o symnmf