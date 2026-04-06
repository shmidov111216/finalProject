symnmf: symnmf.c matrix_util.c
	gcc -ansi -Wall -Wextra -Werror -pedantic-errors symnmf.c matrix_util.c -lm -o symnmf