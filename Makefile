symnmf: symnmf.c matrix_util.c parse_input.c
	gcc -g -ansi -Wall -Wextra -Werror -pedantic-errors symnmf.c matrix_util.c parse_input.c -lm -o symnmf