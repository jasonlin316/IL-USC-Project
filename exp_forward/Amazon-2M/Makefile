
all: prefetch no_prefetch merge no_merge

prefetch: prefetch.c
	gcc -pthread -o prefetch prefetch.c

no_prefetch: no_prefetch.c
	gcc -pthread -o no_prefetch no_prefetch.c

merge: merge.c
	gcc -pthread -o merge merge.c

no_merge: no_merge.c
	gcc -pthread -o no_merge no_merge.c