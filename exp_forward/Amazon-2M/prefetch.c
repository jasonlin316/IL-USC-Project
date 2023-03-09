#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <pthread.h>
#include <math.h>
#include <float.h>
#include <immintrin.h>

struct range
{
	int start;
	int end;
};


int size = 2449029;
int nnz = 123718152;
int feature_length = 100;
int hidden_dim = 128;
int output_dim = 47;

int* indices;
int* indptr;
float* feats;
float* w1;
float* w2;
float* AX;
float* H1;
float* AH1;
float* result;
float* gold;


void * gcnforward(void *threadarg){

	struct range * threadrange;
	threadrange = (struct range *) threadarg;

	int start = threadrange->start;
	int end = threadrange->end;
	float interAX[feature_length];

 	int innz = 0;
	for(int i = start; i < end; i++){
		for(int k = 0; k<feature_length; k++){
			interAX[k] = 0;
		}
		innz += indptr[i + 1] - indptr[i];

		for(int j = indptr[i]; j<indptr[i + 1]; j++){
			int offset = indices[j]*feature_length;
			if( (j+1) < indptr[i + 1]) _mm_prefetch((const void*)&feats[indices[j+1]*feature_length], _MM_HINT_T0);
			for(int k = 0; k<feature_length; k++){
				interAX[k] = feats[offset + k];
			}
		}
		
	}	
	pthread_exit(NULL);
}


int main(int argc, char *argv[]){


	struct timespec start, stop; 
	double time;

	int Nthreads = 64;
	if(argc == 2) Nthreads = atoi(argv[1]);
	else Nthreads = 1;
	
	int interval = size/Nthreads;
	int rc;


	indices = (int *)malloc(nnz*sizeof(int));
	indptr = (int *)malloc((size + 1)*sizeof(int));
	feats = (float *)malloc(size * feature_length *sizeof(float));
	AX = (float *)malloc(size * feature_length *sizeof(float));
	H1 = (float *)malloc(size * hidden_dim *sizeof(float));

	/*load the indices, indices store the column indice of the non-zero elements*/

	/*load data*/

	FILE* featfn = fopen("feats.bin", "rb");
	if (featfn == NULL){
		printf("the feat.bin is not opened correctly!");
	}

	fread(feats, sizeof(float), size * feature_length, featfn);
	fclose(featfn);

	/*load indices*/
	FILE* indicesfn = fopen("indices.bin", "rb");
	if (indicesfn == NULL){
		printf("the indices.bin is not opened correctly!");
	}

	fread(indices, sizeof(int), nnz, indicesfn);
	fclose(indicesfn);

	/*load indptr*/
	FILE* indptrfn = fopen("indptr.bin", "rb");
	if (indptrfn == NULL){
		printf("the indptr.bin is not opened correctly!");
	}

	fread(indptr, sizeof(int), size + 1, indptrfn);
	fclose(indptrfn);


	printf("pthread version amazon with %d threads\n",Nthreads);

	pthread_t thread[Nthreads];
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);


	//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//
	if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}


	struct range irange[Nthreads];
	for(int i = 0;i < Nthreads; i++){
		if(i == Nthreads - 1){
			irange[i].start = i*interval;
			irange[i].end = size;
			rc = pthread_create(&thread[i], &attr, gcnforward, (void*)&irange[i]);
			if(rc){printf("creating error");exit(-1);}
		}
		else{
			irange[i].start = i*interval;
			irange[i].end = (i+1)*interval;
			rc = pthread_create(&thread[i], &attr, gcnforward, (void*)&irange[i]);
			if(rc){printf("creating error");exit(-1);}
		}

	}

	pthread_attr_destroy(&attr);
	for(int i=0; i < Nthreads ; i++){
		rc = pthread_join(thread[i], NULL);
		if(rc){printf("join error %d", rc);exit(-1);}
		//printf("Main: completed join with thread %d\n",i);
	}


	if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}		
	time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
	printf("Execution time:  = %f sec\n", time);	
	long long ops = feature_length*nnz + size*feature_length*feature_length*2;
	printf("Number of operations: %lli\n", ops);

	free(indices);
	free(indptr);
	free(feats);
	free(w1);
	free(w2);
	free(AX);
	free(H1);
	free(AH1);
	free(result);
	free(gold);
	return 0;
}
