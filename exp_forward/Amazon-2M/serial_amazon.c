#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <float.h>



int main(int argc, char const *argv[]){


	struct timespec start, stop; 
	double time;

	int size = 2449029;
	int nnz = 123718152;
	int feature_length = 100;
	int hidden_dim = 128;
	int output_dim = 47;

	/*load the normalized indices, indices store the column indice of the non-zero elements*/
	FILE* datafn_norm = fopen("/home/jason/cluster-gcn/Amazon-2M/data.bin", "rb");
	if (datafn_norm == NULL){
		printf("the data.bin is not opened correctly!");
	}
	float* data_norm = malloc(nnz *sizeof(float));

	fread(data_norm, sizeof(float), nnz, datafn_norm);
	fclose(datafn_norm);

	FILE* indicesfn_norm = fopen("/home/jason/cluster-gcn/Amazon-2M/indices.bin", "rb");
	if (indicesfn_norm == NULL){
		printf("the indices.bin is not opened correctly!");
	}

	int* indices_norm = malloc(nnz*sizeof(int));

	fread(indices_norm, sizeof(int), nnz, indicesfn_norm);
	fclose(indicesfn_norm);

	/*load the indptr, indptr store range of every row*/

	FILE* indptrfn_norm = fopen("/home/jason/cluster-gcn/Amazon-2M/indptr.bin", "rb");
	if (indptrfn_norm == NULL){
		printf("the indptr.bin is not opened correctly!");
	}

	int* indptr_norm = malloc((size + 1)*sizeof(int));

	fread(indptr_norm, sizeof(int), size + 1, indptrfn_norm);
	fclose(indptrfn_norm);

	/*load pre-trainned weight*/

	FILE* w1_file = fopen("/home/jason/cluster-gcn/Amazon-2M/w1.bin", "rb");
	if (w1_file == NULL){
		printf("the w1.bin is not opened correctly!");
	}

	float* w1 = malloc(feature_length * hidden_dim * sizeof(float));

	fread(w1, sizeof(float), feature_length * hidden_dim, w1_file);
	fclose(w1_file);

	FILE* w2_file = fopen("/home/jason/cluster-gcn/Amazon-2M/w2.bin", "rb");
	if (w2_file == NULL){
		printf("the w2.bin is not opened correctly!");
	}

	float* w2 = malloc(output_dim * hidden_dim * sizeof(float));

	fread(w2, sizeof(float), output_dim * hidden_dim, w2_file);
	fclose(w2_file);

	/*load node input X*/
	FILE* featfn = fopen("/home/jason/cluster-gcn/Amazon-2M/feats.bin", "rb");
	if (featfn == NULL){
		printf("the feats.bin is not opened correctly!");
	}
	float* feats = malloc(size * feature_length *sizeof(float));

	fread(feats, sizeof(float), size * feature_length, featfn);
	fclose(featfn);


	/*intermediate result after (AX)*/
	float interAX[feature_length];
	float* AX = malloc(size * feature_length * sizeof(float));

	printf("serial version amazon\n");
	
	//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//
	if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}

	/*My code here*/

	for(int i = 0; i < size; i++){ //row by row
		for(int k = 0; k<feature_length; k++){
			interAX[k] = 0;
		}
		for(int j = indptr_norm[i]; j<indptr_norm[i + 1]; j++){
			for(int k = 0; k<feature_length; k++){ //columnwise compute in one row
				interAX[k] += data_norm[j] * feats[indices_norm[j]*feature_length+k];
			}
		}
		for(int k = 0; k<feature_length;k++){
			AX[i*feature_length + k] = interAX[k];
		}

	}

	if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}		
	time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
	printf("Time for aggregation in layer 1 = %f sec\n", time);	

	for(int k = 0; k<5;k++){
			printf("first row: %f\n",AX[k]);
		}
    
    for(int k = 0; k<5;k++){
			printf("last row: %f\n",AX[(size-1)*feature_length+k]);
		}
	return 0;
	/*time for transformation*/
	float* H1 = malloc(size * hidden_dim * sizeof(float));
	float local_c = 0;
	
	//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//
	if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}

	for(int i = 0; i<size; i++){
		for(int j =0; j<hidden_dim;j++)
		{

			local_c = 0;
			for(int k=0;k<feature_length; k++){
				local_c+=AX[i*feature_length + k]*w1[ k*hidden_dim + j];
			}
			if(local_c < 0) local_c = 0;
			H1[i*hidden_dim + j] = local_c;

		}
	}
	
	if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}		
	time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
	printf("Time for transformation in layer 1  = %f sec\n", time);	

	/* Second Layer */
	printf("start layer 2\n");

	float interAH1[hidden_dim];
	float* AH1 = malloc(size * hidden_dim * sizeof(float));
	float* result = malloc(size*output_dim*sizeof(float));
	int* result_class = malloc(size*sizeof(int));

	printf("feature aggregation...\n");

	if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}

	for(int i = 0; i < size; i++){ //row by row
		for(int k = 0; k<hidden_dim; k++){
			interAH1[k] = 0;
		}
		for(int j = indptr_norm[i]; j<indptr_norm[i + 1]; j++){
			for(int k = 0; k<hidden_dim; k++)
			{ //columnwise compute in one row
				interAH1[k] += data_norm[j] * H1[indices_norm[j]*hidden_dim+k];
			}
		}
		for(int k = 0; k<hidden_dim;k++){
			AH1[i*hidden_dim + k] = interAH1[k];
		}
	}

	if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}		
	time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
	printf("Time for aggregation in layer 2  = %f sec\n", time);	

	if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}
	for(int i = 0; i<size; i++){
		for(int j =0; j<output_dim;j++){

			local_c = 0;

			for(int k=0;k<hidden_dim; k++){
				local_c+=AH1[i*hidden_dim + k]*w2[k*output_dim + j];
			}

			result[i*output_dim + j] = local_c; //No ReLu
		}
	}
	if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}		
	time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
	printf("Time for transformation in layer 2  = %f sec\n", time);	

	printf("finding max...\n");
	if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}
	for(int i = 0; i<size;i++)
	{
		float max = -FLT_MAX;
		int max_idx = 0;
		for(int j = 0; j < output_dim; j++)
		{
			if(result[i*output_dim+j] > max) 
			{
				max = result[i*output_dim+j];
				max_idx = j;
			}
		}
		result_class[i] = max_idx;
	}


	if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}		
	time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
	printf("Time for finding maximum  = %f sec\n", time);	



	free(data_norm);
	free(indices_norm);
	free(indptr_norm);
	free(feats);
	free(w1);
	free(w2);
	free(AX);
	free(H1);
	free(AH1);
	free(result);
	free(result_class);

}
