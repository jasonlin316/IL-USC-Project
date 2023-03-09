#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <float.h>



int main(int argc, char const *argv[]){


	struct timespec start, stop; 
	double time;

	int size = 2449029;
    int p1 = size/2;
	int nnz = 123718152;
	int feature_length = 100;
    int f2 = feature_length/2;
	int hidden_dim = 128;
	int output_dim = 47;

	/*load the normalized indices, indices store the column indice of the non-zero elements*/
	FILE* datafn_norm = fopen("coo_data.bin", "rb");
	if (datafn_norm == NULL){
		printf("the data.bin is not opened correctly!");
	}
	float* data_norm = malloc(nnz *sizeof(float));

	fread(data_norm, sizeof(float), nnz, datafn_norm);
	fclose(datafn_norm);

	FILE* indicesfn_norm = fopen("row.bin", "rb");
	if (indicesfn_norm == NULL){
		printf("the indices.bin is not opened correctly!");
	}

	int* row = malloc(nnz*sizeof(int));

	fread(row, sizeof(int), nnz, indicesfn_norm);
	fclose(indicesfn_norm);

	/*load the indptr, indptr store range of every row*/

	FILE* indptrfn_norm = fopen("col.bin", "rb");
	if (indptrfn_norm == NULL){
		printf("the indptr.bin is not opened correctly!");
	}

	int* col = malloc(nnz*sizeof(int));

	fread(col, sizeof(int), nnz, indptrfn_norm);
	fclose(indptrfn_norm);

	/*load node input X*/
	FILE* featfn = fopen("/home/jason/cluster-gcn/Amazon-2M/feats.bin", "rb");
	if (featfn == NULL){
		printf("the feats.bin is not opened correctly!");
	}
	float* feats = malloc(size * feature_length *sizeof(float));
    float* feats_p1 = malloc(p1 * feature_length *sizeof(float));

	fread(feats, sizeof(float), size * feature_length, featfn);
	fclose(featfn);

    for(int i=0;i<p1;i++)
    {
        for(int j=0;j<feature_length;j++)
        {
            feats_p1[i*feature_length+j] = feats[i*feature_length+j];
        }
    }


	/*intermediate result after (AX)*/
	float interAX[feature_length];
	float* AX = malloc(size * feature_length * sizeof(float));
    float* BX = malloc(size * feature_length * sizeof(float));
    float* accumulate = malloc(size * feature_length * sizeof(float));
	printf("serial version amazon\n");
	
	//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//
	if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}
    int cnt = 0;
	/*My code here*/
    int ptr = 0;
	for(int i = 0; i < size; i++){ //row by row
		for(int k = 0; k<f2; k++){
			interAX[k] = 0;
		}
        
        while(i==row[ptr])
        {
        
            cnt++;
            for(int k = 0; k<f2; k++)
            { //columnwise compute in one row
                interAX[k] += data_norm[ptr] * feats[col[ptr]*feature_length+k];
            } 
    
            ptr++;
        }
			
		for(int k = 0; k<f2;k++){
			AX[i*feature_length + k] = interAX[k];
            //BX[i*feature_length + k] = interAX[k]+1;
		}
	
	}


	if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}		
	time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
	printf("Time for aggregation in layer 1 = %f sec\n", time);	
    printf("nnz: %d \n", cnt);	

	return 0;
	
	//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//


}
