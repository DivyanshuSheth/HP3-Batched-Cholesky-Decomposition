#include<cuda.h>
#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include<cmath>
#define TILE_SIZE 4            // Tile size and block size, both are taken as 32
__device__ void store_full_row(float*,float*,int,int, int, int);
__device__ void load_full_row(float*,float*,int,int, int, int);
__device__ void store_full(float*,float*,int,int,int, int, int);
__device__ void load_full(float*,float*,int,int,int, int, int);
__device__ void store_lower(float*,float*,int,int,int, int, int);
__device__ void load_lower(float*,float*,int,int,int, int, int);
__device__ void potrf_tile(float*);
__device__ void trsm_tile(float*,int,int,int);
__device__ void syrk_tile(float*,float*,int,int,int);
__global__ void right_looking_launch_kernel(float*,int);
__device__ void store_zeros(float*,int);

__device__ void store_full_row(float* read_data,float* write_data,int i,int N, int M, int shared_size_single_matrix)
{
    int global_y;
    int global_x = i*blockDim.y + threadIdx.y;
    for(int j=0;j<N/TILE_SIZE;j++)
    {
        global_y = j*blockDim.z + threadIdx.z;
        write_data[global_y*N*M + global_x*M +  blockIdx.x * blockDim.x + threadIdx.x] = read_data[threadIdx.y + (TILE_SIZE+1)*global_y + threadIdx.x*shared_size_single_matrix];
    }
    __syncthreads();
}
__device__ void load_full_row(float* read_data,float* write_data,int i,int N, int M, int shared_size_single_matrix)
{
    int global_y;
    int global_x = i*blockDim.y + threadIdx.y;
    for(int j=0;j<N/TILE_SIZE;j++)
    {
        global_y = j*blockDim.z + threadIdx.z;
        write_data[threadIdx.y + (TILE_SIZE+1)*global_y + threadIdx.x*shared_size_single_matrix] = read_data[global_y*N*M + global_x*M +  blockIdx.x * blockDim.x + threadIdx.x];
        // printf("%d, %d\n", threadIdx.y + (TILE_SIZE+1)*global_y + threadIdx.x*shared_size_single_matrix, global_y*N*M + global_x*M + threadIdx.x);
    }
    __syncthreads();
}
__device__ void store_full(float* read_data,float* write_data,int i,int j,int N, int M, int shared_size_single_matrix)
{
    int global_y = j*blockDim.z + threadIdx.z;
    int global_x = i*blockDim.y + threadIdx.y;
    write_data[global_y*N*M + global_x*M +  blockIdx.x * blockDim.x + threadIdx.x] = read_data[threadIdx.y + (TILE_SIZE+1)*threadIdx.z + threadIdx.x*shared_size_single_matrix];
    __syncthreads();
}
__device__ void load_full(float* read_data,float* write_data,int i,int j,int N, int M, int shared_size_single_matrix)
{
    int global_y = j*blockDim.z + threadIdx.z;
    int global_x = i*blockDim.y + threadIdx.y;
    write_data[threadIdx.y + (TILE_SIZE+1)*threadIdx.z + threadIdx.x*shared_size_single_matrix] = read_data[global_y*N*M + global_x*M +  blockIdx.x * blockDim.x + threadIdx.x];
    __syncthreads();
}
__device__ void store_lower(float* read_data,float* write_data,int i,int j,int N, int M, int shared_size_single_matrix)
{
    int global_y = j*blockDim.z + threadIdx.z;
    int global_x = i*blockDim.y + threadIdx.y;
    // printf("%f is at %d\n", read_data[threadIdx.y + (TILE_SIZE+1)*threadIdx.z + threadIdx.x*shared_size_single_matrix], threadIdx.y + (TILE_SIZE+1)*threadIdx.z + threadIdx.x*shared_size_single_matrix);
    if(threadIdx.z >= threadIdx.y)
        write_data[global_y*N*M + global_x*M +  blockIdx.x * blockDim.x + threadIdx.x] = read_data[threadIdx.y + (TILE_SIZE+1)*threadIdx.z + threadIdx.x*shared_size_single_matrix];
    else
        write_data[global_y*N*M + global_x*M +  blockIdx.x * blockDim.x + threadIdx.x] = 0.0;
    __syncthreads();
}
__device__ void load_lower(float* read_data,float* write_data,int i,int j,int N, int M, int shared_size_single_matrix)
{
    int global_y = j*blockDim.z + threadIdx.z;
    int global_x = i*blockDim.y + threadIdx.y;
    if(threadIdx.z >= threadIdx.y)
        write_data[threadIdx.y + (TILE_SIZE+1)*threadIdx.z + threadIdx.x*shared_size_single_matrix] = read_data[global_y*N*M + global_x*M +  blockIdx.x * blockDim.x + threadIdx.x];
    else
        write_data[threadIdx.y + (TILE_SIZE+1)*threadIdx.z + threadIdx.x*shared_size_single_matrix] = 0.0;
    __syncthreads();
}
__device__ void potrf_tile(float* t_A)
{
    int t_x = threadIdx.y;
    int t_y = threadIdx.z;
    __shared__ float temp2;
    for(int k=0;k<TILE_SIZE;k++)
    {
        if(t_x==t_y && t_x==k)
        {
            t_A[k*(TILE_SIZE+1) + k] = sqrtf(t_A[k*(TILE_SIZE+1) + k]);
            temp2 = t_A[k*(TILE_SIZE+1) + k];
        }
        __syncthreads();
        if(t_x<t_y && t_x == k)
        {
            t_A[t_y*(TILE_SIZE+1) + k]/= temp2;
        }
        __syncthreads();
        if(k<t_y && k<t_x && t_x<=t_y)
        {
            t_A[t_y*(TILE_SIZE+1) + t_x]-= t_A[t_x*(TILE_SIZE+1) + k]*t_A[t_y*(TILE_SIZE+1) + k];
        }
        __syncthreads();
    }
}
__device__ void trsm_tile(float *row_data,int i,int j,int N)
{
    int global_y = j*blockDim.z + threadIdx.z;
    int global_x = i*blockDim.y + threadIdx.y;
    int t_x = threadIdx.y;
    int t_y = threadIdx.z;
    for(int s=0;s<TILE_SIZE;s++)
    {
	if(t_x==s)
        {
	    row_data[global_y*(TILE_SIZE+1) + t_x]/= row_data[global_x*(TILE_SIZE+1) + t_x];
	}
	__syncthreads();
	if(t_x > s)
        {
	    row_data[global_y*(TILE_SIZE+1) + t_x]-= row_data[global_x*(TILE_SIZE+1) +  s]*row_data[global_y*(TILE_SIZE+1) + s];
	}
	__syncthreads();
    }
}
__device__ void syrk_tile(float* row_data,float* edit_data,int i,int j,int N) 
{
    int global_y = j*blockDim.z + threadIdx.z;
    int global_x = i*blockDim.y + threadIdx.y;
    int t_y = threadIdx.z;
    int t_x = threadIdx.y;
    float valueToSubtract = 0.0;
    for(int r=0;r<TILE_SIZE;r++)
    {
        valueToSubtract+= row_data[r + global_y*(TILE_SIZE+1)]*row_data[r + global_x*(TILE_SIZE+1)];
    }
    edit_data[t_y*(TILE_SIZE+1) + t_x]-= valueToSubtract;
    __syncthreads();
}

__device__ void store_zeros(float* write_data,int N, int M)
{
    int t_y = threadIdx.z;
    int t_x = threadIdx.y;
    int i,j;
    for(i=0;i<N/TILE_SIZE-1;i++)
    {
        for(j=i+1;j<N/TILE_SIZE;j++)
        {
            int global_x = j*blockDim.z + threadIdx.z;
            int global_y = i*blockDim.y + threadIdx.y;
            write_data[global_y*N*M + global_x*M + blockIdx.x * blockDim.x +threadIdx.x]  = 0.0;

        }
            // A[j*blockDim.x + t_x + (i*blockDim.y + t_y)*N] = 0.0;
    }
    __syncthreads();
}

__global__ void right_looking_launch_kernel(float* read_data,int N, int M , int num_of_matrices_per_block, int shared_size_single_matrix) // N -> dim, M -> num of matrices per block
{

    int no_of_tiles = (N / TILE_SIZE) + (N % TILE_SIZE != 0); 
    
    int tx = threadIdx.x;
    float *rA1 = NULL;

    extern __shared__ float row_data[];
    // __shared__ float tile_data[TILE_SIZE*(TILE_SIZE+1)];                // Using TILE_SIZE+1 to avoid Band-conflict in Shared Memory
    int tile_data_index = M * (N*(TILE_SIZE+1) + TILE_SIZE*(TILE_SIZE+1) + 1);
    // __shared__ float* tile_data = &row_data[M * (N*(TILE_SIZE+1) + TILE_SIZE*(TILE_SIZE+1) + 1)];
    int shared_size_single_matrix_tile_data = TILE_SIZE * (TILE_SIZE + 1);


    int i,j,k;
    for(i=0;i<N/TILE_SIZE;i++)
    {
        load_lower(read_data,&row_data[tile_data_index],i,i,N, M, shared_size_single_matrix_tile_data);
        // printf("%d \n", tile_data_index + shared_size_single_matrix_tile_data * M);
        // if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
            // for (int z = tile_data_index; z < tile_data_index + shared_size_single_matrix_tile_data * M; z++) {
                // printf("%f is at %d\n", row_data[z], z);
            // }
        // }
        
        rA1 = &row_data[tile_data_index + tx*shared_size_single_matrix_tile_data];
        // printf("%d\n", tx*shared_size_single_matrix_tile_data);
        // potrf_tile(tile_data);
        potrf_tile(rA1);
        store_lower(&row_data[tile_data_index],read_data,i,i,N, M, shared_size_single_matrix_tile_data);
        load_full_row(read_data,row_data,i,N, M, shared_size_single_matrix);
        for(j=i+1;j<N/TILE_SIZE;j++)
        {
            trsm_tile(&row_data[tx*shared_size_single_matrix],i,j,N);
            for(k=i+1;k<j;k++)
            {
                load_full(read_data,&row_data[tile_data_index],k,j,N, M, shared_size_single_matrix_tile_data);
                rA1 = &row_data[tile_data_index + tx*shared_size_single_matrix_tile_data];
                // syrk_tile(row_data,tile_data,k,j,N);
                syrk_tile(&row_data[tx*shared_size_single_matrix],rA1,k,j,N);
                store_full(&row_data[tile_data_index],read_data,k,j,N, M, shared_size_single_matrix_tile_data);
            }
            load_full(read_data,&row_data[tile_data_index],k,j,N, M, shared_size_single_matrix_tile_data);
            syrk_tile(&row_data[tx*shared_size_single_matrix],&row_data[tile_data_index + tx*shared_size_single_matrix_tile_data],k,j,N);
            store_full(&row_data[tile_data_index],read_data,k,j,N, M, shared_size_single_matrix_tile_data);
        }
        store_full_row(row_data,read_data,i,N, M, shared_size_single_matrix);
    }
    store_zeros(read_data,N,M);
}


int main()
{
    // int n,N;
    // printf("Enter dimension (N) : ");
    // scanf("%d",&n);
    // if((n%TILE_SIZE)==0)
    //     N = n;
    // else
    //     N = (((int) (n/TILE_SIZE)) + 1)*TILE_SIZE;
    // size_t size = N*N*sizeof(float);
    // float *M = (float *)malloc(size);
    // if(M == NULL)
    // {
    //     fprintf(stderr,"Failed to allocate host vectors!\n");
    //     exit(EXIT_FAILURE);
    // }
    // int i,j;
    // printf("Enter input matrix: \n");
    // for(i=0;i<N;i++)
    // {
    //     for(j=0;j<N;j++)
    //     {
    //         if(i>=n || j>=n)
    //             M[i*N + j] = 1;     //Padding the matrix with 1
    //         else
    //             scanf("%f",&M[i*N + j]);
    //     }
    // }

    FILE *fptr;
    fptr = fopen("./dataset/size4_256matrices.txt", "r");
    int num_of_matrices, dim_of_matrix;
    fscanf(fptr, "%d", &num_of_matrices);
    fscanf(fptr, "%d", &dim_of_matrix);
    float read_element;
    float* h_A = NULL;
    int numElements = num_of_matrices * dim_of_matrix * dim_of_matrix;
    size_t size = numElements * sizeof(float);
    cudaDeviceProp devp;
    cudaGetDeviceProperties(&devp, 0);

    h_A = (float *)malloc(size);
    
    int global_id = 0;

    for (int matrix_index = 0; matrix_index < num_of_matrices; matrix_index++)
    {
        for (int row = 0; row < dim_of_matrix; row++)
        {
            for (int column = 0; column < dim_of_matrix; column++)
            {
                fscanf(fptr, "%f", &read_element);
                global_id = row * dim_of_matrix * num_of_matrices + column * num_of_matrices + matrix_index;
                h_A[global_id] = read_element;
                // printf("At pos %d we get %0.2f\n", global_id, h_A[global_id]);
                // printf("%0.2f \n ", h_A[global_id]);
            }
        }
    }
    printf("\nRead from the input file successfully!\n");
    fclose(fptr);

    printf("\nPrinting the host-side input array read from the input file:\n");
    for (int i = 0; i < numElements; i++) {    
        printf("%f ", h_A[i]);
    }
    printf("\n\n");



    // cudaError_t err = cudaSuccess;
    // float *read_data = NULL;
    // err = cudaMalloc((void **)&read_data,N*N*sizeof(float));
    // if(err != cudaSuccess)
    // {
    //     fprintf(stderr,"Failed to allocate matrix on the CUDA device! (error code %s)\n",cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }
    // printf("Coping the matrix from host memory to device memory\n");
    // err = cudaMemcpy(read_data,M,size,cudaMemcpyHostToDevice);
    // if(err != cudaSuccess)
    // {
    //     fprintf(stderr,"Failed to copy matrix from host to device (error code %s)\n",cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }
    // printf("Testing for matrix M [%dx%d]\n",N,N);

    cudaError_t err = cudaSuccess;

    float *d_A = NULL;

    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    else {
        printf("Copied the h_A to device side successfully!\n\n");
    }



    // dim3 grid(1,1,1);
    // dim3 block(TILE_SIZE,TILE_SIZE,1);
    // size_t shared_size = (N*(TILE_SIZE+1) + TILE_SIZE*(TILE_SIZE+1) + 1)*sizeof(float);
    // right_looking_launch_kernel<<<grid,block,shared_size>>>(read_data,N);
    // err = cudaMemcpy(M,read_data,size,cudaMemcpyDeviceToHost);
    // if(err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to copy the output matrix M from device to Host (error code %s)\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }
    
    // int num_of_matrices_per_block = num_of_matrices;
    int num_of_matrices_per_block = min(128/(TILE_SIZE * TILE_SIZE) , num_of_matrices);	
    dim3 grid((num_of_matrices) / num_of_matrices_per_block , 1, 1);	
    dim3 block(num_of_matrices_per_block, TILE_SIZE, TILE_SIZE);

    // dim3 grid(1, 1, 1);
    // dim3 block(num_of_matrices, TILE_SIZE, TILE_SIZE);
    // no of tiles in a column
    // int INPUT_SIZE = dim_of_matrix;
    // int no_of_tiles = (INPUT_SIZE / TILE_SIZE) + (INPUT_SIZE % TILE_SIZE != 0); // ceil of (INPUT_SIZE / TILE_SIZE)
    int N = dim_of_matrix;
    size_t shared_size = num_of_matrices * (N*(TILE_SIZE+1) + TILE_SIZE*(TILE_SIZE+1) + 1)*sizeof(float) + num_of_matrices_per_block * TILE_SIZE*(TILE_SIZE+1) * sizeof(float);
    
    right_looking_launch_kernel<<<grid,block,shared_size>>>(d_A, dim_of_matrix, num_of_matrices, num_of_matrices ,(num_of_matrices * (N*(TILE_SIZE+1) + TILE_SIZE*(TILE_SIZE+1) + 1))/num_of_matrices);
    //left_looking_kernel<<<grid, block, num_of_matrices_per_block * 1 * TILE_SIZE * TILE_SIZE * sizeof(float)>>>(d_A, dim_of_matrix, num_of_matrices ,1 * TILE_SIZE * TILE_SIZE);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        printf("kernel launch failed with error \"%s\".\n",
                cudaGetErrorString(cudaerr));
    }

    // if(TILE_SIZE == INPUT_SIZE)
    // {
    //     // printf("The if statement works.\n");
    //     left_looking_kernel<<<grid, block, num_of_matrices * 1 * TILE_SIZE * TILE_SIZE * sizeof(float)>>>(d_A, dim_of_matrix, num_of_matrices ,1 * TILE_SIZE * TILE_SIZE);
    // }

    // else if((no_of_tiles + 2) * TILE_SIZE * TILE_SIZE * sizeof(float) < devp.sharedMemPerBlock)
    // {
    //     //printf("The if statement works.\n");
    //     left_looking_kernel_less_mem<<<grid, block, num_of_matrices * 4 * TILE_SIZE * TILE_SIZE * sizeof(float)>>>(d_A, dim_of_matrix, num_of_matrices ,4 * TILE_SIZE * TILE_SIZE);
    //     // left_looking_kernel<<<grid, block,num_of_matrices * (no_of_tiles + 2) * TILE_SIZE * TILE_SIZE * sizeof(float)>>>(d_A, dim_of_matrix, num_of_matrices ,(no_of_tiles + 2) * TILE_SIZE * TILE_SIZE);
    // }
    // else
    // {
    //     left_looking_kernel_less_mem<<<grid, block, num_of_matrices * 4 * TILE_SIZE * TILE_SIZE * sizeof(float)>>>(d_A, dim_of_matrix, num_of_matrices ,4 * TILE_SIZE * TILE_SIZE);
    // }






    // printf("Printing output matrix\n");
    // for(i=0;i<n;i++)
    // {
    //     for(j=0;j<n;j++)
    //     {
    //         if(j<=i)
    //             printf("%f\t",M[i*N + j]);
    //         else
    //             printf("%f\t",0.0);
    //     }
    //     printf("\n");
    // }
    // err = cudaFree(read_data);
    // if(err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to free device matrix M (error code %s)\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }
    // err = cudaDeviceReset();
    // if(err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to deinitialize the CUDA device (error code %s)\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }
    // free(M);
    // printf("DONE!\n");

    err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
    
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    else {
        printf("\nCopied d_A to host side successfully!\n");
    }
    
    printf("\nPrinting the output of cudememcopyDeviceToHost, i.e. the host-side array returned from device side:\n");
    for (int i = 0; i < numElements; i++) {    
        printf("%f ", h_A[i]);
    }


    err = cudaFree(d_A);
    if(err != cudaSuccess)
    {
        fprintf(stderr, "\nFailed to free device matrix M (error code %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaDeviceReset();
    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the CUDA device (error code %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    FILE *fptr1;
    fptr1 = fopen("./output_r.txt", "w+");
    float write_element;
    fprintf(fptr1, "%d\n", num_of_matrices);
    fprintf(fptr1, "%d\n", dim_of_matrix);

    for (int matrix_index = 0; matrix_index < num_of_matrices; matrix_index++)
    {
        for (int row = 0; row < dim_of_matrix; row++)
        {
            for (int column = 0; column < dim_of_matrix; column++)
            {
                //write_element = h_A[matrix_index * dim_of_matrix * dim_of_matrix + row * dim_of_matrix + column];
                global_id = row * dim_of_matrix * num_of_matrices + column * num_of_matrices + matrix_index;
                write_element = h_A[global_id] ;
                fprintf(fptr1, "%0.2f ", write_element);
            }
         fprintf(fptr1,"\n");
        }
        fprintf(fptr1,"\n");
    }
    fclose(fptr1);
    free(h_A);
    printf("\n\nAll tasks completed successfully!\n\n");

    return 0;
}
