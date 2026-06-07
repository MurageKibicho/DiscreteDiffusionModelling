#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>
#include <stdbool.h>
#include "Dependencies/cJSON.h" 
#include "Dependencies/stb_ds.h" 
#define Kibicho_Index2D(x, y, cols) ((x) * (cols) + (y))

//Run: clear && gcc Safetensor.c Dependencies/cJSON.c -lm -o m.o && ./m.o
typedef struct kibicho_tensor_struct *KibichoTensor;
struct kibicho_tensor_struct
{
	int size;
	int dimensionCount;
	int foundKibichoTensor;
	size_t offsetStart;
	size_t offsetEnd;
	int *shape;
	int *strides;
	float *data;
};

KibichoTensor CreateKibichoTensor()
{
	KibichoTensor tensor = malloc(sizeof(struct kibicho_tensor_struct));
	tensor->dimensionCount = 0;
	tensor->size = 0;
	tensor->dimensionCount = 0;
	tensor->foundKibichoTensor = -1;
	tensor->offsetStart = 0;
	tensor->offsetEnd = 0;
	tensor->shape = NULL;
	tensor->strides = NULL;
	tensor->data = NULL;
	return tensor;
}

void SetKibichoTensor(cJSON *tensorData, char *tensorName, KibichoTensor tensor, unsigned char *weightData)
{
	//Reset KibichoTensor
	tensor->foundKibichoTensor = -1;tensor->dimensionCount = 0;arrsetlen(tensor->shape, 0);arrsetlen(tensor->strides, 0);
	cJSON *item = NULL;cJSON *offset = NULL;cJSON *dtype = NULL;cJSON *data_offsets = NULL;cJSON *shape = NULL;cJSON *eachShape = NULL;
	cJSON_ArrayForEach(item, tensorData)
	{
		dtype = cJSON_GetObjectItem(item, "dtype");
		data_offsets = cJSON_GetObjectItem(item, "data_offsets");
		shape = cJSON_GetObjectItem(item, "shape");
		if(dtype && data_offsets && shape)
		{
			if(strcmp(tensorName, item->string) == 0)
			{
				//printf("Key: %s\n", item->string);printf("  dtype: %s\n", dtype->valuestring);printf("  data_offsets: ");	
				cJSON_ArrayForEach(eachShape, shape)
				{
					arrput(tensor->shape, eachShape->valueint);
					tensor->dimensionCount += 1;
				}
				cJSON_ArrayForEach(offset, data_offsets)
				{
					tensor->foundKibichoTensor += 1;
					if(tensor->foundKibichoTensor == 0)
					{
						tensor->offsetStart = (size_t) offset->valuedouble;
					}
					else if(tensor->foundKibichoTensor == 1)
					{
						tensor->offsetEnd   = (size_t) offset->valuedouble;
					}
				}
				break;
			}
		}
	}
	
	//Set strides
	int stride = 1;
	tensor->size = 1;
	arrsetlen(tensor->strides, tensor->dimensionCount);
	for(int i = 0, j = tensor->dimensionCount - 1; i < tensor->dimensionCount && j > -1; i++, j--)
	{
		tensor->size *= tensor->shape[i];
		tensor->strides[j] = stride;
		stride *= tensor->shape[j];
	}
	
	//Set Weights
	tensor->data =  (float *) (weightData + tensor->offsetStart);
}

void PrintKibichoTensor(KibichoTensor tensor, bool printTensorData)
{
	if(tensor && tensor->foundKibichoTensor > -1)
	{
		printf("Size: %d, Dimensions: %d\nOffsets[%ld,%ld]\nShape[%d", tensor->size,tensor->dimensionCount, tensor->offsetStart,tensor->offsetEnd,tensor->shape[0]);
		for(int i = 1; i < tensor->dimensionCount; i++)
		{
			printf(",%d", tensor->shape[i]);
		}
		printf("]\nStrides[%d",tensor->strides[0]);
		for(int i = 1; i < tensor->dimensionCount; i++)
		{
			printf(",%d", tensor->strides[i]);
		}
		printf("]\n");
		if(tensor->dimensionCount == 2 && tensor->data != NULL && printTensorData == true)
		{
			int rows = tensor->shape[0];
			int cols = tensor->shape[1];
			printf("\nTensor Data [%d x %d]:\n", rows, cols);

			for(int i = 0; i < rows; i++)
			{
				printf("[");
				for(int j = 0; j < cols; j++)
				{
					int idx = i * tensor->strides[0] + j * tensor->strides[1];
					printf("%8.4f ", tensor->data[idx]);
				}
				printf("]\n");
			}
			printf("\n");
		}
	}
}
void DestroyKibichoTensor(KibichoTensor tensor)
{
	if(tensor && tensor->foundKibichoTensor > -1)
	{
		if(tensor->shape){arrfree(tensor->shape);}
		if(tensor->strides){arrfree(tensor->strides);}
		free(tensor);
	}
}

size_t kmProf_GetFileSize(char *fileName)
{
	FILE *fp = fopen(fileName, "rb");
	assert(fp != NULL);
	fseek(fp, 0L, SEEK_END);
	size_t currentFileSize = ftell(fp);rewind(fp);
	fclose(fp);
	return currentFileSize;
}

void kmProf_PrintIntArray(int length, int *array)
{
	for(int i = 0; i < length; i++)
	{
		printf("%3d, ", array[i]);
	}
	printf("\n");
}

void PrintFloatMatrix(int rows, int cols, float *matrix)
{
	for(int i = 0; i < rows; i++)
	{
		printf("[");
		for(int j = 0; j < cols; j++)
		{
			int index = Kibicho_Index2D(i,j,cols);
			printf("%.3f, ", matrix[index]);
		}
		printf("]\n");
	}
	printf("\n");
}

unsigned char *kmProf_LoadSafeTensorData(char *fileName, size_t *fileSizeHolder)
{
	size_t fileSize = kmProf_GetFileSize(fileName);
	FILE *fp = fopen(fileName, "rb");assert(fp != NULL);
	int fileNumber = fileno(fp);
	unsigned char *fileData = mmap(NULL,fileSize, PROT_READ, MAP_PRIVATE, fileNumber, 0);assert(fileData != NULL);
	assert(fileData != MAP_FAILED);
	fclose(fp);
	*fileSizeHolder = fileSize;
	return fileData;
}

size_t kmProf_GetHeaderLength(size_t fileSize, unsigned char *safeTensorData)
{
	assert(fileSize > 8);
	size_t headerLength = 0;
	for(int i = 7; i >= 0; i--)
	{
		headerLength <<= 8;
		headerLength += safeTensorData[i];
	}
	return headerLength;
}

float GetTensorItem_Float(KibichoTensor tensor, int Kibicho_Index2DLength, int *indices)
{
	if(tensor)
	{
		assert(Kibicho_Index2DLength == tensor->dimensionCount);
		assert(tensor->strides);assert(tensor->data);
		int Kibicho_Index2D = 0;
		for(int i = 0; i < tensor->dimensionCount; i++)
		{
			assert(indices[i] >= 0);
			assert(indices[i] < tensor->shape[i]);
			Kibicho_Index2D += indices[i] * tensor->strides[i];
		}
		assert(Kibicho_Index2D > -1);
		assert(Kibicho_Index2D < tensor->size);
		return tensor->data[Kibicho_Index2D];
	}
}

void KibichoTensor_LinearLayer(int inputRows, int inputCols, int outputCols, float *output, float *input, float *weight, float *bias, bool hasBias)
{	
	//Follows Pytorch Y = X * W^T + B
	//Access like transposed matrix
	for(int b = 0; b < inputRows; b++)
	{
		for(int o = 0; o < outputCols; o++)
		{
			float sum = hasBias ? bias[o] : 0.0f;
			for(int i = 0; i < inputCols; i++)
			{
				sum += input[b * inputCols + i] * weight[o * inputCols + i];
			}
			output[b * outputCols + o] = sum;
		}
	}
}

void KibichoTensor_ElementWiseSum(int rows, int cols, float *out, float *a, float *b)
{
	for(int i = 0; i < rows; i++)
	{
		for(int j = 0; j < cols; j++)
		{
			int index  = Kibicho_Index2D(i,j,cols);
			out[index] = a[index] + b[index];
		}
	}	
}

void KibichoTensor_EmbeddingLayer(int sequenceLength, int vocabularySize, int embeddingDimension, float *output, int *inputTokens, float *embedding)
{
	for(int i = 0; i < sequenceLength; i++)
	{
		int token = inputTokens[i];
		assert(token >= 0 && token < vocabularySize);
		for(int j = 0; j < embeddingDimension; j++)
		{
			int outputIndex = Kibicho_Index2D(i,j,embeddingDimension);
			int embedIndex  = Kibicho_Index2D(token,j,embeddingDimension);			
			output[outputIndex] = embedding[embedIndex];
		}	
	}
}

float SiLU(float x)
{
	return x / (1.0f + expf(-x));
}

void ApplySiLU(int length, float *array)
{
	for(int i = 0; i < length; i++)
	{
		array[i] = SiLU(array[i]);
	}
}

void ApplyDropout(int length, float *array, bool trainPhase)
{
	//TODO:Write dropout
	if(trainPhase == true)
	{
		for(int i = 0; i < length; i++)
		{
			//array[i] = SiLU(array[i]);
		}
	}
}

//TODO:Maybe write layer norm for CNN: https://dkleine.substack.com/p/understanding-layer-normalization 
void ApplyLayerNorm_PerToken_NoBias(int rows, int cols, float *output, float *input, float *weight)
{
	float epsilon  = 1e-5;
	for(int token = 0; token < rows; token++)
	{
		float tokenMean    = 0.0f;
		float tokenVariance = 0.0f;
		for(int col = 0; col < cols; col++)
		{
			int index = token * cols + col;
			tokenMean += input[index];
		}
		tokenMean /= cols;
		for(int col = 0; col < cols; col++)
		{
			int index = token * cols + col;
			float diff = input[index] - tokenMean;
			tokenVariance += diff * diff;
		}
		tokenVariance /= cols;
		float stdInverse = 1.0f / sqrtf(tokenVariance + epsilon);
		for(int col = 0; col < cols; col++)
		{
			int index = token * cols + col;
			float norm = (input[index] - tokenMean) * stdInverse;
			output[index] = norm * weight[col];
		}
	}
}

void ApplyModulation_InPlace(int rows, int cols, float *x, float *shift, float *scale)
{
	for(int r = 0; r < rows; r++)
	{
		for(int c = 0; c < cols; c++)
		{
			int idx = r * cols + c;  
			x[idx]  = x[idx] * (1.0f + scale[c]) + shift[c];
		}
	}	
}


void BidirectionalAttention(int batchSize,int sequenceLength,int neuralEmbeddingDimension,int attentionHeadCount,float *projectedOutput, float *input, float *qkvWeights, float *projectionWeights, bool trainPhase)
{
	assert(batchSize == 1);
	int T = sequenceLength;
	int C = neuralEmbeddingDimension;
	int H = attentionHeadCount;
	int D = C / H;

	float *qkv = calloc(T * 3 * C, sizeof(float));
	for(int t = 0; t < T; t++)
	{
		for(int o = 0; o < 3 * C; o++)
		{
			float sum = 0.0f;
			for(int i = 0; i < C; i++)
			{
				sum += input[t * C + i] * qkvWeights[o * C + i];
			}
			qkv[t * 3 * C + o] = sum;
		}
	}
	float *out = calloc(T * C, sizeof(float));
	for(int h = 0; h < H; h++)
	{
		for(int t = 0; t < T; t++)
		{
			float scores[T];
			//QK^T
			for(int t2 = 0; t2 < T; t2++)
			{
				float score = 0.0f;
				for(int d = 0; d < D; d++)
				{
					int q_idx = t * 3 * C + 0 * C + h * D + d;
					int k_idx = t2 * 3 * C + 1 * C + h * D + d;
					score += qkv[q_idx] * qkv[k_idx];
				}
				scores[t2] = score / sqrtf(D);
			}
			//softmax
			float max = scores[0];for(int i = 1; i < T; i++){if(scores[i] > max){max = scores[i];}}
			float sum = 0.0f;
			for(int i = 0; i < T; i++){scores[i] = expf(scores[i] - max);sum += scores[i];}
			for(int i = 0; i < T; i++){scores[i] /= sum;}
			//weighted sum V
			for(int d = 0; d < D; d++)
			{
				float val = 0.0f;
				for(int t2 = 0; t2 < T; t2++){int v_idx = t2 * 3 * C + 2 * C + h * D + d;val += scores[t2] * qkv[v_idx];}

				out[t * C + h * D + d] = val;
			}
		}
	}
	//output projection
	KibichoTensor_LinearLayer(T, C, C, projectedOutput, out, projectionWeights, NULL, false);			
	if(trainPhase == true)
	{
		assert(trainPhase == false);
		//TODO: include dropout
	}
	free(qkv);
	free(out);
}


