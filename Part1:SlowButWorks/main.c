#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#define STB_DS_IMPLEMENTATION
#include "Safetensors.h"
static const char itos[65] = {
    '\n', ' ', '!', '$', '&', '\'', ',', '-', '.', '3',
    ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
    'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
    'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
    'u', 'v', 'w', 'x', 'y', 'z'
};

void PrintDecode(int contextLength, int vocabularySize, int wrapModulo, int *data)
{
	for(int i = 0; i < contextLength; i++)
	{
		assert(data[i] >= 0);
		assert(data[i] <  vocabularySize);
		printf("%c", itos[data[i]]);
		if(i % wrapModulo == 0){printf("\n");}
	}
	printf("\n");
}
//clear && gcc main.c Dependencies/cJSON.c -lm -o m.o && ./m.o
float GeometricNoise(float timeStep)
{
	float sigmaMin = 1e-4;
	float sigmaMax = 20;
	float result = pow(sigmaMin, (1 - timeStep)) * pow(sigmaMax, timeStep);
	return result;
}

void TimestepEmbedding(float sigma, int frequencyEmbeddingDimension, float *frequencies, int maxPeriod)
{
	int half = frequencyEmbeddingDimension / 2;
	for(int i = 0; i < half; i++)
	{
		float freq = expf(-logf(maxPeriod) * ((float)i / half));
		float arg  = sigma * freq;
		//printf("%d: %.4f, %.4f\n", i, cosf(arg), sinf(arg));
		frequencies[i]        = cosf(arg);
		frequencies[i + half] = sinf(arg);
	}
	if(frequencyEmbeddingDimension % 2 == 1)
	{
		frequencies[frequencyEmbeddingDimension - 1] = 0.0f;
	}
}


void ForwardPass(int contextLength, int vocabularySize, float *finalOutput, int *input, float currentSigmaBar, cJSON *tensorData, unsigned char  *weightData)
{
	bool trainPhase = false;
	int batchSize = 1;assert(batchSize == 1);
	int attentionHeadCount = 6;
	int frequencyEmbeddingDimension = contextLength;
	int neuralEmbeddingDimension  = 384;
	int conditionDimension = 64;
	
	int maxPeriod = 10000;
	int *positions = calloc(contextLength,sizeof(int));for(int i = 0; i < contextLength; i++){positions[i] = i;}
	float *frequencyEmbeddingOut     = calloc(batchSize * conditionDimension, sizeof(float));	
	float *positionTokenEmbeddingOut = calloc(contextLength * neuralEmbeddingDimension, sizeof(float));
		
	{
		//Frequency embedding stage
		float *frequencies  = calloc(frequencyEmbeddingDimension, sizeof(float));
		float *embeddingLinearOut0 = calloc(batchSize * conditionDimension, sizeof(float));
		float *embeddingLinearOut1 = calloc(batchSize * conditionDimension, sizeof(float));
		KibichoTensor tensor_embeddingMLPWeight0 = CreateKibichoTensor();
		KibichoTensor tensor_embeddingMLPBias0   = CreateKibichoTensor();
		KibichoTensor tensor_embeddingMLPWeight1 = CreateKibichoTensor();
		KibichoTensor tensor_embeddingMLPBias1   = CreateKibichoTensor();
		
		SetKibichoTensor(tensorData, "sigma_map.mlp.0.weight", tensor_embeddingMLPWeight0, weightData);
		SetKibichoTensor(tensorData, "sigma_map.mlp.0.bias"  , tensor_embeddingMLPBias0  , weightData);
		SetKibichoTensor(tensorData, "sigma_map.mlp.2.weight", tensor_embeddingMLPWeight1, weightData);
		SetKibichoTensor(tensorData, "sigma_map.mlp.2.bias"  , tensor_embeddingMLPBias1  , weightData);
		//PrintKibichoTensor(tensor_embeddingMLPWeight1, false);
		//Find frequencies
		TimestepEmbedding(currentSigmaBar, frequencyEmbeddingDimension,frequencies, maxPeriod);
		
		//Pass frequencies through MLP: (1,256) Input->Linear(1,64)->SiLU(1,64)->Linear(1,64)
		assert(tensor_embeddingMLPWeight0->dimensionCount == 2);assert(frequencyEmbeddingDimension == tensor_embeddingMLPWeight0->shape[1]);assert(conditionDimension == tensor_embeddingMLPWeight0->shape[0]);
		KibichoTensor_LinearLayer(batchSize, frequencyEmbeddingDimension, conditionDimension, embeddingLinearOut0, frequencies, tensor_embeddingMLPWeight0->data, tensor_embeddingMLPBias0->data, true);
		ApplySiLU(conditionDimension, embeddingLinearOut0);
		assert(tensor_embeddingMLPWeight1->dimensionCount == 2);assert(conditionDimension == tensor_embeddingMLPWeight1->shape[1]);assert(conditionDimension == tensor_embeddingMLPWeight1->shape[1]);
		KibichoTensor_LinearLayer(batchSize, conditionDimension, conditionDimension, embeddingLinearOut1, embeddingLinearOut0, tensor_embeddingMLPWeight1->data, tensor_embeddingMLPBias1->data, true);		
		ApplySiLU(conditionDimension, embeddingLinearOut1);	
		//PrintFloatMatrix(batchSize, conditionDimension, embeddingLinearOut1);
		memcpy(frequencyEmbeddingOut, embeddingLinearOut1, batchSize * conditionDimension * sizeof(float));
		//Free embedding memory
		DestroyKibichoTensor(tensor_embeddingMLPWeight0);
		DestroyKibichoTensor(tensor_embeddingMLPBias0);
		DestroyKibichoTensor(tensor_embeddingMLPWeight1);
		DestroyKibichoTensor(tensor_embeddingMLPBias1);
		free(embeddingLinearOut0);
		free(embeddingLinearOut1);
		free(frequencies);
	}
	
	{
		float *positionEmbeddingOut = calloc(contextLength * neuralEmbeddingDimension, sizeof(float));
		float *tokenEmbeddingOut    = calloc(contextLength * neuralEmbeddingDimension, sizeof(float));
		//Token embedding and Position embedding stage
		KibichoTensor tensor_positionEmbedding = CreateKibichoTensor();
		KibichoTensor tensor_tokenEmbedding    = CreateKibichoTensor();
		SetKibichoTensor(tensorData, "transformer.wpe.weight", tensor_positionEmbedding, weightData);
		SetKibichoTensor(tensorData, "transformer.wte.weight", tensor_tokenEmbedding   , weightData);
		//PrintKibichoTensor(tensor_positionEmbedding, false);
		//PrintKibichoTensor(tensor_tokenEmbedding, false);
		assert(contextLength == tensor_positionEmbedding->shape[0]);assert(neuralEmbeddingDimension == tensor_positionEmbedding->shape[1]);
		KibichoTensor_EmbeddingLayer(contextLength, tensor_positionEmbedding->shape[0], tensor_positionEmbedding->shape[1], positionEmbeddingOut, positions, tensor_positionEmbedding->data);
		//PrintFloatMatrix(contextLength, neuralEmbeddingDimension, positionEmbeddingOut);
		
		assert(vocabularySize == tensor_tokenEmbedding->shape[0]);assert(neuralEmbeddingDimension == tensor_tokenEmbedding->shape[1]);
		KibichoTensor_EmbeddingLayer(contextLength, tensor_tokenEmbedding->shape[0], tensor_tokenEmbedding->shape[1], tokenEmbeddingOut, input, tensor_tokenEmbedding->data);
		//PrintFloatMatrix(contextLength, neuralEmbeddingDimension, tokenEmbeddingOut);
		
		KibichoTensor_ElementWiseSum(contextLength, neuralEmbeddingDimension, positionTokenEmbeddingOut, tokenEmbeddingOut, positionEmbeddingOut);
		ApplyDropout(contextLength*neuralEmbeddingDimension, positionTokenEmbeddingOut, trainPhase);
		//PrintFloatMatrix(contextLength, neuralEmbeddingDimension, positionTokenEmbeddingOut);

		DestroyKibichoTensor(tensor_positionEmbedding);
		DestroyKibichoTensor(tensor_tokenEmbedding);
		free(positionEmbeddingOut);
		free(tokenEmbeddingOut);
	}
	
	//Pass through Diffusion Blocks
	for(int headIndex = 0; headIndex < attentionHeadCount; headIndex++)
	{
		//x = tok_emb + pos_emb, c = frequencyEmbeddingOut
		//Ada LN phase
		char tensorName_adaLN_modulationBias[128];
		char tensorName_adaLN_modulationWeight[128];
		float *adaLN_out = calloc(batchSize * 6 * neuralEmbeddingDimension, sizeof(float));
		snprintf(tensorName_adaLN_modulationWeight, sizeof(tensorName_adaLN_modulationWeight), "transformer.h.%d.adaLN_modulation.weight",headIndex);
		snprintf(tensorName_adaLN_modulationBias,   sizeof(tensorName_adaLN_modulationBias)  , "transformer.h.%d.adaLN_modulation.bias"  ,headIndex);
		//printf("%s %s\n",tensorName_adaLN_modulationBias, tensorName_adaLN_modulationWeight);
		KibichoTensor tensor_adaLN_modulationBias   = CreateKibichoTensor();
		KibichoTensor tensor_adaLN_modulationWeight = CreateKibichoTensor();
		
		SetKibichoTensor(tensorData, tensorName_adaLN_modulationBias,   tensor_adaLN_modulationBias  , weightData);
		SetKibichoTensor(tensorData, tensorName_adaLN_modulationWeight, tensor_adaLN_modulationWeight, weightData);
		
		//PrintKibichoTensor(tensor_adaLN_modulationWeight, false);
		//PrintKibichoTensor(tensor_adaLN_modulationBias, false);
		assert(tensor_adaLN_modulationWeight->shape[1] == conditionDimension);assert(tensor_adaLN_modulationWeight->shape[0] == 6 * neuralEmbeddingDimension);
		KibichoTensor_LinearLayer(batchSize, conditionDimension, 6 * neuralEmbeddingDimension, adaLN_out, frequencyEmbeddingOut, tensor_adaLN_modulationWeight->data, tensor_adaLN_modulationBias->data, true);		
		assert(batchSize == 1);//TODO:Handle batchSize > 1
		float *shift_msa = adaLN_out + 0 * neuralEmbeddingDimension;
		float *scale_msa = adaLN_out + 1 * neuralEmbeddingDimension;
		float *gate_msa  = adaLN_out + 2 * neuralEmbeddingDimension;
		float *shift_mlp = adaLN_out + 3 * neuralEmbeddingDimension;
		float *scale_mlp = adaLN_out + 4 * neuralEmbeddingDimension;
		float *gate_mlp  = adaLN_out + 5 * neuralEmbeddingDimension;
		//PrintFloatMatrix(batchSize, neuralEmbeddingDimension, shift_msa);
		//PrintFloatMatrix(batchSize, neuralEmbeddingDimension, scale_msa);		
		
		DestroyKibichoTensor(tensor_adaLN_modulationBias);
		DestroyKibichoTensor(tensor_adaLN_modulationWeight);
		
		
		//Create x skip connection
		float *positionTokenEmbeddingOut_Skip = calloc(contextLength * neuralEmbeddingDimension, sizeof(float));		
		memcpy(positionTokenEmbeddingOut_Skip, positionTokenEmbeddingOut, contextLength * neuralEmbeddingDimension * sizeof(float));
		
		//Layer norm phase
		char tensorName_LN1_Weight[128];
		char tensorName_LN2_Weight[128];
		float *LN1_out = calloc(contextLength * neuralEmbeddingDimension, sizeof(float));
		float *LN2_out = calloc(contextLength * neuralEmbeddingDimension, sizeof(float));	
		snprintf(tensorName_LN1_Weight, sizeof(tensorName_LN1_Weight), "transformer.h.%d.ln_1.weight",headIndex);
		snprintf(tensorName_LN2_Weight, sizeof(tensorName_LN2_Weight), "transformer.h.%d.ln_2.weight",headIndex);
		
		KibichoTensor tensor_LN1_Weight   = CreateKibichoTensor();
		KibichoTensor tensor_LN2_Weight   = CreateKibichoTensor();
		SetKibichoTensor(tensorData, tensorName_LN1_Weight, tensor_LN1_Weight, weightData);
		SetKibichoTensor(tensorData, tensorName_LN2_Weight, tensor_LN2_Weight, weightData);
		//PrintKibichoTensor(tensor_LN1_Weight, false);		
		assert(tensor_LN1_Weight->dimensionCount == 1);
		assert(tensor_LN1_Weight->shape[0] == neuralEmbeddingDimension);
		ApplyLayerNorm_PerToken_NoBias(contextLength, neuralEmbeddingDimension, LN1_out, positionTokenEmbeddingOut, tensor_LN1_Weight->data);		
		//PrintFloatMatrix(contextLength, neuralEmbeddingDimension, LN1_out);
		
		assert(batchSize == 1);//TODO:Change modulation for batchSize > 1
		ApplyModulation_InPlace(contextLength, neuralEmbeddingDimension, LN1_out, shift_msa, scale_msa);
		//PrintFloatMatrix(contextLength, neuralEmbeddingDimension, LN1_out);
		
		//Attention phase
		char tensorName_c_attn_Weight[128];
		char tensorName_c_proj_Weight[128];
		snprintf(tensorName_c_attn_Weight, sizeof(tensorName_c_attn_Weight), "transformer.h.%d.attn.c_attn.weight",headIndex);
		snprintf(tensorName_c_proj_Weight, sizeof(tensorName_c_proj_Weight), "transformer.h.%d.attn.c_proj.weight",headIndex);
		KibichoTensor tensor_c_attn_Weight   = CreateKibichoTensor();
		KibichoTensor tensor_c_proj_Weight   = CreateKibichoTensor();
		SetKibichoTensor(tensorData, tensorName_c_attn_Weight, tensor_c_attn_Weight, weightData);
		SetKibichoTensor(tensorData, tensorName_c_proj_Weight, tensor_c_proj_Weight, weightData);
		assert(tensor_c_attn_Weight->shape[0] == 3 * neuralEmbeddingDimension);assert(tensor_c_attn_Weight->shape[1] == neuralEmbeddingDimension);
		assert(tensor_c_proj_Weight->shape[0] == neuralEmbeddingDimension);assert(tensor_c_proj_Weight->shape[1] == neuralEmbeddingDimension);
		BidirectionalAttention(batchSize, contextLength, neuralEmbeddingDimension, attentionHeadCount, positionTokenEmbeddingOut, LN1_out, tensor_c_attn_Weight->data,tensor_c_proj_Weight->data, false);
		//PrintKibichoTensor(tensor_c_attn_Weight, false);	
		//PrintFloatMatrix(contextLength, neuralEmbeddingDimension, positionTokenEmbeddingOut);
		
		ApplyLayerNorm_PerToken_NoBias(contextLength, neuralEmbeddingDimension, LN1_out, positionTokenEmbeddingOut, tensor_LN1_Weight->data);		
		//PrintFloatMatrix(contextLength, neuralEmbeddingDimension, LN1_out);
		BidirectionalAttention(batchSize, contextLength, neuralEmbeddingDimension, attentionHeadCount, positionTokenEmbeddingOut, LN1_out, tensor_c_attn_Weight->data,tensor_c_proj_Weight->data, false);				
		//PrintFloatMatrix(contextLength, neuralEmbeddingDimension, positionTokenEmbeddingOut);
		
		float *positionTokenEmbeddingOut_AddScale = calloc(contextLength * neuralEmbeddingDimension, sizeof(float));
		ApplyAddScale_InPlace(contextLength, neuralEmbeddingDimension, positionTokenEmbeddingOut, gate_msa, positionTokenEmbeddingOut_Skip);
		memcpy(positionTokenEmbeddingOut_AddScale, positionTokenEmbeddingOut, contextLength * neuralEmbeddingDimension * sizeof(float));
		//PrintFloatMatrix(batchSize, neuralEmbeddingDimension, gate_msa);
		//PrintFloatMatrix(contextLength, neuralEmbeddingDimension, positionTokenEmbeddingOut);
		assert(tensor_LN2_Weight->dimensionCount == 1);
		assert(tensor_LN2_Weight->shape[0] == neuralEmbeddingDimension);
		ApplyLayerNorm_PerToken_NoBias(contextLength, neuralEmbeddingDimension, LN2_out, positionTokenEmbeddingOut_AddScale, tensor_LN2_Weight->data);		
		//PrintFloatMatrix(contextLength, neuralEmbeddingDimension, LN2_out);
		ApplyModulation_InPlace(contextLength, neuralEmbeddingDimension, LN2_out, shift_mlp, scale_mlp);		
		//PrintFloatMatrix(contextLength, neuralEmbeddingDimension, LN2_out);
		
		//MLP Phase
		char tensorName_mlp_cfc[128];
		char tensorName_mlp_cproj[128];
		snprintf(tensorName_mlp_cfc, sizeof(tensorName_mlp_cfc), "transformer.h.%d.mlp.c_fc.weight",headIndex);
		snprintf(tensorName_mlp_cproj, sizeof(tensorName_mlp_cproj), "transformer.h.%d.mlp.c_proj.weight",headIndex);
		KibichoTensor tensor_mlp_cfc_Weight    = CreateKibichoTensor();
		KibichoTensor tensor_mlp_proj_Weight   = CreateKibichoTensor();
		SetKibichoTensor(tensorData, tensorName_mlp_cfc, tensor_mlp_cfc_Weight, weightData);
		SetKibichoTensor(tensorData, tensorName_mlp_cproj, tensor_mlp_proj_Weight, weightData);
		
		assert(tensor_mlp_cfc_Weight->shape[0] == 4 * neuralEmbeddingDimension);
		float *cfc_out = calloc(neuralEmbeddingDimension * 4 * neuralEmbeddingDimension, sizeof(float));
		KibichoTensor_LinearLayer(contextLength, neuralEmbeddingDimension, 4 * neuralEmbeddingDimension, cfc_out, LN2_out, tensor_mlp_cfc_Weight->data, NULL, false);				
		ApplyGeLU(contextLength * 4 * neuralEmbeddingDimension, cfc_out);
		//PrintFloatMatrix(contextLength, 4 * neuralEmbeddingDimension, cfc_out);
		assert(tensor_mlp_proj_Weight->shape[1] == 4 * neuralEmbeddingDimension);
		KibichoTensor_LinearLayer(contextLength, 4 * neuralEmbeddingDimension, neuralEmbeddingDimension, positionTokenEmbeddingOut, cfc_out, tensor_mlp_proj_Weight->data, NULL, false);				
		//PrintFloatMatrix(contextLength, neuralEmbeddingDimension, positionTokenEmbeddingOut);
		ApplyDropout(contextLength * neuralEmbeddingDimension, positionTokenEmbeddingOut, false);
		//We use positionTokenEmbeddingOut_Skip as a temp variable
		ApplyAddScale_InPlace(contextLength, neuralEmbeddingDimension, positionTokenEmbeddingOut, gate_mlp, positionTokenEmbeddingOut_AddScale);
		//PrintFloatMatrix(contextLength, neuralEmbeddingDimension, positionTokenEmbeddingOut);
				
		DestroyKibichoTensor(tensor_LN1_Weight);
		DestroyKibichoTensor(tensor_LN2_Weight);
		DestroyKibichoTensor(tensor_c_attn_Weight);
		DestroyKibichoTensor(tensor_c_proj_Weight);
		DestroyKibichoTensor(tensor_mlp_cfc_Weight);
		DestroyKibichoTensor(tensor_mlp_proj_Weight);
		free(positionTokenEmbeddingOut_Skip);free(LN1_out);free(LN2_out);free(adaLN_out);
		free(cfc_out);free(positionTokenEmbeddingOut_AddScale);
		//break;
	}

	KibichoTensor tensor_lnf   = CreateKibichoTensor();
	SetKibichoTensor(tensorData, "transformer.ln_f.weight", tensor_lnf, weightData);
	assert(tensor_lnf->dimensionCount == 1);
	assert(tensor_lnf->shape[0] == neuralEmbeddingDimension);
	//Inplace layer norm
	ApplyLayerNorm_PerToken_NoBias(contextLength, neuralEmbeddingDimension, positionTokenEmbeddingOut, positionTokenEmbeddingOut, tensor_lnf->data);		
	//PrintFloatMatrix(contextLength, neuralEmbeddingDimension, positionTokenEmbeddingOut);			
	DestroyKibichoTensor(tensor_lnf);
	
	//Final Layer
	{
		KibichoTensor tensor_adaLN_modulationBias   = CreateKibichoTensor();
		KibichoTensor tensor_adaLN_modulationWeight = CreateKibichoTensor();
		KibichoTensor tensor_linearBias       = CreateKibichoTensor();
		KibichoTensor tensor_linearWeight     = CreateKibichoTensor();
		KibichoTensor tensor_norm_final   = CreateKibichoTensor();
		SetKibichoTensor(tensorData, "lm_head.adaLN_modulation.bias", tensor_adaLN_modulationBias, weightData);
		SetKibichoTensor(tensorData, "lm_head.adaLN_modulation.weight", tensor_adaLN_modulationWeight, weightData);
		SetKibichoTensor(tensorData, "lm_head.linear.bias", tensor_linearBias, weightData);
		SetKibichoTensor(tensorData, "lm_head.linear.weight", tensor_linearWeight, weightData);
		SetKibichoTensor(tensorData, "lm_head.norm_final.weight", tensor_norm_final, weightData);
		//Inplace layer norm
		ApplyLayerNorm_PerToken_NoBias(contextLength, neuralEmbeddingDimension, positionTokenEmbeddingOut, positionTokenEmbeddingOut, tensor_norm_final->data);		
		//PrintFloatMatrix(contextLength, neuralEmbeddingDimension, positionTokenEmbeddingOut);			
		assert(tensor_adaLN_modulationWeight->shape[1] == conditionDimension);assert(tensor_adaLN_modulationWeight->shape[0] == 2 * neuralEmbeddingDimension);
		float *adaLN_out = calloc(batchSize * 2 * neuralEmbeddingDimension, sizeof(float));
		KibichoTensor_LinearLayer(batchSize, conditionDimension, 2 * neuralEmbeddingDimension, adaLN_out, frequencyEmbeddingOut, tensor_adaLN_modulationWeight->data, tensor_adaLN_modulationBias->data, true);		
		float *shift = adaLN_out + 0 * neuralEmbeddingDimension;
		float *scale = adaLN_out + 1 * neuralEmbeddingDimension;
		//PrintFloatMatrix(batchSize, neuralEmbeddingDimension, shift);
		//PrintFloatMatrix(batchSize, neuralEmbeddingDimension, scale);		
		assert(batchSize == 1);//TODO:Change modulation for batchSize > 1
		ApplyModulation_InPlace(contextLength, neuralEmbeddingDimension, positionTokenEmbeddingOut, shift, scale);
		//PrintFloatMatrix(contextLength, neuralEmbeddingDimension, positionTokenEmbeddingOut);			
		assert(tensor_linearWeight->shape[1] == neuralEmbeddingDimension);
		assert(tensor_linearWeight->shape[0] == vocabularySize);
		KibichoTensor_LinearLayer(contextLength, neuralEmbeddingDimension, vocabularySize, finalOutput, positionTokenEmbeddingOut, tensor_linearWeight->data, tensor_linearBias->data, true);
		//PrintFloatMatrix(contextLength, vocabularySize, finalOutput);	
		//PrintKibichoTensor(tensor_linearWeight, false);
			
		DestroyKibichoTensor(tensor_adaLN_modulationBias);
		DestroyKibichoTensor(tensor_adaLN_modulationWeight);
		DestroyKibichoTensor(tensor_linearBias);
		DestroyKibichoTensor(tensor_linearWeight);
		DestroyKibichoTensor(tensor_norm_final);
		free(adaLN_out);
	}
	//PrintFloatMatrix(contextLength, vocabularySize, finalOutput);	
	for(int row = 0; row < contextLength; row++)
	{
		int col = input[row];
		finalOutput[row * vocabularySize + col] = 0.0f;
	}
		
	
	free(frequencyEmbeddingOut);free(positionTokenEmbeddingOut);
	free(positions);
}

void Test()
{
	char *fileName  = "converted_safetensors/model_epoch_25.safetensors"; 
	size_t fileSize = 0; 
	unsigned char *safeTensorData = kmProf_LoadSafeTensorData(fileName, &fileSize);
	assert(safeTensorData != NULL);assert(fileSize > 8);assert(safeTensorData[8] == '{');
	
	size_t headerLength = kmProf_GetHeaderLength(fileSize, safeTensorData);
	//Move to weights section
	unsigned char  *weightData = (safeTensorData+8+headerLength);
	
	printf("FileSize : %ld bytes\n", fileSize);
	printf("HeaderSize : %ld bytes\n", headerLength);
	
	//Parse tensor data with cJSON
	cJSON *tensorData = cJSON_ParseWithLength(safeTensorData+8, headerLength);assert(tensorData != NULL);
	char *formatted_json = cJSON_Print(tensorData);assert(formatted_json != NULL);
	//printf("%s\n",formatted_json);

	int input[256] = {
	50, 60, 17, 20, 19, 25, 30, 26, 30, 5, 11, 36, 64, 23, 32, 26, 59, 26,
	29, 14, 17, 35, 17, 8, 54, 1, 27, 58, 58, 43, 14, 10, 32, 56, 47, 14,
	4, 38, 7, 56, 44, 29, 21, 10, 27, 31, 12, 13, 20, 26, 50, 20, 58, 43,
	59, 10, 8, 10, 24, 2, 25, 30, 38, 47, 53, 17, 12, 55, 17, 31, 56, 16,
	21, 39, 16, 38, 21, 56, 26, 29, 3, 32, 40, 29, 20, 6, 18, 15, 12, 29,
	54, 48, 24, 9, 8, 49, 37, 10, 61, 3, 20, 41, 40, 47, 19, 35, 20, 35,
	26, 62, 13, 9, 18, 37, 37, 56, 3, 58, 13, 43, 6, 1, 15, 59, 40, 47,
	38, 52, 6, 32, 36, 17, 55, 37, 53, 12, 45, 11, 18, 38, 42, 47, 41, 41,
	6, 36, 17, 19, 14, 14, 31, 31, 22, 58, 39, 24, 17, 9, 3, 26, 63, 60,
	61, 31, 27, 57, 21, 9, 31, 62, 4, 18, 24, 55, 54, 10, 64, 30, 43, 20,
	6, 45, 29, 19, 47, 36, 10, 27, 51, 25, 4, 5, 46, 63, 24, 3, 41, 16,
	30, 42, 35, 58, 56, 0, 10, 4, 55, 63, 53, 48, 57, 2, 19, 14, 63, 35,
	49, 51, 53, 37, 45, 62, 25, 32, 6, 26, 46, 53, 6, 63, 20, 51, 60, 10,
	38, 33, 11, 34, 22, 52, 4, 15, 64, 18, 1, 45, 16, 7, 4, 59, 44, 64,
	34, 17, 29, 45
	};
	int vocabularySize = 65;
	int contextLength = 256;
	int steps      = 128;
	float epsilon  = 1e-5;
	float stepSize = (1 - epsilon) / (float) steps;
	for(int timeStep = 0; timeStep < steps+1; timeStep++)
	{
		float timeStepInRange = 1 - (timeStep *(1.0 / (float)steps));
		float currentSigmaBar = GeometricNoise(timeStepInRange);
		float deltaSigma = currentSigmaBar;
		float *logScore = calloc(contextLength * vocabularySize, sizeof(float));
		float *transitionMatrix = calloc(contextLength * vocabularySize, sizeof(float));
		float *probabilities   = calloc(contextLength * vocabularySize, sizeof(float));
		
		//printf("%d: %.4f %.4f\n",timeStep, timeStepInRange, currentSigmaBar);
		if(timeStep < steps)
		{
			float nextSigmaBar = GeometricNoise(timeStepInRange - stepSize);
			deltaSigma         = currentSigmaBar - nextSigmaBar;
			//printf("%d: %.4f %.4f %.4f %.4f\n",timeStep, timeStepInRange, currentSigmaBar, nextSigmaBar, deltaSigma);
		}
		ForwardPass(contextLength, vocabularySize, logScore, input, currentSigmaBar, tensorData, weightData);
		ApplyExpf(contextLength * vocabularySize, logScore);
		StaggeredScore(contextLength, vocabularySize, deltaSigma, logScore);
		Transition(contextLength, vocabularySize, transitionMatrix, input, deltaSigma);
		KibichoTensor_ElementWiseMul(contextLength, vocabularySize, probabilities, logScore, transitionMatrix);
		//PrintFloatMatrix(contextLength, vocabularySize, probabilities);	
		SampleCategorical(contextLength, vocabularySize, input, probabilities);
		printf("Text at timeStep(%d):\n",timeStep);
		PrintDecode(contextLength, vocabularySize,80, input);

		free(logScore);free(transitionMatrix);
		free(probabilities);
		//break;
	}
	
	
	free(formatted_json);
	//Delete cJSON data structures
	cJSON_Delete(tensorData);
	//unmap memory
	assert(munmap(safeTensorData, fileSize) != -1);
	
}



int main()
{
	//srand(556);
	Test();
	return 0;
}
